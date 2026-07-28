import argparse
import gc
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _preparse_dataset() -> None:
    """Reads --dataset from sys.argv and exports it as an environment
    variable BEFORE importing src.config, which consumes it at import time.

    This allows generating the 3D grid for either the 2023 dataset
    (NETC/SNFH/ET) or the 2024 dataset (NETC/SNFH/ET/RC), since
    OUT_CHANNELS, ROI_SIZE, etc. depend on the selected dataset.
    """
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument("--dataset", choices=["2023", "2024"], default=None)
    known_args, _ = pre_parser.parse_known_args()

    if known_args.dataset is not None:
        os.environ["BRATS_DATASET"] = known_args.dataset


_preparse_dataset()


import matplotlib
matplotlib.use("Agg")


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage import measure


from src.config import (
    DATASET,
    ROI_SIZE,
    SPACING,
    IN_CHANNELS,
    OUT_CHANNELS,
    USE_CHECKPOINT,
)
from src.data.dataset import get_case_data
from src.data.transforms import get_test_transforms
from src.models.get_model import get_model
from src.utils.checkpoints import load_checkpoint


HAS_RC = OUT_CHANNELS >= 5

CLASS_VALUES: Dict[str, List[int]] = {
    "NETC": [1],
    "SNFH": [2],
    "ET": [3],
}

if HAS_RC:
    CLASS_VALUES["RC"] = [4]

CLASS_ORDER = ["NETC", "SNFH", "ET"] + (["RC"] if HAS_RC else []) + ["All"]
CLASS_DRAW_ORDER_ALL = ["SNFH"] + (["RC"] if HAS_RC else []) + ["NETC", "ET"]

CLASS_COLORS = {
    "NETC": "#ff595e",
    "SNFH": "#5ee35e",
    "ET": "#6c6bd6",
    "RC": "#fff45c",
}

CLASS_ALPHA = {
    "NETC": 0.92,
    "SNFH": 0.78,
    "ET": 0.92,
    "RC": 0.88,
}

CLASS_ALPHA_ALL = {
    "NETC": 0.88,
    "SNFH": 0.50,
    "ET": 0.82,
    "RC": 0.68,
}

REGION_VALUES: Dict[str, List[int]] = {
    "ET": [3],
    "TC": [1, 3],
    "WT": [1, 2, 3],
}

REGION_ORDER = ["ET", "TC", "WT"]

REGION_COLORS = {
    "ET": "#6c6bd6",
    "TC": "#ff595e",
    "WT": "#5ee35e",
}

REGION_ALPHA = {
    "ET": 0.92,
    "TC": 0.85,
    "WT": 0.55,
}

CLASS_CONFIG = {
    "values": CLASS_VALUES,
    "order": CLASS_ORDER,
    "draw_order_all": CLASS_DRAW_ORDER_ALL,
    "colors": CLASS_COLORS,
    "alpha": CLASS_ALPHA,
    "alpha_all": CLASS_ALPHA_ALL,
    "legend_regions": ["NETC", "SNFH", "ET"] + (["RC"] if HAS_RC else []),
    "title_label": "Class",
    "filename_suffix": "",
}

REGION_CONFIG = {
    "values": REGION_VALUES,
    "order": REGION_ORDER,
    "draw_order_all": [],
    "colors": REGION_COLORS,
    "alpha": REGION_ALPHA,
    "alpha_all": {},
    "legend_regions": ["ET", "TC", "WT"],
    "title_label": "Region (ET/TC/WT)",
    "filename_suffix": "_composite_regions",
}


def get_region_config(region_type: str) -> dict:
    """Returns the column/color/legend configuration for a region type.

    Args:
        region_type: Either "classes" (NETC/SNFH/ET/RC) or "regions"
            (composite ET/TC/WT).

    Returns:
        Configuration dictionary consumed by the plotting functions.

    Raises:
        ValueError: If region_type is not supported.
    """
    if region_type == "classes":
        return CLASS_CONFIG

    if region_type == "regions":
        return REGION_CONFIG

    raise ValueError(f"Unsupported region_type: {region_type}")


def pretty_model_name(model_name: str) -> str:
    """Returns a readable model name.

    Args:
        model_name: Internal model name.

    Returns:
        Display name used in the figure.
    """
    name = model_name.lower()

    if name == "unet3d":
        return "UNet 3D"

    if name == "resunet3d":
        return "ResUNet 3D"

    if name == "segresnet":
        return "SegResNet"

    if name == "swin_unetr":
        return "Swin UNETR"

    if name == "segmamba":
        return "SegMamba"

    if name == "segmambav2":
        return "SegMamba v2"

    if name == "dense_unet_plus":
        return "DenseUNet+"

    return model_name


def cleanup_memory() -> None:
    """Releases unused CPU and GPU memory."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def lighten_color(color: str, amount: float = 0.10) -> np.ndarray:
    """Lightens a color by mixing it with white.

    Args:
        color: Color in any Matplotlib-compatible format.
        amount: Amount of white mixed into the original color.

    Returns:
        Lightened RGB color.
    """
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)

    return np.clip(rgb + (white - rgb) * amount, 0.0, 1.0)


def load_case_tensors(case_dir: Path) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Loads one BraTS case and applies the test transforms.

    Args:
        case_dir: Path to the BraTS case directory.

    Returns:
        A tuple with the case identifier, the image tensor and the label tensor.

    Raises:
        FileNotFoundError: If the case files cannot be found.
        ValueError: If the transformed image or label has an unexpected shape.
    """
    case_data = get_case_data(case_dir, include_label=True)

    if case_data is None:
        raise FileNotFoundError(
            f"Could not build the case from {case_dir}. "
            "Check that the four modalities and the segmentation mask exist."
        )

    transforms = get_test_transforms(
        roi_size=ROI_SIZE,
        spacing=SPACING,
    )

    item = transforms(case_data)

    image = item["image"]
    label = item["label"]

    if image.ndim != 4:
        raise ValueError(
            f"Expected an image with shape [C, H, W, D], "
            f"but got {tuple(image.shape)}."
        )

    image = image.unsqueeze(0)

    if label.ndim == 4 and label.shape[0] == 1:
        label = label[0]

    if label.ndim != 3:
        raise ValueError(
            f"Expected a label with shape [H, W, D], "
            f"but got {tuple(label.shape)}."
        )

    return case_data["case_id"], image, label.long()


def resolve_case_key(case_dir: Path) -> str:
    """Returns a stable cache key for a case directory.

    Args:
        case_dir: Path to the BraTS case directory.

    Returns:
        Cache key derived from the case directory name.
    """
    return Path(case_dir).name


def gt_cache_paths(cache_dir: Path, case_key: str) -> Tuple[Path, Path]:
    """Returns the (array, metadata) cache paths for a ground truth.

    Args:
        cache_dir: Directory used as prediction/ground-truth cache.
        case_key: Cache key returned by resolve_case_key.

    Returns:
        Tuple with the .npy path and the .meta.json path.
    """
    array_path = cache_dir / f"{case_key}__gt.npy"
    meta_path = cache_dir / f"{case_key}__gt.meta.json"

    return array_path, meta_path


def pred_cache_path(cache_dir: Path, case_key: str, model_name: str) -> Path:
    """Returns the cache path for one model's prediction.

    Args:
        cache_dir: Directory used as prediction cache.
        case_key: Cache key returned by resolve_case_key.
        model_name: Internal model name.

    Returns:
        Path to the cached prediction .npy file.
    """
    safe_name = model_name.lower().strip()

    return cache_dir / f"{case_key}__pred__{safe_name}.npy"


def save_ground_truth_cache(
    cache_dir: Path,
    case_key: str,
    case_id: str,
    gt_label_np: np.ndarray,
) -> None:
    """Saves the ground-truth label map and its case id to the cache.

    Args:
        cache_dir: Directory used as prediction/ground-truth cache.
        case_key: Cache key returned by resolve_case_key.
        case_id: Human-readable case identifier used in the figure title.
        gt_label_np: Ground-truth label map.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    array_path, meta_path = gt_cache_paths(cache_dir, case_key)

    np.save(array_path, gt_label_np)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"case_id": case_id}, f)


def load_ground_truth_cache(
    cache_dir: Path,
    case_key: str,
) -> Tuple[str, np.ndarray]:
    """Loads the cached ground-truth label map and case id.

    Args:
        cache_dir: Directory used as prediction/ground-truth cache.
        case_key: Cache key returned by resolve_case_key.

    Returns:
        A tuple with the case id and the ground-truth label map.

    Raises:
        FileNotFoundError: If the ground truth was not cached yet.
    """
    array_path, meta_path = gt_cache_paths(cache_dir, case_key)

    if not array_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No cached ground truth found for '{case_key}' in '{cache_dir}'. "
            "Run '--mode predict' (or '--mode all') at least once for this "
            "case before using '--mode plot'."
        )

    gt_label_np = np.load(array_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta["case_id"], gt_label_np


def save_prediction_cache(
    cache_dir: Path,
    case_key: str,
    model_name: str,
    pred_np: np.ndarray,
) -> None:
    """Saves one model's prediction to the cache.

    Args:
        cache_dir: Directory used as prediction cache.
        case_key: Cache key returned by resolve_case_key.
        model_name: Internal model name.
        pred_np: Predicted label map.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.save(pred_cache_path(cache_dir, case_key, model_name), pred_np)


def load_prediction_cache(
    cache_dir: Path,
    case_key: str,
    model_name: str,
) -> np.ndarray:
    """Loads one model's cached prediction.

    Args:
        cache_dir: Directory used as prediction cache.
        case_key: Cache key returned by resolve_case_key.
        model_name: Internal model name.

    Returns:
        Predicted label map.

    Raises:
        FileNotFoundError: If the prediction was not cached yet.
    """
    path = pred_cache_path(cache_dir, case_key, model_name)

    if not path.exists():
        raise FileNotFoundError(
            f"No cached prediction found for model '{model_name}' and case "
            f"'{case_key}' in '{cache_dir}'. Run '--mode predict' for this "
            "model (in whichever environment it needs) before using "
            "'--mode plot'."
        )

    return np.load(path)


def load_model_for_inference(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Builds a model and loads its checkpoint.

    Args:
        model_name: Name of the model architecture.
        checkpoint_path: Path to the saved checkpoint.
        device: Device where the model is loaded.

    Returns:
        Model ready for inference.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(
        model_name=model_name,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        use_checkpoint=USE_CHECKPOINT,
    ).to(device)

    model, _, epoch, best_metric = load_checkpoint(
        model=model,
        optimizer=None,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.eval()

    print(
        f"[INFO] Model '{model_name}' loaded from '{checkpoint_path}'. "
        f"epoch={epoch}, best_metric={best_metric}"
    )

    return model


def predict_label_map(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
) -> np.ndarray:
    """Runs sliding-window inference and returns a label map.

    Args:
        model: Segmentation model.
        image: Input image tensor.
        device: Device used for inference.
        roi_size: Sliding-window region of interest.
        sw_batch_size: Number of windows processed in parallel.

    Returns:
        Predicted label map as a NumPy array.
    """
    image = image.to(device)

    use_amp = device.type == "cuda"
    autocast_context = (
        torch.amp.autocast("cuda", enabled=True)
        if use_amp
        else nullcontext()
    )

    with torch.no_grad():
        with autocast_context:
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
            )

        pred = torch.argmax(logits, dim=1)[0]
        pred_np = pred.detach().cpu().numpy().astype(np.int16)

    return pred_np


def create_region_mask(
    label_map: np.ndarray,
    region_name: str,
    region_values: Dict[str, List[int]],
) -> np.ndarray:
    """Creates a binary mask for a region or class.

    Args:
        label_map: Label map with integer segmentation classes.
        region_name: Region or class name.
        region_values: Mapping from region/class name to label values.

    Returns:
        Binary mask for the selected region.

    Raises:
        ValueError: If the region is not supported.
    """
    if region_name not in region_values:
        raise ValueError(f"Unsupported region: {region_name}")

    return np.isin(label_map, region_values[region_name])


def get_common_bbox(
    label_maps: List[np.ndarray],
    padding: int = 10,
) -> Tuple[slice, slice, slice]:
    """Returns a common bounding box around all non-zero voxels.

    Args:
        label_maps: Label maps to include in the common crop.
        padding: Number of voxels added around the crop.

    Returns:
        A tuple of slices defining the common crop.

    Raises:
        ValueError: If the list is empty or the shapes do not match.
    """
    if len(label_maps) == 0:
        raise ValueError("The label map list is empty.")

    combined = np.zeros_like(label_maps[0], dtype=bool)

    for label_map in label_maps:
        if label_map.shape != label_maps[0].shape:
            raise ValueError(
                f"All label maps must have the same shape. "
                f"Expected {label_maps[0].shape}, got {label_map.shape}."
            )

        combined |= label_map > 0

    coords = np.argwhere(combined)

    if coords.size == 0:
        return (
            slice(0, label_maps[0].shape[0]),
            slice(0, label_maps[0].shape[1]),
            slice(0, label_maps[0].shape[2]),
        )

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    mins = np.maximum(mins - padding, 0)
    maxs = np.minimum(maxs + padding, np.array(label_maps[0].shape))

    return (
        slice(int(mins[0]), int(maxs[0])),
        slice(int(mins[1]), int(maxs[1])),
        slice(int(mins[2]), int(maxs[2])),
    )


def configure_3d_axis(
    ax,
    volume_shape: Tuple[int, int, int],
    title: str = "",
    elev: float = 24.0,
    azim: float = -55.0,
) -> None:
    """Configures a clean 3D axis.

    Args:
        ax: Matplotlib 3D axis.
        volume_shape: Shape of the cropped volume.
        title: Axis title.
        elev: Camera elevation.
        azim: Camera azimuth.
    """
    ax.set_title(title, fontsize=10, fontweight="bold", pad=3)
    ax.set_facecolor("white")

    try:
        ax.set_proj_type("persp", focal_length=0.85)
    except TypeError:
        ax.set_proj_type("persp")

    ax.view_init(elev=elev, azim=azim)

    height, width, depth = volume_shape

    ax.set_box_aspect((depth, width, height), zoom=1.3)
    ax.set_xlim(0, depth)
    ax.set_ylim(0, width)
    ax.set_zlim(0, height)

    ax.set_axis_off()
    ax.grid(False)


def compute_face_lighting(
    verts_xyz: np.ndarray,
    faces: np.ndarray,
    base_color: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Computes face colors with a soft 3D lighting effect.

    Args:
        verts_xyz: Mesh vertices in x, y, z order.
        faces: Mesh faces.
        base_color: Base RGB color.
        alpha: Face transparency.

    Returns:
        RGBA colors for all mesh faces.
    """
    triangles = verts_xyz[faces]

    v1 = triangles[:, 1] - triangles[:, 0]
    v2 = triangles[:, 2] - triangles[:, 0]

    normals = np.cross(v1, v2)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normals = normals / norm

    light_dir = np.array([0.35, -0.45, 0.82], dtype=float)
    light_dir = light_dir / np.linalg.norm(light_dir)

    dot = np.dot(normals, light_dir)
    intensity = 0.80 + 0.20 * np.clip(dot, 0.0, 1.0)

    white_mix = 0.06
    shaded_rgb = base_color[None, :] * intensity[:, None]
    shaded_rgb = shaded_rgb * (1.0 - white_mix) + white_mix
    shaded_rgb = np.clip(shaded_rgb, 0.0, 1.0)

    rgba = np.concatenate(
        [
            shaded_rgb,
            np.full((shaded_rgb.shape[0], 1), alpha),
        ],
        axis=1,
    )

    return rgba


def prepare_surface_volume(
    mask: np.ndarray,
    smooth_sigma: float,
) -> np.ndarray:
    """Prepares a binary mask for smoother marching-cubes visualization.

    Args:
        mask: Binary region mask.
        smooth_sigma: Gaussian smoothing sigma.

    Returns:
        Floating-point volume used for surface extraction.
    """
    mask = mask.astype(np.float32)

    if smooth_sigma > 0:
        mask = gaussian_filter(mask, sigma=smooth_sigma)

    return mask


def add_mask_mesh(
    ax,
    mask: np.ndarray,
    color: str,
    alpha: float = 0.85,
    step_size: int = 1,
    smooth_sigma: float = 0.65,
    lighten_amount: float = 0.06,
    show_edges: bool = False,
) -> bool:
    """Extracts and draws a 3D mesh from a binary mask.

    Args:
        ax: Matplotlib 3D axis.
        mask: Binary mask to visualize.
        color: Base mesh color.
        alpha: Mesh transparency.
        step_size: Marching-cubes step size. Lower values increase detail.
        smooth_sigma: Gaussian smoothing sigma.
        lighten_amount: Amount used to lighten the color.
        show_edges: Whether to show mesh edges.

    Returns:
        True if a mesh was drawn, otherwise False.
    """
    if mask is None or int(mask.sum()) == 0:
        return False

    surface_volume = prepare_surface_volume(
        mask=mask,
        smooth_sigma=smooth_sigma,
    )

    min_value = float(surface_volume.min())
    max_value = float(surface_volume.max())

    if max_value <= 0.0:
        return False

    level = 0.5
    if not (min_value < level < max_value):
        level = min_value + 0.5 * (max_value - min_value)

    try:
        verts, faces, _, _ = measure.marching_cubes(
            surface_volume,
            level=level,
            step_size=max(1, int(step_size)),
            allow_degenerate=False,
        )
    except ValueError:
        return False

    verts_xyz = verts[:, [2, 1, 0]]

    base_color = lighten_color(color, amount=lighten_amount)
    facecolors = compute_face_lighting(
        verts_xyz=verts_xyz,
        faces=faces,
        base_color=base_color,
        alpha=alpha,
    )

    mesh = Poly3DCollection(
        verts_xyz[faces],
        facecolors=facecolors,
        linewidths=0.0,
    )

    if show_edges:
        mesh.set_edgecolor((0.15, 0.15, 0.15, 0.06))
        mesh.set_linewidth(0.01)
    else:
        mesh.set_edgecolor("none")
        mesh.set_linewidth(0.0)

    ax.add_collection3d(mesh)

    return True


def draw_grid_cell(
    ax,
    label_map: np.ndarray,
    region_name: str,
    volume_shape: Tuple[int, int, int],
    step_size: int,
    smooth_sigma: float,
    elev: float,
    azim: float,
    region_config: dict,
    show_title: bool = True,
) -> None:
    """Draws one cell of the 3D comparison grid.

    Args:
        ax: Matplotlib 3D axis.
        label_map: Label map to visualize.
        region_name: Region displayed in the cell.
        volume_shape: Shape of the cropped label map.
        step_size: Marching-cubes step size.
        smooth_sigma: Gaussian smoothing sigma.
        elev: Camera elevation.
        azim: Camera azimuth.
        region_config: Configuration returned by get_region_config.
        show_title: Whether to show the column title.
    """
    title = region_name if show_title else ""

    configure_3d_axis(
        ax=ax,
        volume_shape=volume_shape,
        title=title,
        elev=elev,
        azim=azim,
    )

    anything_drawn = False

    if region_name == "All":
        for subregion in region_config["draw_order_all"]:
            mask = create_region_mask(label_map, subregion, region_config["values"])

            added = add_mask_mesh(
                ax=ax,
                mask=mask,
                color=region_config["colors"][subregion],
                alpha=region_config["alpha_all"][subregion],
                step_size=step_size,
                smooth_sigma=smooth_sigma,
                lighten_amount=0.04,
                show_edges=False,
            )

            anything_drawn = anything_drawn or added

    else:
        mask = create_region_mask(label_map, region_name, region_config["values"])

        anything_drawn = add_mask_mesh(
            ax=ax,
            mask=mask,
            color=region_config["colors"][region_name],
            alpha=region_config["alpha"][region_name],
            step_size=step_size,
            smooth_sigma=smooth_sigma,
            lighten_amount=0.04,
            show_edges=False,
        )

    if not anything_drawn:
        ax.text2D(
            0.5,
            0.5,
            "Empty",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )


def save_models_region_grid(
    case_id: str,
    gt_label_map: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_dir: Path,
    region_config: dict,
    step_size: int = 1,
    smooth_sigma: float = 0.65,
    elev: float = 24.0,
    azim: float = -55.0,
    padding: int = 10,
    fig_width: float = 7.2,
    fig_height: float = 10.6,
    dpi: int = 300,
) -> None:
    """Saves a PNG grid comparing ground truth and model predictions.

    Args:
        case_id: Case identifier.
        gt_label_map: Ground-truth label map.
        predictions: Dictionary with one prediction per model.
        output_dir: Directory where the PNG figure is saved.
        region_config: Configuration returned by get_region_config.
        step_size: Marching-cubes step size.
        smooth_sigma: Gaussian smoothing sigma.
        elev: Camera elevation.
        azim: Camera azimuth.
        padding: Padding used for the common tumor crop.
        fig_width: Figure width in inches.
        fig_height: Figure height in inches.
        dpi: Output resolution for the PNG figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    row_items = [("Reference", gt_label_map)] + list(predictions.items())

    all_label_maps = [label_map for _, label_map in row_items]
    bbox = get_common_bbox(all_label_maps, padding=padding)

    row_items = [
        (row_name, label_map[bbox])
        for row_name, label_map in row_items
    ]

    region_order = region_config["order"]

    n_rows = len(row_items)
    n_cols = len(region_order)

    fig = plt.figure(
        figsize=(fig_width, fig_height),
        facecolor="white",
        constrained_layout=False,
    )


    left = 0.18
    right = 0.995
    bottom = 0.045
    top = 0.935

    grid = fig.add_gridspec(
        n_rows,
        n_cols,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=0.0,
        hspace=0.3,
    )


    for row_idx, (row_name, label_map) in enumerate(row_items):
        for col_idx, region_name in enumerate(region_order):
            ax = fig.add_subplot(
                grid[row_idx, col_idx],
                projection="3d",
            )

            draw_grid_cell(
                ax=ax,
                label_map=label_map,
                region_name=region_name,
                volume_shape=label_map.shape,
                step_size=step_size,
                smooth_sigma=smooth_sigma,
                elev=elev,
                azim=azim,
                region_config=region_config,
                show_title=row_idx == 0,
            )

    # for row_idx, (row_name, _) in enumerate(row_items):
    #     y = top - ((row_idx + 0.5) * (top - bottom) / n_rows)

    #     fig.text(
    #         0.035,
    #         y,
    #         row_name,
    #         rotation=90,
    #         va="center",
    #         ha="center",
    #         fontsize=10,
    #         fontweight="bold",
    #         color="black",
    #     )

    for row_idx, (row_name, _) in enumerate(row_items):
            y = top - ((row_idx + 0.5) * (top - bottom) / n_rows)

            fig.text(
                left - 0.02,
                y,
                row_name,
                rotation=0,
                va="center",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )


    legend_regions = region_config["legend_regions"]

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=region_config["colors"][region],
            markeredgecolor="black",
            markersize=7,
            label=region,
        )
        for region in legend_regions
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.014),
        ncol=len(legend_regions),
        frameon=True,
        fontsize=9,
        handlelength=0.8,
        columnspacing=0.8,
    )
    
    png_path = (
        output_dir
        / f"{case_id}_models_region_grid{region_config['filename_suffix']}.png"
    )

    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(fig)

    print(f"[OK] PNG figure saved at: {png_path}")


def run_predict_mode(
    case_dir: Path,
    model_names: List[str],
    checkpoint_paths: List[Path],
    cache_dir: Path,
    device: torch.device,
    sw_batch_size: int,
) -> None:
    """Runs inference for the given models and caches their predictions.

    This only needs the packages required by the given models (e.g. it can
    be run once per Python environment: one call for the models that share
    an environment, and a separate call in a different environment for a
    model like segmambav2 that needs incompatible dependencies).

    Args:
        case_dir: Path to the BraTS case directory.
        model_names: List of model names to run in this call.
        checkpoint_paths: List of checkpoint paths, same order as model_names.
        cache_dir: Directory where predictions are cached as .npy files.
        device: Device used for inference.
        sw_batch_size: Sliding-window inference batch size.

    Raises:
        ValueError: If the number of models and checkpoints does not match.
    """
    if len(model_names) != len(checkpoint_paths):
        raise ValueError(
            "The number of models and checkpoints must match. "
            f"Models: {len(model_names)}, checkpoints: {len(checkpoint_paths)}."
        )

    case_id, image, gt_label = load_case_tensors(case_dir)
    gt_label_np = gt_label.detach().cpu().numpy().astype(np.int16)

    case_key = resolve_case_key(case_dir)

    save_ground_truth_cache(
        cache_dir=cache_dir,
        case_key=case_key,
        case_id=case_id,
        gt_label_np=gt_label_np,
    )

    for model_name, checkpoint_path in zip(model_names, checkpoint_paths):
        print(f"\n[INFO] Processing model: {model_name}")

        model = load_model_for_inference(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        pred_np = predict_label_map(
            model=model,
            image=image,
            device=device,
            roi_size=ROI_SIZE,
            sw_batch_size=sw_batch_size,
        )

        save_prediction_cache(
            cache_dir=cache_dir,
            case_key=case_key,
            model_name=model_name,
            pred_np=pred_np,
        )

        print(
            f"[OK] Cached prediction for '{model_name}' at "
            f"'{pred_cache_path(cache_dir, case_key, model_name)}'"
        )

        del model
        cleanup_memory()


def run_plot_mode(
    case_dir: Path,
    model_names: List[str],
    cache_dir: Path,
    output_dir: Path,
    region_config: dict,
    step_size: int,
    smooth_sigma: float,
    elev: float,
    azim: float,
    padding: int,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    """Builds the comparison grid from previously cached predictions.

    This does not load any model or checkpoint, so it can be run in
    whichever environment has matplotlib/scikit-image, regardless of which
    environment(s) were used to generate the cached predictions.

    Args:
        case_dir: Path to the BraTS case directory (only used to resolve
            the cache key; its images are not read again).
        model_names: Model names to include in the grid, in display order.
            Each one must have been cached already via '--mode predict'.
        cache_dir: Directory where predictions were cached as .npy files.
        output_dir: Directory where the PNG figure is saved.
        region_config: Configuration returned by get_region_config.
        step_size: Marching-cubes step size.
        smooth_sigma: Gaussian smoothing sigma.
        elev: Camera elevation.
        azim: Camera azimuth.
        padding: Padding used for the common tumor crop.
        fig_width: Figure width in inches.
        fig_height: Figure height in inches.
        dpi: Output resolution for the PNG figure.
    """
    case_key = resolve_case_key(case_dir)

    case_id, gt_label_np = load_ground_truth_cache(
        cache_dir=cache_dir,
        case_key=case_key,
    )

    predictions: Dict[str, np.ndarray] = {}

    for model_name in model_names:
        pred_np = load_prediction_cache(
            cache_dir=cache_dir,
            case_key=case_key,
            model_name=model_name,
        )

        predictions[pretty_model_name(model_name)] = pred_np

    save_models_region_grid(
        case_id=case_id,
        gt_label_map=gt_label_np,
        predictions=predictions,
        output_dir=output_dir,
        region_config=region_config,
        step_size=step_size,
        smooth_sigma=smooth_sigma,
        elev=elev,
        azim=azim,
        padding=padding,
        fig_width=fig_width,
        fig_height=fig_height,
        dpi=dpi,
    )


def create_models_grid(
    case_dir: Path,
    model_names: List[str],
    checkpoint_paths: List[Path],
    output_dir: Path,
    cache_dir: Path,
    device: torch.device,
    sw_batch_size: int,
    region_config: dict,
    step_size: int,
    smooth_sigma: float,
    elev: float,
    azim: float,
    padding: int,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    """Creates predictions for all models and saves the comparison grid.

    Convenience wrapper for '--mode all': runs inference for every model in
    this same process and then builds the grid. Use '--mode predict' /
    '--mode plot' separately instead when a model (e.g. segmambav2) needs a
    different Python environment.

    Args:
        case_dir: Path to the BraTS case directory.
        model_names: List of model names.
        checkpoint_paths: List of checkpoint paths.
        output_dir: Directory where the PNG figure is saved.
        cache_dir: Directory used to cache predictions between steps.
        device: Device used for inference.
        sw_batch_size: Sliding-window inference batch size.
        region_config: Configuration returned by get_region_config.
        step_size: Marching-cubes step size.
        smooth_sigma: Gaussian smoothing sigma.
        elev: Camera elevation.
        azim: Camera azimuth.
        padding: Padding used for the common tumor crop.
        fig_width: Figure width in inches.
        fig_height: Figure height in inches.
        dpi: Output resolution for the PNG figure.

    Raises:
        ValueError: If the number of models and checkpoints does not match.
    """
    run_predict_mode(
        case_dir=case_dir,
        model_names=model_names,
        checkpoint_paths=checkpoint_paths,
        cache_dir=cache_dir,
        device=device,
        sw_batch_size=sw_batch_size,
    )

    run_plot_mode(
        case_dir=case_dir,
        model_names=model_names,
        cache_dir=cache_dir,
        output_dir=output_dir,
        region_config=region_config,
        step_size=step_size,
        smooth_sigma=smooth_sigma,
        elev=elev,
        azim=azim,
        padding=padding,
        fig_width=fig_width,
        fig_height=fig_height,
        dpi=dpi,
    )


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["2023", "2024"],
        default=DATASET,
        help=(
            "BraTS dataset to use (2023 or 2024). Already applied via the "
            "BRATS_DATASET environment variable before importing src.config."
        ),
    )

    parser.add_argument(
        "--case_dir",
        type=str,
        required=True,
        help="Path to the BraTS case directory.",
    )

    parser.add_argument(
        "--mode",
        choices=["all", "predict", "plot"],
        default="all",
        help=(
            "'all': runs inference for every model and builds the grid in "
            "this same process (default). "
            "'predict': only runs inference for --models and caches their "
            "predictions to --cache_dir; use this once per Python "
            "environment (e.g. one call for models sharing an environment, "
            "and a separate call in a different environment for "
            "segmambav2). "
            "'plot': only builds the grid from predictions already cached "
            "via '--mode predict'; does not load any model or checkpoint."
        ),
    )

    parser.add_argument(
        "--region_type",
        choices=["classes", "regions"],
        default="classes",
        help=(
            "'classes': one column per original BraTS class "
            "(NETC/SNFH/ET/RC). "
            "'regions': one column per composite region "
            "(ET/TC/WT), where TC=NETC+ET and WT=NETC+SNFH+ET."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "Model names in the same order as the checkpoints (for "
            "'--mode all/predict'), or just the models to include in the "
            "grid, in display order (for '--mode plot', no --checkpoints "
            "needed). E.g. unet3d segresnet swin_unetr segmamba segmambav2."
        ),
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help=(
            "Checkpoint paths in the same order as the models. Required for "
            "'--mode all' and '--mode predict'; not used in '--mode plot'."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/segmentation_3d",
        help="Output directory.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Directory used to cache per-model predictions between "
            "'--mode predict' and '--mode plot' calls. Defaults to "
            "'<output_dir>/_cache'. Must be the same path across all calls "
            "for a given case."
        ),
    )

    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=2,
        help="Batch size used by sliding-window inference.",
    )

    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Marching-cubes resolution. A value of 1 gives more detail.",
    )

    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=0.65,
        help=(
            "Visual surface smoothing. "
            "Use 0 to disable smoothing. Recommended values: 0.4-0.8."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for inference: cuda or cpu.",
    )

    parser.add_argument(
        "--elev",
        type=float,
        default=24.0,
        help="3D camera elevation.",
    )

    parser.add_argument(
        "--azim",
        type=float,
        default=-55.0,
        help="3D camera azimuth.",
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding added around the automatic tumor crop.",
    )

    parser.add_argument(
        "--fig_width",
        type=float,
        default=7.2,
        help="Figure width in inches.",
    )

    parser.add_argument(
        "--fig_height",
        type=float,
        default=10.6,
        help="Figure height in inches.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output resolution. Use 250 or 200 if the file is still too large.",
    )

    return parser.parse_args()


def main() -> None:
    """Runs the 3D segmentation grid generation workflow."""
    args = parse_args()

    device = torch.device(args.device)

    case_dir = Path(args.case_dir)
    output_dir = Path(args.output_dir)
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else output_dir / "_cache"
    )

    region_config = get_region_config(args.region_type)

    if args.mode in ("all", "predict") and args.checkpoints is None:
        raise ValueError(
            f"--checkpoints is required for --mode {args.mode!r}."
        )

    if args.mode == "predict":
        run_predict_mode(
            case_dir=case_dir,
            model_names=args.models,
            checkpoint_paths=[Path(path) for path in args.checkpoints],
            cache_dir=cache_dir,
            device=device,
            sw_batch_size=args.sw_batch_size,
        )

    elif args.mode == "plot":
        run_plot_mode(
            case_dir=case_dir,
            model_names=args.models,
            cache_dir=cache_dir,
            output_dir=output_dir,
            region_config=region_config,
            step_size=args.step_size,
            smooth_sigma=args.smooth_sigma,
            elev=args.elev,
            azim=args.azim,
            padding=args.padding,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
        )

    else:
        create_models_grid(
            case_dir=case_dir,
            model_names=args.models,
            checkpoint_paths=[Path(path) for path in args.checkpoints],
            output_dir=output_dir,
            cache_dir=cache_dir,
            device=device,
            sw_batch_size=args.sw_batch_size,
            region_config=region_config,
            step_size=args.step_size,
            smooth_sigma=args.smooth_sigma,
            elev=args.elev,
            azim=args.azim,
            padding=args.padding,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()


# Example usage:

## 2024
#  python -m scripts.create_3d_models_region_grid --mode predict --dataset 2024 --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models unet3d segresnet swin_unetr segmamba --checkpoints experiments/brats2024_unet3d_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2024_segresnet_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2024_swin_unetr_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2024_segmamba_roi128_bs1_nworkers2_cv5_without_background/fold_1/best_model.pt --output_dir figures/eda_article/brats2024/segmentation_3d
#  python -m scripts.create_3d_models_region_grid --mode predict --dataset 2024 --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models segmambav2 --checkpoints experiments/brats2024_segmambav2_roi128_bs1_nworkers2_cv5_without_background/fold_1/best_model.pt --output_dir figures/eda_article/brats2024/segmentation_3d

# python -m scripts.create_3d_models_region_grid --mode plot --dataset 2024 --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models unet3d segresnet swin_unetr segmamba segmambav2 --output_dir figures/eda_article/brats2024/segmentation_3d
# python -m scripts.create_3d_models_region_grid --mode plot --dataset 2024 --region_type classes --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models unet3d segresnet swin_unetr segmamba segmambav2 --output_dir figures/eda_article/brats2024/segmentation_3d --fig_height 8 --fig_width 6
# python -m scripts.create_3d_models_region_grid --mode plot --dataset 2024 --region_type regions --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models unet3d segresnet swin_unetr segmamba segmambav2 --output_dir figures/eda_article/brats2024/segmentation_3d --fig_height 8 --fig_width 6

## 2023
#  python -m scripts.create_3d_models_region_grid --mode predict --dataset 2023 --case_dir data_diego/BraTS-MEN-00891-000 --models unet3d segresnet swin_unetr segmamba --checkpoints experiments/brats2023_unet3d_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2023_segresnet_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2023_swin_unetr_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2023_segmamba_roi128_bs1_nworkers2_cv5_without_background/fold_1/best_model.pt --output_dir figures/eda_article/brats2023/segmentation_3d
#  python -m scripts.create_3d_models_region_grid --mode predict --dataset 2023 --case_dir data_diego/BraTS-MEN-00891-000 --models segmambav2 --checkpoints experiments/brats2023_segmambav2_roi128_bs1_nworkers2_cv5_without_background/fold_1/best_model.pt --output_dir figures/eda_article/brats2023/segmentation_3d

# python -m scripts.create_3d_models_region_grid --mode plot --dataset 2023 --case_dir data_diego/BraTS-MEN-00891-000 --models unet3d segresnet swin_unetr segmamba segmambav2 --output_dir figures/eda_article/brats2023/segmentation_3d
# python -m scripts.create_3d_models_region_grid --mode plot --dataset 2023 --region_type classes --case_dir data_diego/BraTS-MEN-00891-000 --models unet3d segresnet swin_unetr segmamba segmambav2 --output_dir figures/eda_article/brats2023/segmentation_3d --fig_height 8 --fig_width 6
# python -m scripts.create_3d_models_region_grid --mode plot --dataset 2023 --region_type regions --case_dir data_diego/BraTS-MEN-00891-000 --models unet3d segresnet swin_unetr segmamba segmambav2 --output_dir figures/eda_article/brats2023/segmentation_3d --fig_height 8 --fig_width 6