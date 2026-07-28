"""Builds a 2D axial comparison grid: mask overlay on T1c, one figure per case.

Rows = Reference (ground truth) + one row per model.
Columns = classes (NETC/SNFH/ET/RC/All) or composite regions (ET/TC/WT),
same structure as scripts.create_3d_models_region_grid.

This script reuses the case loading, prediction caching and model
inference logic from scripts.create_3d_models_region_grid instead of
duplicating it, so predictions cached by the 3D script (or by this one)
are interchangeable between the two.

ASSUMPTIONS TO VERIFY BEFORE TRUSTING THE OUTPUT:
  * T1c is assumed to be channel index 1 of the transformed image
    (typical BraTS order: t1n, t1c, t2w, t2f). Override with
    --t1c_channel_index if your src.data.dataset uses a different order.
  * The axial slice is assumed to be the last spatial axis (index 2) of
    the transformed [H, W, D] volume. Override with --axial_axis if the
    displayed slice does not look axial. Use --rot_k to fix the display
    orientation (radiological convention) without touching the code.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importing this module also pre-parses --dataset and exports
# BRATS_DATASET before src.config is imported (see its _preparse_dataset).
from scripts.create_3d_models_region_grid import (  # noqa: E402
    DATASET,
    create_region_mask,
    get_common_bbox,
    get_region_config,
    load_case_tensors,
    load_ground_truth_cache,
    load_prediction_cache,
    pretty_model_name,
    resolve_case_key,
    run_predict_mode,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def extract_modality_volume(image: torch.Tensor, channel_index: int) -> np.ndarray:
    """Extracts one modality channel as a NumPy volume.

    Args:
        image: Transformed image tensor with shape [1, C, H, W, D], as
            returned by load_case_tensors.
        channel_index: Index of the modality channel to extract.

    Returns:
        Volume with shape [H, W, D].

    Raises:
        IndexError: If channel_index is out of range.
    """
    n_channels = image.shape[1]

    if not (0 <= channel_index < n_channels):
        raise IndexError(
            f"--t1c_channel_index={channel_index} is out of range for an "
            f"image with {n_channels} channels."
        )

    volume = image[0, channel_index]

    return volume.detach().cpu().numpy().astype(np.float32)


def take_slice(volume: np.ndarray, axial_axis: int, slice_index: int) -> np.ndarray:
    """Extracts a 2D slice from a 3D volume along the given axis.

    Args:
        volume: 3D array.
        axial_axis: Axis index (0, 1 or 2) treated as the axial axis.
        slice_index: Index of the slice within that axis.

    Returns:
        2D array.
    """
    indexer = [slice(None)] * volume.ndim
    indexer[axial_axis] = slice_index

    return volume[tuple(indexer)]


def select_axial_slice_index(
    label_maps: List[np.ndarray],
    axial_axis: int,
    slice_index: Optional[int],
) -> int:
    """Picks the axial slice with the largest total tumor area.

    Args:
        label_maps: Label maps (already cropped to the common bbox) used
            to decide which slice has the most tumor to show.
        axial_axis: Axis index treated as the axial axis.
        slice_index: If given, this index is returned as-is (manual
            override) instead of being computed automatically.

    Returns:
        Index of the chosen slice along axial_axis.
    """
    if slice_index is not None:
        return slice_index

    combined = np.zeros_like(label_maps[0], dtype=bool)

    for label_map in label_maps:
        combined |= label_map > 0

    axes_to_sum = tuple(i for i in range(combined.ndim) if i != axial_axis)
    counts = combined.sum(axis=axes_to_sum)

    if counts.max() == 0:
        return combined.shape[axial_axis] // 2

    return int(np.argmax(counts))


def normalize_grayscale(slice_2d: np.ndarray) -> np.ndarray:
    """Normalizes a modality slice to [0, 1] using robust percentiles.

    Args:
        slice_2d: Raw modality slice.

    Returns:
        Normalized slice clipped to [0, 1].
    """
    slice_2d = slice_2d.astype(np.float32)

    p1, p99 = np.percentile(slice_2d, [1, 99])

    if p99 <= p1:
        return np.zeros_like(slice_2d)

    normed = (slice_2d - p1) / (p99 - p1)

    return np.clip(normed, 0.0, 1.0)


def build_overlay_rgb(
    t1c_slice_norm: np.ndarray,
    label_slice: np.ndarray,
    region_name: str,
    region_config: dict,
    alpha_override: Optional[float] = None,
) -> np.ndarray:
    """Builds an RGB image with the mask alpha-blended onto the T1c slice.

    Args:
        t1c_slice_norm: T1c slice normalized to [0, 1].
        label_slice: Label slice with integer segmentation classes.
        region_name: Region or class name, or "All" to stack every
            subregion (only meaningful for the "classes" config).
        region_config: Configuration returned by get_region_config.
        alpha_override: If given, replaces every per-region alpha.

    Returns:
        RGB image with shape [H, W, 3], values in [0, 1].
    """
    rgb = np.stack([t1c_slice_norm] * 3, axis=-1)

    if region_name == "All":
        draw_order = region_config["draw_order_all"]
        alphas = region_config["alpha_all"]
    else:
        draw_order = [region_name]
        alphas = region_config["alpha"]

    for subregion in draw_order:
        mask = create_region_mask(label_slice, subregion, region_config["values"])

        if mask.sum() == 0:
            continue

        color = np.array(mcolors.to_rgb(region_config["colors"][subregion]))
        alpha = alpha_override if alpha_override is not None else alphas[subregion]

        rgb[mask] = rgb[mask] * (1.0 - alpha) + color * alpha

    return np.clip(rgb, 0.0, 1.0)


def save_models_region_grid_2d(
    case_id: str,
    gt_label_map: np.ndarray,
    predictions: Dict[str, np.ndarray],
    t1c_volume: np.ndarray,
    output_dir: Path,
    region_config: dict,
    axial_axis: int = 2,
    slice_index: Optional[int] = None,
    rot_k: int = 1,
    padding: int = 10,
    alpha: Optional[float] = None,
    fig_width: float = 7.2,
    fig_height: float = 10.6,
    dpi: int = 300,
) -> None:
    """Saves a PNG grid of T1c axial slices with the mask overlaid.

    Args:
        case_id: Case identifier.
        gt_label_map: Ground-truth label map.
        predictions: Dictionary with one prediction per model.
        t1c_volume: T1c volume, same spatial shape as the label maps.
        output_dir: Directory where the PNG figure is saved.
        region_config: Configuration returned by get_region_config.
        axial_axis: Axis index treated as the axial axis.
        slice_index: Manual slice override (post-crop index). If None,
            the slice with the largest total tumor area is used.
        rot_k: Number of 90-degree counter-clockwise rotations applied
            to each slice before display, for orientation purposes.
        padding: Padding used for the common tumor crop.
        alpha: If given, overrides every per-region overlay alpha.
        fig_width: Figure width in inches.
        fig_height: Figure height in inches.
        dpi: Output resolution for the PNG figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    row_items = [("Reference", gt_label_map)] + list(predictions.items())

    all_label_maps = [label_map for _, label_map in row_items]
    bbox = get_common_bbox(all_label_maps, padding=padding)

    row_items = [(row_name, label_map[bbox]) for row_name, label_map in row_items]
    t1c_cropped = t1c_volume[bbox]

    chosen_slice = select_axial_slice_index(
        [label_map for _, label_map in row_items],
        axial_axis=axial_axis,
        slice_index=slice_index,
    )

    t1c_slice_norm = normalize_grayscale(
        take_slice(t1c_cropped, axial_axis, chosen_slice)
    )

    region_order = region_config["order"]
    n_rows = len(row_items)
    n_cols = len(region_order)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        facecolor="white",
        squeeze=False,
        layout="compressed",
    )

    for row_idx, (row_name, label_map) in enumerate(row_items):
        label_slice = take_slice(label_map, axial_axis, chosen_slice)

        for col_idx, region_name in enumerate(region_order):
            ax = axes[row_idx, col_idx]

            overlay = build_overlay_rgb(
                t1c_slice_norm=t1c_slice_norm,
                label_slice=label_slice,
                region_name=region_name,
                region_config=region_config,
                alpha_override=alpha,
            )

            display_overlay = np.rot90(overlay, k=rot_k)

            ax.imshow(display_overlay, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            if row_idx == 0:
                ax.set_title(region_name, fontsize=10, fontweight="bold", pad=3)

            if col_idx == 0:
                ax.set_ylabel(
                    row_name, fontsize=10, fontweight="bold", rotation=90
                )

    legend_regions = region_config["legend_regions"]

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=region_config["colors"][region],
            markeredgecolor="black",
            markersize=9,
            label=region,
        )
        for region in legend_regions
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(legend_regions),
        frameon=True,
        fontsize=8,
        handlelength=0.8,
        columnspacing=0.8,
    )

    fig.subplots_adjust(
        left=0.06, right=0.995, bottom=0.06, top=0.94, wspace=0.02, hspace=0.02
    )

    png_path = (
        output_dir
        / f"{case_id}_models_region_grid_2d_axial{region_config['filename_suffix']}.png"
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
    print(
        f"[INFO] Axial slice used (post-crop index, axial_axis={axial_axis}): "
        f"{chosen_slice}"
    )


def run_plot_mode_2d(
    case_dir: Path,
    model_names: List[str],
    cache_dir: Path,
    output_dir: Path,
    region_config: dict,
    t1c_channel_index: int,
    axial_axis: int,
    slice_index: Optional[int],
    rot_k: int,
    padding: int,
    alpha: Optional[float],
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    """Builds the 2D axial grid from previously cached predictions.

    Does not load any model or checkpoint. Reloads the transformed image
    (cheap, no model involved) only to extract the T1c channel used as
    the anatomical background.

    Args:
        case_dir: Path to the BraTS case directory.
        model_names: Model names to include in the grid, in display order.
            Each one must have been cached already via '--mode predict'.
        cache_dir: Directory where predictions were cached as .npy files.
        output_dir: Directory where the PNG figure is saved.
        region_config: Configuration returned by get_region_config.
        t1c_channel_index: Channel index of T1c in the transformed image.
        axial_axis: Axis index treated as the axial axis.
        slice_index: Manual slice override, or None for automatic.
        rot_k: Number of 90-degree rotations applied for display.
        padding: Padding used for the common tumor crop.
        alpha: If given, overrides every per-region overlay alpha.
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

    # Only the image is used here; the label is discarded since the
    # cached ground truth already matches these same transforms.
    _, image, _ = load_case_tensors(case_dir)
    t1c_volume = extract_modality_volume(image, t1c_channel_index)

    save_models_region_grid_2d(
        case_id=case_id,
        gt_label_map=gt_label_np,
        predictions=predictions,
        t1c_volume=t1c_volume,
        output_dir=output_dir,
        region_config=region_config,
        axial_axis=axial_axis,
        slice_index=slice_index,
        rot_k=rot_k,
        padding=padding,
        alpha=alpha,
        fig_width=fig_width,
        fig_height=fig_height,
        dpi=dpi,
    )


def create_models_grid_2d(
    case_dir: Path,
    model_names: List[str],
    checkpoint_paths: List[Path],
    output_dir: Path,
    cache_dir: Path,
    device: torch.device,
    sw_batch_size: int,
    region_config: dict,
    t1c_channel_index: int,
    axial_axis: int,
    slice_index: Optional[int],
    rot_k: int,
    padding: int,
    alpha: Optional[float],
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    """Creates predictions for all models and saves the 2D axial grid.

    Convenience wrapper for '--mode all': runs inference for every model
    in this same process (via run_predict_mode from the 3D script) and
    then builds the 2D grid. Use '--mode predict' / '--mode plot'
    separately instead when a model needs a different Python environment.
    """
    run_predict_mode(
        case_dir=case_dir,
        model_names=model_names,
        checkpoint_paths=checkpoint_paths,
        cache_dir=cache_dir,
        device=device,
        sw_batch_size=sw_batch_size,
    )

    run_plot_mode_2d(
        case_dir=case_dir,
        model_names=model_names,
        cache_dir=cache_dir,
        output_dir=output_dir,
        region_config=region_config,
        t1c_channel_index=t1c_channel_index,
        axial_axis=axial_axis,
        slice_index=slice_index,
        rot_k=rot_k,
        padding=padding,
        alpha=alpha,
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
            "BRATS_DATASET environment variable when scripts."
            "create_3d_models_region_grid was imported."
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
            "'all': runs inference for every model and builds the 2D grid "
            "in this same process (default). "
            "'predict': only runs inference for --models and caches their "
            "predictions to --cache_dir (shared cache format with the 3D "
            "script). "
            "'plot': only builds the 2D grid from predictions already "
            "cached via '--mode predict'; does not load any model or "
            "checkpoint."
        ),
    )

    parser.add_argument(
        "--region_type",
        choices=["classes", "regions"],
        default="classes",
        help=(
            "'classes': one column per original BraTS class "
            "(NETC/SNFH/ET/RC/All). "
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
            "needed)."
        ),
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help=(
            "Checkpoint paths in the same order as the models. Required "
            "for '--mode all' and '--mode predict'; not used in "
            "'--mode plot'."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/segmentation_2d_axial",
        help="Output directory.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Directory used to cache per-model predictions. Uses the same "
            "cache file format as scripts.create_3d_models_region_grid, so "
            "it can be shared between both scripts. Defaults to "
            "'<output_dir>/_cache'."
        ),
    )

    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=2,
        help="Batch size used by sliding-window inference.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for inference: cuda or cpu.",
    )

    parser.add_argument(
        "--t1c_channel_index",
        type=int,
        default=1,
        help=(
            "Channel index of T1c in the transformed image. Default "
            "assumes the typical BraTS modality order (t1n, t1c, t2w, "
            "t2f) -> index 1. Verify against src.data.dataset."
        ),
    )

    parser.add_argument(
        "--axial_axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help=(
            "Axis of the transformed [H, W, D] volume treated as the "
            "axial axis. Default assumes it is the last axis (D=2). "
            "Change if the displayed slice does not look axial."
        ),
    )

    parser.add_argument(
        "--slice_index",
        type=int,
        default=None,
        help=(
            "Manual axial slice index (post-crop, along --axial_axis). "
            "If not given, the slice with the largest total tumor area "
            "across all rows is selected automatically."
        ),
    )

    parser.add_argument(
        "--rot_k",
        type=int,
        default=1,
        help=(
            "Number of 90-degree counter-clockwise rotations applied to "
            "each slice before display. Adjust if the orientation looks "
            "wrong (e.g. sideways or upside down)."
        ),
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding added around the automatic tumor crop.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "If given, overrides every per-region overlay alpha (0-1). "
            "By default each region uses the alpha values defined in "
            "scripts.create_3d_models_region_grid."
        ),
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
    """Runs the 2D axial segmentation grid generation workflow."""
    args = parse_args()

    device = torch.device(args.device)

    case_dir = Path(args.case_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "_cache"

    region_config = get_region_config(args.region_type)

    if args.mode in ("all", "predict") and args.checkpoints is None:
        raise ValueError(f"--checkpoints is required for --mode {args.mode!r}.")

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
        run_plot_mode_2d(
            case_dir=case_dir,
            model_names=args.models,
            cache_dir=cache_dir,
            output_dir=output_dir,
            region_config=region_config,
            t1c_channel_index=args.t1c_channel_index,
            axial_axis=args.axial_axis,
            slice_index=args.slice_index,
            rot_k=args.rot_k,
            padding=args.padding,
            alpha=args.alpha,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
        )

    else:
        create_models_grid_2d(
            case_dir=case_dir,
            model_names=args.models,
            checkpoint_paths=[Path(path) for path in args.checkpoints],
            output_dir=output_dir,
            cache_dir=cache_dir,
            device=device,
            sw_batch_size=args.sw_batch_size,
            region_config=region_config,
            t1c_channel_index=args.t1c_channel_index,
            axial_axis=args.axial_axis,
            slice_index=args.slice_index,
            rot_k=args.rot_k,
            padding=args.padding,
            alpha=args.alpha,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()


# Example usage:

## 2024
#  python -m scripts.create_2d_axial_models_region_grid --mode predict --dataset 2024 --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models unet3d segresnet swin_unetr segmamba --checkpoints experiments/brats2024_unet3d_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2024_segresnet_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2024_swin_unetr_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2024_segmamba_roi128_bs1_nworkers2_cv5_without_background/fold_1/best_model.pt --output_dir figures/eda_article/brats2024/segmentation_2d_axial

# python -m scripts.create_2d_axial_models_region_grid --mode plot --dataset 2024 --region_type classes --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models unet3d segresnet swin_unetr segmamba segmambav2 --output_dir figures/eda_article/brats2024/segmentation_2d_axial --fig_height 8 --fig_width 6
# python -m scripts.create_2d_axial_models_region_grid --mode plot --dataset 2024 --region_type regions --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models unet3d segresnet swin_unetr segmamba segmambav2 --output_dir figures/eda_article/brats2024/segmentation_2d_axial --fig_height 8 --fig_width 6

## 2023
# python -m scripts.create_2d_axial_models_region_grid --mode plot --dataset 2023 --region_type classes --case_dir data_diego/BraTS-MEN-00891-000 --models unet3d segresnet swin_unetr segmamba segmambav2 --output_dir figures/eda_article/brats2023/segmentation_3d --fig_height 8 --fig_width 6
# python -m scripts.create_2d_axial_models_region_grid --mode plot --dataset 2023 --region_type regions --case_dir data_diego/BraTS-MEN-00891-000 --models unet3d segresnet swin_unetr segmamba segmambav2 --output_dir figures/eda_article/brats2023/segmentation_3d --fig_height 8 --fig_width 6