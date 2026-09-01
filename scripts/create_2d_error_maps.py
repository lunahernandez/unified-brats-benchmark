"""Builds 2D axial error-map grids for BraTS segmentation predictions.


Rows = one row per evaluated model.
Columns = individual tumor classes/sub-regions:
    BraTS 2023: NETC, SNFH, ET
    BraTS 2024: NETC, SNFH, ET, RC


Each error map compares the model prediction against the reference mask
for one class and overlays the result on the T1c anatomical slice:


    Green = True Positive  (TP)
    Red   = False Positive (FP)
    Blue  = False Negative (FN)


True-negative voxels remain visible as the grayscale anatomical image.


The script reuses the same prediction cache and case-loading logic as
scripts.create_3d_models_region_grid and
scripts.create_2d_axial_models_region_grid.


Therefore, predictions already generated for the previous 2D/3D figures
can be reused directly with --mode plot.


ASSUMPTIONS:
  * T1c is channel index 1 of the transformed image:
        t1n, t1c, t2w, t2f
  * The axial spatial axis is index 2.
  * Use the same --slice_index, --axial_axis, --rot_k, and --padding as
    the original 2D qualitative figure if you want an exact visual
    correspondence.
"""


import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional




PROJECT_ROOT = Path(__file__).resolve().parent.parent


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




# Reuse exactly the same loading / inference / cache logic
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
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402




# ---------------------------------------------------------------------
# Anatomical-image utilities
# ---------------------------------------------------------------------




def extract_modality_volume(
    image: torch.Tensor,
    channel_index: int,
) -> np.ndarray:
    """Extract one modality from [1, C, H, W, D].


    Args:
        image: Transformed image tensor.
        channel_index: MRI modality channel to extract.


    Returns:
        NumPy volume with shape [H, W, D].
    """
    n_channels = image.shape[1]


    if not 0 <= channel_index < n_channels:
        raise IndexError(
            f"--t1c_channel_index={channel_index} is invalid for an "
            f"image with {n_channels} channels."
        )


    volume = image[0, channel_index]


    return (
        volume.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )




def take_slice(
    volume: np.ndarray,
    axial_axis: int,
    slice_index: int,
) -> np.ndarray:
    """Extract a 2D slice from a 3D array."""
    indexer = [slice(None)] * volume.ndim
    indexer[axial_axis] = slice_index


    return volume[tuple(indexer)]




def normalize_grayscale(
    slice_2d: np.ndarray,
) -> np.ndarray:
    """Robustly normalize an MRI slice to [0, 1]."""
    slice_2d = slice_2d.astype(np.float32)


    p1, p99 = np.percentile(slice_2d, [1, 99])


    if p99 <= p1:
        return np.zeros_like(slice_2d)


    normalized = (slice_2d - p1) / (p99 - p1)


    return np.clip(normalized, 0.0, 1.0)




# ---------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------




def select_axial_slice_index(
    label_maps: List[np.ndarray],
    axial_axis: int,
    slice_index: Optional[int],
) -> int:
    """Select the slice with the largest total tumor area.


    This intentionally reproduces the selection logic of the previous
    2D qualitative visualization, so using the same GT and predictions
    should select the same slice.


    Args:
        label_maps: GT + predictions after common crop.
        axial_axis: Axis considered axial.
        slice_index: Optional manual override.


    Returns:
        Selected slice index in the cropped volume.
    """
    if slice_index is not None:
        return slice_index


    combined = np.zeros_like(
        label_maps[0],
        dtype=bool,
    )


    for label_map in label_maps:
        combined |= label_map > 0


    axes_to_sum = tuple(
        i
        for i in range(combined.ndim)
        if i != axial_axis
    )


    counts = combined.sum(axis=axes_to_sum)


    if counts.max() == 0:
        return combined.shape[axial_axis] // 2


    return int(np.argmax(counts))




# ---------------------------------------------------------------------
# Error-map generation
# ---------------------------------------------------------------------




def compute_error_masks(
    gt_label_slice: np.ndarray,
    pred_label_slice: np.ndarray,
    region_name: str,
    region_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute TP, FP and FN masks for one region.


    For "All", all foreground tumor classes are considered jointly:
        foreground = label > 0


    Args:
        gt_label_slice: Reference label map for one 2D slice.
        pred_label_slice: Predicted label map for the same slice.
        region_name: Class or composite-region name.
        region_config: Region configuration returned by
            get_region_config().


    Returns:
        Tuple:
            tp_mask,
            fp_mask,
            fn_mask
    """
    if region_name == "All":
        gt_mask = gt_label_slice > 0
        pred_mask = pred_label_slice > 0


    else:
        gt_mask = create_region_mask(
            gt_label_slice,
            region_name,
            region_config["values"],
        ).astype(bool)


        pred_mask = create_region_mask(
            pred_label_slice,
            region_name,
            region_config["values"],
        ).astype(bool)


    tp_mask = gt_mask & pred_mask
    fp_mask = (~gt_mask) & pred_mask
    fn_mask = gt_mask & (~pred_mask)


    return tp_mask, fp_mask, fn_mask







def apply_color_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    """Alpha blend one color onto an RGB image in place."""
    if not np.any(mask):
        return


    color_rgb = np.asarray(
        mcolors.to_rgb(color),
        dtype=np.float32,
    )


    rgb[mask] = (
        rgb[mask] * (1.0 - alpha)
        + color_rgb * alpha
    )




def build_error_map_rgb(
    t1c_slice_norm: np.ndarray,
    gt_label_slice: np.ndarray,
    pred_label_slice: np.ndarray,
    region_name: str,
    region_config: dict,
    alpha: float,
) -> np.ndarray:
    """Create an anatomical TP/FP/FN error-map overlay.


    Colors:
        TP = green
        FP = red
        FN = blue


    Args:
        t1c_slice_norm: Anatomical background normalized to [0, 1].
        gt_label_slice: Ground truth segmentation.
        pred_label_slice: Model segmentation.
        region_name: Class / region being evaluated.
        region_config: BraTS region configuration.
        alpha: Error-overlay opacity.


    Returns:
        RGB image with shape [H, W, 3].
    """
    rgb = np.stack(
        [t1c_slice_norm] * 3,
        axis=-1,
    ).astype(np.float32)


    tp_mask, fp_mask, fn_mask = compute_error_masks(
        gt_label_slice=gt_label_slice,
        pred_label_slice=pred_label_slice,
        region_name=region_name,
        region_config=region_config,
    )


    # These masks are mutually exclusive.
    apply_color_overlay(
        rgb,
        tp_mask,
        color="limegreen",
        alpha=alpha,
    )


    apply_color_overlay(
        rgb,
        fp_mask,
        color="red",
        alpha=alpha,
    )


    apply_color_overlay(
        rgb,
        fn_mask,
        color="dodgerblue",
        alpha=alpha,
    )


    return np.clip(
        rgb,
        0.0,
        1.0,
    )




# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------




def save_error_map_grid_2d(
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
    alpha: float = 0.70,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    dpi: int = 300,
) -> None:
    """Save a grid of error maps.


    Rows:
        models


    Columns:
        individual classes / regions


    Args:
        case_id: BraTS case ID.
        gt_label_map: Ground-truth 3D label map.
        predictions: Prediction label map for each model.
        t1c_volume: T1c anatomical volume.
        output_dir: Output directory.
        region_config: Configuration for classes or composite regions.
        axial_axis: Anatomical axial axis.
        slice_index: Optional manually chosen cropped slice index.
        rot_k: 90-degree rotations before display.
        padding: Common crop padding.
        alpha: Error-map overlay opacity.
        fig_width: Figure width, or automatic if None.
        fig_height: Figure height, or automatic if None.
        dpi: Output resolution.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------------
    # Use GT + all model predictions for exactly the same common crop
    # logic used by the previous qualitative figure.
    # --------------------------------------------------------------


    all_label_maps = [
        gt_label_map,
        *predictions.values(),
    ]


    bbox = get_common_bbox(
        all_label_maps,
        padding=padding,
    )


    gt_cropped = gt_label_map[bbox]


    predictions_cropped = {
        model_name: prediction[bbox]
        for model_name, prediction in predictions.items()
    }


    t1c_cropped = t1c_volume[bbox]


    # --------------------------------------------------------------
    # Same automatic slice-selection strategy as previous 2D figure
    # --------------------------------------------------------------


    cropped_label_maps = [
        gt_cropped,
        *predictions_cropped.values(),
    ]


    chosen_slice = select_axial_slice_index(
        label_maps=cropped_label_maps,
        axial_axis=axial_axis,
        slice_index=slice_index,
    )


    gt_slice = take_slice(
        gt_cropped,
        axial_axis,
        chosen_slice,
    )


    t1c_slice = take_slice(
        t1c_cropped,
        axial_axis,
        chosen_slice,
    )


    t1c_slice_norm = normalize_grayscale(
        t1c_slice
    )


    # --------------------------------------------------------------
    # Error maps should be class/region-specific.
    #
    # "All" is useful for the regular overlay grid but is not useful
    # here because TP/FP/FN should be interpretable per class.
    # --------------------------------------------------------------


    region_order = [
        region
        for region in region_config["order"]
        if region != "All"
    ]


    region_order.append("All")


    n_rows = len(predictions_cropped)
    n_cols = len(region_order)


    if fig_width is None:
        fig_width = max(
            6.0,
            2.15 * n_cols,
        )


    if fig_height is None:
        fig_height = max(
            6.0,
            1.75 * n_rows,
        )


    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        facecolor="white",
        squeeze=False,
        layout="compressed",
    )


    # --------------------------------------------------------------
    # Draw grid
    # --------------------------------------------------------------


    for row_idx, (
        model_name,
        pred_label_map,
    ) in enumerate(predictions_cropped.items()):


        pred_slice = take_slice(
            pred_label_map,
            axial_axis,
            chosen_slice,
        )


        for col_idx, region_name in enumerate(region_order):
            ax = axes[row_idx, col_idx]


            error_rgb = build_error_map_rgb(
                t1c_slice_norm=t1c_slice_norm,
                gt_label_slice=gt_slice,
                pred_label_slice=pred_slice,
                region_name=region_name,
                region_config=region_config,
                alpha=alpha,
            )


            display_image = np.rot90(
                error_rgb,
                k=rot_k,
            )


            ax.imshow(
                display_image,
                interpolation="nearest",
            )


            ax.set_xticks([])
            ax.set_yticks([])


            for spine in ax.spines.values():
                spine.set_visible(False)


            if row_idx == 0:
                ax.set_title(
                    region_name,
                    fontsize=11,
                    fontweight="bold",
                    pad=4,
                )


            if col_idx == 0:
                ax.set_ylabel(
                    model_name,
                    fontsize=11,
                    fontweight="bold",
                    rotation=90,
                    labelpad=7,
                )


    # --------------------------------------------------------------
    # Shared error-map legend
    # --------------------------------------------------------------


    legend_handles = [
        Patch(
            facecolor="limegreen",
            edgecolor="black",
            label="True Positive",
        ),
        Patch(
            facecolor="red",
            edgecolor="black",
            label="False Positive",
        ),
        Patch(
            facecolor="dodgerblue",
            edgecolor="black",
            label="False Negative",
        ),
    ]

    # 2024
    # fig.legend(
    #     handles=legend_handles,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 0.015),
    #     ncol=3,
    #     frameon=True,
    #     fontsize=10,
    #     columnspacing=1.4,
    # )

    # 2023
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=True,
        fontsize=10,
        columnspacing=1.4,
    )






    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        bottom=0.17,
        top=0.95,
        wspace=0.015,
        hspace=0.015,
    )


    filename_suffix = region_config.get(
        "filename_suffix",
        "",
    )


    png_path = (
        output_dir
        / (
            f"{case_id}_models_error_map_grid_2d_axial"
            f"{filename_suffix}.png"
        )
    )


    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )


    plt.close(fig)


    print(
        f"[OK] Error-map figure saved at: {png_path}"
    )


    print(
        "[INFO] Axial slice used "
        f"(post-crop index, axial_axis={axial_axis}): "
        f"{chosen_slice}"
    )


    print(
        "[INFO] Error colors: "
        "green=TP, red=FP, blue=FN"
    )




# ---------------------------------------------------------------------
# Plot mode
# ---------------------------------------------------------------------




def run_plot_mode_error_maps(
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
    alpha: float,
    fig_width: Optional[float],
    fig_height: Optional[float],
    dpi: int,
) -> None:
    """Build error maps from previously cached predictions."""
    case_key = resolve_case_key(
        case_dir
    )


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


        predictions[
            pretty_model_name(model_name)
        ] = pred_np


    # Reload only the anatomical image.
    _, image, _ = load_case_tensors(
        case_dir
    )


    t1c_volume = extract_modality_volume(
        image=image,
        channel_index=t1c_channel_index,
    )


    save_error_map_grid_2d(
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




# ---------------------------------------------------------------------
# All mode
# ---------------------------------------------------------------------




def create_error_maps_2d(
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
    alpha: float,
    fig_width: Optional[float],
    fig_height: Optional[float],
    dpi: int,
) -> None:
    """Run predictions and then generate the error-map figure."""
    run_predict_mode(
        case_dir=case_dir,
        model_names=model_names,
        checkpoint_paths=checkpoint_paths,
        cache_dir=cache_dir,
        device=device,
        sw_batch_size=sw_batch_size,
    )


    run_plot_mode_error_maps(
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




# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------




def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--dataset",
        type=str,
        choices=["2023", "2024"],
        default=DATASET,
    )


    parser.add_argument(
        "--case_dir",
        type=str,
        required=True,
        help="Path to the BraTS case directory.",
    )


    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "predict",
            "plot",
        ],
        default="plot",
        help=(
            "'predict': run inference and cache predictions. "
            "'plot': build error maps from existing cache. "
            "'all': run inference and then plot."
        ),
    )


    parser.add_argument(
        "--region_type",
        choices=[
            "classes",
            "regions",
        ],
        default="classes",
        help=(
            "Use 'classes' for NETC/SNFH/ET[/RC]. "
            "Use 'regions' for ET/TC/WT. "
            "For the reviewer-requested figures, use 'classes'."
        ),
    )


    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models in the display order.",
    )


    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help=(
            "Checkpoint paths matching --models. "
            "Required only for --mode predict/all."
        ),
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/error_maps",
    )


    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Prediction cache. Point this to the _cache directory "
            "already used by your previous 2D/3D visualization."
        ),
    )


    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=2,
    )


    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )


    parser.add_argument(
        "--t1c_channel_index",
        type=int,
        default=1,
    )


    parser.add_argument(
        "--axial_axis",
        type=int,
        default=2,
        choices=[
            0,
            1,
            2,
        ],
    )


    parser.add_argument(
        "--slice_index",
        type=int,
        default=None,
        help=(
            "Manual post-crop axial slice. "
            "Use the same value as the original 2D figure "
            "if that figure used an explicit slice."
        ),
    )


    parser.add_argument(
        "--rot_k",
        type=int,
        default=1,
    )


    parser.add_argument(
        "--padding",
        type=int,
        default=10,
    )


    parser.add_argument(
        "--alpha",
        type=float,
        default=0.70,
        help="Opacity of TP/FP/FN overlays.",
    )


    parser.add_argument(
        "--fig_width",
        type=float,
        default=None,
    )


    parser.add_argument(
        "--fig_height",
        type=float,
        default=None,
    )


    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )


    return parser.parse_args()




def main() -> None:
    """Run the 2D error-map workflow."""
    args = parse_args()


    device = torch.device(
        args.device
    )


    case_dir = Path(
        args.case_dir
    )


    output_dir = Path(
        args.output_dir
    )


    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else output_dir / "_cache"
    )


    region_config = get_region_config(
        args.region_type
    )


    if (
        args.mode in ("all", "predict")
        and args.checkpoints is None
    ):
        raise ValueError(
            f"--checkpoints is required for --mode {args.mode!r}."
        )


    if args.mode == "predict":
        run_predict_mode(
            case_dir=case_dir,
            model_names=args.models,
            checkpoint_paths=[
                Path(path)
                for path in args.checkpoints
            ],
            cache_dir=cache_dir,
            device=device,
            sw_batch_size=args.sw_batch_size,
        )


    elif args.mode == "plot":
        run_plot_mode_error_maps(
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
        create_error_maps_2d(
            case_dir=case_dir,
            model_names=args.models,
            checkpoint_paths=[
                Path(path)
                for path in args.checkpoints
            ],
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

# 2024
# python -m scripts.create_2d_error_maps --mode predict --dataset 2024 --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models unet3d segresnet swin_unetr segmamba --checkpoints experiments/brats2024_unet3d_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2024_segresnet_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2024_swin_unetr_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2024_segmamba_roi128_bs1_nworkers2_cv5_without_background/fold_1/best_model.pt --cache_dir figures/eda_article/brats2024/segmentation_2d_axial/_cache
# python -m scripts.create_2d_error_maps --mode predict --dataset 2024 --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models segmambav2 --checkpoints experiments/brats2024_segmambav2_roi128_bs1_nworkers2_cv5_without_background/fold_1/best_model.pt --cache_dir figures/eda_article/brats2024/segmentation_2d_axial/_cache

# python -m scripts.create_2d_error_maps --mode plot --dataset 2024 --region_type classes --case_dir data/training_data1_v2/BraTS-GLI-03063-100/ --models unet3d segresnet swin_unetr segmamba segmambav2 --cache_dir figures/eda_article/brats2024/segmentation_2d_axial/_cache --output_dir figures/eda_article/brats2024/error_maps --fig_width 10 --fig_height 8

# 2023
# python -m scripts.create_2d_error_maps --mode predict --dataset 2023 --case_dir data/brats_train_val_2023/BraTS-MEN-Train/BraTS-MEN-00891-000 --models unet3d segresnet swin_unetr segmamba --checkpoints experiments/brats2023_unet3d_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2023_segresnet_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2023_swin_unetr_roi128_bs1_nworkers4_cv5_without_background/fold_1/best_model.pt experiments/brats2023_segmamba_roi128_bs1_nworkers2_cv5_without_background/fold_1/best_model.pt --cache_dir figures/eda_article/brats2023/segmentation_2d_axial/_cache
# python -m scripts.create_2d_error_maps --mode predict --dataset 2023 --case_dir data/brats_train_val_2023/BraTS-MEN-Train/BraTS-MEN-00891-000 --models segmambav2 --checkpoints experiments/brats2023_segmambav2_roi128_bs1_nworkers2_cv5_without_background/fold_1/best_model.pt --cache_dir figures/eda_article/brats2023/segmentation_2d_axial/_cache

# python -m scripts.create_2d_error_maps --mode plot --dataset 2023 --region_type classes --case_dir data/brats_train_val_2023/BraTS-MEN-Train/BraTS-MEN-00891-000 --models unet3d segresnet swin_unetr segmamba segmambav2 --cache_dir figures/eda_article/brats2023/segmentation_2d_axial/_cache --output_dir figures/eda_article/brats2023/error_maps --fig_width 9 --fig_height 8
