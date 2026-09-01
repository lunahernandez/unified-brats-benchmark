from __future__ import annotations


import os
import sys
import time
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from scipy.ndimage import binary_dilation, label as cc_label
from tqdm.auto import tqdm


from src.utils.metrics_GLI import (
    get_LesionWiseResults as get_gli_lesionwise_results,
)
from src.utils.metrics_MEN import (
    get_LesionWiseResults as get_men_lesionwise_results,
)

from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    MeanIoU,
)

from monai.transforms import AsDiscrete
from monai.data import decollate_batch


LesionWiseFn = Callable[..., Any]




@dataclass(frozen=True)
class LesionWiseEvaluatorConfig:
    """Configuration for a BraTS lesion-wise evaluator."""


    dataset: str
    challenge_name: str
    labels: list[str]
    region_values: OrderedDict[str, list[int]]
    evaluator_fn: LesionWiseFn




def get_mean(values: Iterable[float | None]) -> float | None:
    """Calculate the mean of a sequence, ignoring None and NaN values."""
    valid_values: list[float] = []


    for value in values:
        if value is None:
            continue


        value = float(value)


        if np.isnan(value):
            continue


        valid_values.append(value)


    if len(valid_values) == 0:
        return None


    return float(sum(valid_values) / len(valid_values))


def convert_to_brats_regions(onehot_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a one-hot encoded segmentation tensor into the standard BraTS
    evaluation regions.

    The output contains three channels corresponding to:
        - ET (Enhancing Tumor)
        - TC (Tumor Core = NETC ∪ ET)
        - WT (Whole Tumor = NETC ∪ SNFH ∪ ET)

    For BraTS 2024, the RC (Resection Cavity) class is intentionally excluded
    from the regional masks and is evaluated only as an individual class.

    Args:
        onehot_tensor: One-hot encoded segmentation tensor with shape
            (C, H, W, D), where C is:
            - 4 for BraTS 2023 (Background, NETC, SNFH, ET)
            - 5 for BraTS 2024 (Background, NETC, SNFH, ET, RC)

    Returns:
        A tensor of shape (3, H, W, D) containing the ET, TC, and WT regions.
    """

    if onehot_tensor.shape[0] == 4:          # Background + NETC + SNFH + ET
        netc = onehot_tensor[1:2]
        snfh = onehot_tensor[2:3]
        et = onehot_tensor[3:4]

    elif onehot_tensor.shape[0] == 5:        # Background + NETC + SNFH + ET + RC
        netc = onehot_tensor[1:2]
        snfh = onehot_tensor[2:3]
        et = onehot_tensor[3:4]

    else:
        raise ValueError(f"Unexpected number of channels: {onehot_tensor.shape[0]}")

    tc = torch.logical_or(netc, et).float()
    wt = torch.logical_or(tc, snfh).float()

    return torch.cat([et.float(), tc, wt], dim=0)

def get_lesionwise_evaluator_config(dataset: str) -> LesionWiseEvaluatorConfig:
    """Return the lesion-wise evaluator configuration for a dataset."""
    dataset = str(dataset)


    if dataset == "2023":
        region_values = OrderedDict(
            {
                "WT": [1, 2, 3],
                "TC": [1, 3],
                "NETC": [1],
                "SNFH": [2],
                "ET": [3],
            }
        )


        return LesionWiseEvaluatorConfig(
            dataset="2023",
            challenge_name="BraTS-MEN",
            labels=list(region_values.keys()),
            region_values=region_values,
            evaluator_fn=get_men_lesionwise_results,
        )


    if dataset == "2024":
        region_values = OrderedDict(
            {
                "WT": [1, 2, 3],
                "TC": [1, 3],
                "NETC": [1],
                "SNFH": [2],
                "ET": [3],
                "RC": [4],
            }
        )


        return LesionWiseEvaluatorConfig(
            dataset="2024",
            challenge_name="BraTS-GLI",
            labels=list(region_values.keys()),
            region_values=region_values,
            evaluator_fn=get_gli_lesionwise_results,
        )


    raise ValueError(f"Unsupported dataset: {dataset}. Use '2023' or '2024'.")




def initialize_lesionwise_metric_store(
    labels: list[str],
) -> dict[str, dict[str, list[float]]]:
    """Initialize lesion-wise Dice and HD95 storage."""
    return {
        label: {
            "dice": [],
            "hd95": [],
        }
        for label in labels
    }




def initialize_lesionwise_iou_metric_store(
    labels: list[str],
) -> dict[str, list[float]]:
    """Initialize lesion-wise IoU storage."""
    return {label: [] for label in labels}




def create_binary_region_mask(
    label_map: np.ndarray,
    region_values: list[int],
) -> np.ndarray:
    """Create a binary mask for a region or class."""
    return np.isin(label_map, region_values)




def calculate_binary_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """Calculate binary Intersection over Union."""
    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)


    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()


    if union == 0:
        return 1.0


    return float(intersection / union)




def calculate_lesionwise_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dilation_iterations: int = 1,
    min_lesion_size: int = 0,
    penalize_fp: bool = True,
) -> float:
    """Calculate lesion-wise IoU for one binary region.


    Ground-truth and predicted lesions are obtained with 26-connectivity.
    Each ground-truth lesion is matched with predicted components intersecting
    its dilated mask. Missed lesions and unmatched predicted lesions contribute
    an IoU of 0.
    """
    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)


    structure = np.ones((3, 3, 3), dtype=bool)
    gt_components, num_gt = cc_label(gt_mask, structure=structure)
    pred_components, num_pred = cc_label(pred_mask, structure=structure)


    lesion_ious: list[float] = []
    matched_pred_ids: set[int] = set()


    for gt_id in range(1, num_gt + 1):
        gt_lesion = gt_components == gt_id
        gt_size = int(gt_lesion.sum())


        if gt_size <= min_lesion_size:
            continue


        if dilation_iterations > 0:
            matching_region = binary_dilation(
                gt_lesion,
                structure=structure,
                iterations=dilation_iterations,
            )
        else:
            matching_region = gt_lesion


        pred_ids = np.unique(pred_components[matching_region])
        pred_ids = [int(pred_id) for pred_id in pred_ids if pred_id != 0]


        valid_pred_ids: list[int] = []


        for pred_id in pred_ids:
            pred_lesion = pred_components == pred_id
            pred_size = int(pred_lesion.sum())


            if pred_size > min_lesion_size:
                valid_pred_ids.append(pred_id)


        if len(valid_pred_ids) == 0:
            lesion_ious.append(0.0)
            continue


        matched_pred_ids.update(valid_pred_ids)
        matched_prediction = np.isin(pred_components, valid_pred_ids)


        lesion_ious.append(
            calculate_binary_iou(
                pred_mask=matched_prediction,
                gt_mask=gt_lesion,
            )
        )


    if penalize_fp:
        for pred_id in range(1, num_pred + 1):
            if pred_id in matched_pred_ids:
                continue


            pred_lesion = pred_components == pred_id
            pred_size = int(pred_lesion.sum())


            if pred_size <= min_lesion_size:
                continue


            lesion_ious.append(0.0)


    if len(lesion_ious) == 0:
        return 1.0


    return float(np.mean(lesion_ious))




def update_lesionwise_iou_metric_store(
    pred_label_map: np.ndarray,
    gt_label_map: np.ndarray,
    region_values: OrderedDict[str, list[int]],
    iou_store: dict[str, list[float]],
    dilation_iterations: int = 1,
    min_lesion_size: int = 0,
) -> dict[str, float]:
    """Update lesion-wise IoU values and return the values for one case."""
    case_ious: dict[str, float] = {}

    for region_name, values in region_values.items():
        pred_mask = create_binary_region_mask(
            label_map=pred_label_map,
            region_values=values,
        )
        gt_mask = create_binary_region_mask(
            label_map=gt_label_map,
            region_values=values,
        )

        iou = calculate_lesionwise_iou(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            dilation_iterations=dilation_iterations,
            min_lesion_size=min_lesion_size,
            penalize_fp=True,
        )

        iou = float(iou)

        iou_store[region_name].append(iou)
        case_ious[region_name] = iou

    return case_ious





def call_lesionwise_evaluator(
    config: LesionWiseEvaluatorConfig,
    pred_path: Path,
    gt_path: Path,
    work_dir: Path,
) -> pd.DataFrame:
    """Call the selected BraTS lesion-wise evaluator."""
    old_cwd = os.getcwd()


    try:
        os.chdir(work_dir)
        evaluator_output = config.evaluator_fn(
            pred_file=str(pred_path.resolve()),
            gt_file=str(gt_path.resolve()),
            challenge_name=config.challenge_name,
        )
    finally:
        os.chdir(old_cwd)


    if isinstance(evaluator_output, tuple):
        return evaluator_output[0]


    return evaluator_output




def update_lesionwise_metric_store(
    results_df: pd.DataFrame,
    metric_store: dict[str, dict[str, list[float]]],
) -> None:
    """Update Dice and HD95 storage from a lesion-wise results dataframe."""
    for _, row in results_df.iterrows():
        label = row["Labels"]


        if label not in metric_store:
            continue


        dice_value = row["LesionWise_Score_Dice"]
        hd95_value = row["LesionWise_Score_HD95"]


        if not pd.isna(dice_value):
            metric_store[label]["dice"].append(float(dice_value))


        if not pd.isna(hd95_value):
            metric_store[label]["hd95"].append(float(hd95_value))




def summarize_lesionwise_metric_store(
    metric_store: dict[str, dict[str, list[float]]],
) -> tuple[dict[str, dict[str, float | int | None]], float | None, float | None]:
    """Summarize accumulated lesion-wise Dice and HD95 metrics."""
    by_label_mean: dict[str, dict[str, float | int | None]] = {}
    global_dice: list[float] = []
    global_hd95: list[float] = []


    for label, values in metric_store.items():
        mean_dice = get_mean(values["dice"])
        mean_hd95 = get_mean(values["hd95"])


        by_label_mean[label] = {
            "lesionwise_dice": mean_dice,
            "lesionwise_hd95": mean_hd95,
            "num_cases_with_dice": len(values["dice"]),
            "num_cases_with_hd95": len(values["hd95"]),
        }


        if mean_dice is not None:
            global_dice.append(mean_dice)


        if mean_hd95 is not None:
            global_hd95.append(mean_hd95)


    return by_label_mean, get_mean(global_dice), get_mean(global_hd95)




def summarize_lesionwise_iou_metric_store(
    iou_store: dict[str, list[float]],
) -> tuple[dict[str, dict[str, float | int | None]], float | None]:
    """Summarize accumulated lesion-wise IoU metrics."""
    by_label_iou: dict[str, dict[str, float | int | None]] = {}
    global_iou: list[float] = []


    for label, values in iou_store.items():
        mean_iou = get_mean(values)


        by_label_iou[label] = {
            "lesionwise_iou": mean_iou,
            "num_cases_with_iou": len(values),
        }


        if mean_iou is not None:
            global_iou.append(mean_iou)


    return by_label_iou, get_mean(global_iou)




def merge_metric_summaries(
    lesionwise_summary: dict[str, dict[str, float | int | None]],
    iou_summary: dict[str, dict[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    """Merge Dice, HD95 and IoU summaries by region."""
    merged: dict[str, dict[str, float | int | None]] = {}


    for label in lesionwise_summary:
        merged[label] = dict(lesionwise_summary[label])
        merged[label].update(iou_summary.get(label, {}))


    for label, values in iou_summary.items():
        if label not in merged:
            merged[label] = dict(values)


    return merged




def extract_case_id(batch: dict[str, Any], index: int) -> str:
    """Extract a case identifier from a MONAI batch."""
    for key in ("id", "case_id", "patient_id", "name"):
        if key in batch:
            value = batch[key]


            if isinstance(value, (list, tuple)):
                return str(value[index])


            return str(value)


    for key in ("image_meta_dict", "label_meta_dict"):
        meta = batch.get(key)


        if isinstance(meta, dict):
            for filename_key in ("filename_or_obj", "filename"):
                if filename_key not in meta:
                    continue


                value = meta[filename_key]


                if isinstance(value, (list, tuple)):
                    return Path(str(value[index])).name.split(".nii")[0]


                return Path(str(value)).name.split(".nii")[0]


    return f"case_{index:04d}"




def extract_affine(batch: dict[str, Any], index: int) -> np.ndarray:
    """Extract an affine matrix from a MONAI batch if available."""
    for tensor_key in ("label", "image"):
        tensor = batch.get(tensor_key)
        meta = getattr(tensor, "meta", None)


        if isinstance(meta, dict) and "affine" in meta:
            affine = meta["affine"]


            if torch.is_tensor(affine):
                affine = affine.detach().cpu().numpy()


            affine = np.asarray(affine)


            if affine.ndim == 3:
                return affine[index]


            if affine.shape == (4, 4):
                return affine


    for meta_key in ("label_meta_dict", "image_meta_dict"):
        meta = batch.get(meta_key)


        if isinstance(meta, dict) and "affine" in meta:
            affine = meta["affine"]


            if torch.is_tensor(affine):
                affine = affine.detach().cpu().numpy()


            affine = np.asarray(affine)


            if affine.ndim == 3:
                return affine[index]


            if affine.shape == (4, 4):
                return affine


    return np.eye(4)




def tensor_to_label_maps(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output or labels to integer label maps."""
    tensor = tensor.detach().cpu()


    if tensor.ndim == 5:
        if tensor.shape[1] > 1:
            tensor = torch.argmax(tensor, dim=1)
        else:
            tensor = tensor[:, 0]


    if tensor.ndim == 4:
        return tensor.numpy().astype(np.uint8)


    raise ValueError(f"Unsupported tensor shape for label maps: {tuple(tensor.shape)}")




def save_label_map(path: Path, label_map: np.ndarray, affine: np.ndarray) -> None:
    """Save a label map as a NIfTI file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(label_map.astype(np.uint8), affine=affine),
        str(path),
    )

def summarize_voxelwise_metrics(
    dice: np.ndarray,
    iou: np.ndarray,
    hd95: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """
    Convert voxel-wise metric arrays into a dictionary by label.
    """

    result = {}

    for idx, label in enumerate(labels):
        result[label] = {
            "dice": float(dice[idx]),
            "iou": float(iou[idx]),
            "hd95": float(hd95[idx]),
        }

    return result

def summarize_global_lesionwise_metrics(
    metrics: dict[str, dict[str, float]],
) -> dict[str, float | None]:
    """
    Compute macro-average lesion-wise metrics across labels.
    """

    dice_values = []
    iou_values = []
    hd95_values = []

    for value in metrics.values():
        dice_values.append(value["lesionwise_dice"])
        iou_values.append(value["lesionwise_iou"])
        hd95_values.append(value["lesionwise_hd95"])

    return {
        "dice": get_mean(dice_values),
        "iou": get_mean(iou_values),
        "hd95": get_mean(hd95_values),
    }




def summarize_global_metrics(
    metrics: dict[str, dict[str, float]],
) -> dict[str, float | None]:
    """
    Compute macro-average metrics across labels.
    """

    dice_values = []
    iou_values = []
    hd95_values = []


    for value in metrics.values():

        dice_values.append(value["dice"])
        iou_values.append(value["iou"])
        hd95_values.append(value["hd95"])


    return {
        "dice": get_mean(dice_values),
        "iou": get_mean(iou_values),
        "hd95": get_mean(hd95_values),
    }


def metric_values_for_case(
    metric_output: torch.Tensor,
    case_index: int,
) -> list[float | None]:
    """Extract per-label metric values for one case from a MONAI metric output."""
    values = metric_output.detach().cpu().numpy()

    if values.ndim == 1:
        row = values
    else:
        row = values[case_index]

    row = np.asarray(row).reshape(-1)

    result: list[float | None] = []

    for value in row:
        value = float(value)

        if np.isnan(value):
            result.append(None)
        else:
            result.append(value)

    return result





def evaluate_test(
    model: torch.nn.Module,
    test_loader: Any,
    device: torch.device,
    dataset: str,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    output_dir: Path,
    model_name: str | None = None,
    fold_idx: int | None = None,
) -> dict[str, Any]:

    """Evaluate a model on the test set using lesion-wise metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    config = get_lesionwise_evaluator_config(dataset)
    metric_store = initialize_lesionwise_metric_store(config.labels)
    iou_store = initialize_lesionwise_iou_metric_store(config.labels)


    prediction_dir = output_dir / "predictions"
    ground_truth_dir = output_dir / "ground_truth"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)


    inference_times: list[float] = []

    per_case_records: list[dict[str, Any]] = []

    if dataset == "2023":
        voxel_class_names = [
            "NETC",
            "SNFH",
            "ET",
        ]
    else:
        voxel_class_names = [
            "NETC",
            "SNFH",
            "ET",
            "RC",
        ]

    voxel_region_names = [
        "ET",
        "TC",
        "WT",
    ]


    model.eval()

    num_classes = 4 if dataset == "2023" else 5

    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    dice_metric_cls = DiceMetric(
        include_background=False,
        reduction="mean_batch",
    )

    iou_metric_cls = MeanIoU(
        include_background=False,
        reduction="mean_batch",
    )

    hd95_metric_cls = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean_batch",
    )

    dice_metric_reg = DiceMetric(
        include_background=True,
        reduction="mean_batch",
    )

    iou_metric_reg = MeanIoU(
        include_background=True,
        reduction="mean_batch",
    )

    hd95_metric_reg = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,
        reduction="mean_batch",
    )

    try:
        progress_total = len(test_loader.dataset)
    except Exception:
        progress_total = len(test_loader)


    progress_bar = tqdm(
        total=progress_total,
        desc=f"Test BraTS {dataset}",
        unit="case",
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
        disable=False,
    )


    with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir_name:
        work_dir = Path(tmp_dir_name)


        with torch.no_grad():
            for batch_index, batch in enumerate(test_loader):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)


                if torch.cuda.is_available():
                    torch.cuda.synchronize()


                start_time = time.perf_counter()
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                )


                if torch.cuda.is_available():
                    torch.cuda.synchronize()


                elapsed = time.perf_counter() - start_time

                outputs_list = [post_pred(i) for i in decollate_batch(logits)]
                labels_list = [post_label(i) for i in decollate_batch(labels)]

                batch_dice_cls = dice_metric_cls(
                    outputs_list,
                    labels_list,
                )

                batch_iou_cls = iou_metric_cls(
                    outputs_list,
                    labels_list,
                )

                batch_hd95_cls = hd95_metric_cls(
                    outputs_list,
                    labels_list,
                )


                outputs_regions = [
                    convert_to_brats_regions(i)
                    for i in outputs_list
                ]

                labels_regions = [
                    convert_to_brats_regions(i)
                    for i in labels_list
                ]

                batch_dice_reg = dice_metric_reg(
                    outputs_regions,
                    labels_regions,
                )

                batch_iou_reg = iou_metric_reg(
                    outputs_regions,
                    labels_regions,
                )

                batch_hd95_reg = hd95_metric_reg(
                    outputs_regions,
                    labels_regions,
                )


                pred_maps = tensor_to_label_maps(logits)
                gt_maps = tensor_to_label_maps(labels)


                batch_size = int(pred_maps.shape[0])
                inference_times.extend([elapsed / batch_size] * batch_size)


                for item_index in range(batch_size):
                    raw_case_id = extract_case_id(
                        batch,
                        item_index,
                    )

                    raw_case_id = (
                        raw_case_id
                        .replace(".nii.gz", "")
                        .replace(".nii", "")
                    )

                    file_case_id = (
                        f"batch{batch_index:04d}_"
                        f"{item_index:02d}_"
                        f"{raw_case_id}"
                    )

                    affine = extract_affine(batch, item_index)
                    pred_map = pred_maps[item_index]
                    gt_map = gt_maps[item_index]

                    pred_path = prediction_dir / f"{file_case_id}_pred.nii.gz"
                    gt_path = ground_truth_dir / f"{file_case_id}_gt.nii.gz"

                    save_label_map(pred_path, pred_map, affine)
                    save_label_map(gt_path, gt_map, affine)


                    results_df = call_lesionwise_evaluator(
                        config=config,
                        pred_path=pred_path,
                        gt_path=gt_path,
                        work_dir=work_dir,
                    )


                    update_lesionwise_metric_store(
                        results_df=results_df,
                        metric_store=metric_store,
                    )

                    case_ious = update_lesionwise_iou_metric_store(
                        pred_label_map=pred_map,
                        gt_label_map=gt_map,
                        region_values=config.region_values,
                        iou_store=iou_store,
                        dilation_iterations=1,
                        min_lesion_size=0,
                    )

                    case_record: dict[str, Any] = {
                        "case_id": raw_case_id,
                        "dataset": dataset,
                        "model": model_name,
                        "fold": fold_idx,
                    }


                    # ---------------------------------------------------------
                    # VOXEL-WISE CLASSES
                    # ---------------------------------------------------------

                    voxel_class_dice = metric_values_for_case(
                        batch_dice_cls,
                        item_index,
                    )

                    voxel_class_iou = metric_values_for_case(
                        batch_iou_cls,
                        item_index,
                    )

                    voxel_class_hd95 = metric_values_for_case(
                        batch_hd95_cls,
                        item_index,
                    )


                    for label_name, dice, iou, hd95 in zip(
                        voxel_class_names,
                        voxel_class_dice,
                        voxel_class_iou,
                        voxel_class_hd95,
                    ):
                        case_record[f"voxel_class_{label_name}_dice"] = dice
                        case_record[f"voxel_class_{label_name}_iou"] = iou
                        case_record[f"voxel_class_{label_name}_hd95"] = hd95


                    case_record["voxel_global_classes_dice"] = get_mean(
                        voxel_class_dice
                    )

                    case_record["voxel_global_classes_iou"] = get_mean(
                        voxel_class_iou
                    )

                    case_record["voxel_global_classes_hd95"] = get_mean(
                        voxel_class_hd95
                    )


                    # ---------------------------------------------------------
                    # VOXEL-WISE REGIONS
                    # ---------------------------------------------------------

                    voxel_region_dice = metric_values_for_case(
                        batch_dice_reg,
                        item_index,
                    )

                    voxel_region_iou = metric_values_for_case(
                        batch_iou_reg,
                        item_index,
                    )

                    voxel_region_hd95 = metric_values_for_case(
                        batch_hd95_reg,
                        item_index,
                    )


                    for region_name, dice, iou, hd95 in zip(
                        voxel_region_names,
                        voxel_region_dice,
                        voxel_region_iou,
                        voxel_region_hd95,
                    ):
                        case_record[f"voxel_region_{region_name}_dice"] = dice
                        case_record[f"voxel_region_{region_name}_iou"] = iou
                        case_record[f"voxel_region_{region_name}_hd95"] = hd95


                    case_record["voxel_global_regions_dice"] = get_mean(
                        voxel_region_dice
                    )

                    case_record["voxel_global_regions_iou"] = get_mean(
                        voxel_region_iou
                    )

                    case_record["voxel_global_regions_hd95"] = get_mean(
                        voxel_region_hd95
                    )


                    # ---------------------------------------------------------
                    # LESION-WISE
                    # ---------------------------------------------------------

                    for _, row in results_df.iterrows():
                        label_name = str(row["Labels"])

                        if label_name not in config.labels:
                            continue

                        dice_value = row["LesionWise_Score_Dice"]
                        hd95_value = row["LesionWise_Score_HD95"]

                        case_record[f"lesion_{label_name}_dice"] = (
                            None
                            if pd.isna(dice_value)
                            else float(dice_value)
                        )

                        case_record[f"lesion_{label_name}_hd95"] = (
                            None
                            if pd.isna(hd95_value)
                            else float(hd95_value)
                        )

                        case_record[f"lesion_{label_name}_iou"] = (
                            case_ious.get(label_name)
                        )


                    # ---------------------------------------------------------
                    # LESION-WISE GLOBAL CLASSES
                    # ---------------------------------------------------------

                    case_record["lesion_global_classes_dice"] = get_mean(
                        case_record.get(f"lesion_{label}_dice")
                        for label in voxel_class_names
                    )

                    case_record["lesion_global_classes_iou"] = get_mean(
                        case_record.get(f"lesion_{label}_iou")
                        for label in voxel_class_names
                    )

                    case_record["lesion_global_classes_hd95"] = get_mean(
                        case_record.get(f"lesion_{label}_hd95")
                        for label in voxel_class_names
                    )


                    # ---------------------------------------------------------
                    # LESION-WISE GLOBAL REGIONS
                    # ---------------------------------------------------------

                    case_record["lesion_global_regions_dice"] = get_mean(
                        case_record.get(f"lesion_{region}_dice")
                        for region in voxel_region_names
                    )

                    case_record["lesion_global_regions_iou"] = get_mean(
                        case_record.get(f"lesion_{region}_iou")
                        for region in voxel_region_names
                    )

                    case_record["lesion_global_regions_hd95"] = get_mean(
                        case_record.get(f"lesion_{region}_hd95")
                        for region in voxel_region_names
                    )


                    per_case_records.append(case_record)




                    
                    pred_path.unlink(missing_ok=True)
                    gt_path.unlink(missing_ok=True)

                    avg_time = get_mean(inference_times) or 0.0
                    progress_bar.set_postfix(
                        {
                            "avg_inf_s": f"{avg_time:.2f}",
                            "case": raw_case_id[-24:],
                        }
                    )
                    progress_bar.update(1)


    progress_bar.close()

    per_case_df = pd.DataFrame(per_case_records)

    per_case_csv_path = (
        output_dir
        / "per_case_metrics.csv"
    )

    per_case_df.to_csv(
        per_case_csv_path,
        index=False,
    )

    print(
        f"Per-case metrics saved to: "
        f"{per_case_csv_path}"
    )



    by_label, mean_dice, mean_hd95 = summarize_lesionwise_metric_store(
        metric_store
    )
    by_label_iou, mean_iou = summarize_lesionwise_iou_metric_store(iou_store)
    merged_by_label = merge_metric_summaries(by_label, by_label_iou)

    dice_cls_tensor = dice_metric_cls.aggregate()
    iou_cls_tensor = iou_metric_cls.aggregate()
    hd95_cls_tensor = hd95_metric_cls.aggregate()

    dice_reg_tensor = dice_metric_reg.aggregate()
    iou_reg_tensor = iou_metric_reg.aggregate()
    hd95_reg_tensor = hd95_metric_reg.aggregate()
    
    dice_cls = dice_cls_tensor.cpu().numpy()
    iou_cls = iou_cls_tensor.cpu().numpy()
    hd95_cls = hd95_cls_tensor.cpu().numpy()

    dice_reg = dice_reg_tensor.cpu().numpy()
    iou_reg = iou_reg_tensor.cpu().numpy()
    hd95_reg = hd95_reg_tensor.cpu().numpy()

    voxelwise_classes = summarize_voxelwise_metrics(
        dice=dice_cls,
        iou=iou_cls,
        hd95=hd95_cls,
        labels=voxel_class_names,
    )


    voxelwise_regions = summarize_voxelwise_metrics(
        dice=dice_reg,
        iou=iou_reg,
        hd95=hd95_reg,
        labels=voxel_region_names,
    )


    voxelwise_global_classes = summarize_global_metrics(
        voxelwise_classes
    )


    voxelwise_global_regions = summarize_global_metrics(
        voxelwise_regions
    )


    lesionwise_classes = {
        key: value
        for key, value in merged_by_label.items()
        if key in voxel_class_names
    }


    lesionwise_regions = {
        key: value
        for key, value in merged_by_label.items()
        if key in voxel_region_names
    }


    lesionwise_global_classes = summarize_global_lesionwise_metrics(
        lesionwise_classes
    )

    lesionwise_global_regions = summarize_global_lesionwise_metrics(
        lesionwise_regions
    )



    return {
        "dataset": config.dataset,
        "challenge_name": config.challenge_name,


        "voxelwise": {
            "global_classes": voxelwise_global_classes,

            "global_regions": voxelwise_global_regions,

            "classes": voxelwise_classes,

            "regions": voxelwise_regions,
        },


        "lesionwise": {
            "global_classes": lesionwise_global_classes,

            "global_regions": lesionwise_global_regions,

            "classes": lesionwise_classes,

            "regions": lesionwise_regions,
        },


        "avg_inference_time_sec": get_mean(inference_times),
    }
