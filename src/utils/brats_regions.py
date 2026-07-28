import math
from collections import OrderedDict
from typing import Any

import torch
from monai.metrics import HausdorffDistanceMetric
from monai.utils.enums import MetricReduction


BRATS_REGION_LABELS_BY_DATASET = {
    "2023": OrderedDict(
        {
            "NETC": [1],
            "SNFH": [2],
            "ET": [3],
            "TC": [1, 3],        # NETC + ET
            "WT": [1, 2, 3],     # NETC + SNFH + ET
        }
    ),
    "2024": OrderedDict(
        {
            "NETC": [1],
            "SNFH": [2],
            "ET": [3],
            "RC": [4],
            "TC": [1, 3],        # NETC + ET
            "WT": [1, 2, 3],     # NETC + SNFH + ET
        }
    ),
}


def get_brats_region_labels(dataset: str) -> OrderedDict[str, list[int]]:
    """Return BraTS region definitions for the selected dataset.

    Args:
        dataset: Dataset identifier. Expected values are "2023" or "2024".

    Returns:
        Ordered dictionary mapping region names to label values.

    Raises:
        ValueError: If the dataset is not supported.
    """
    dataset = str(dataset)

    if dataset not in BRATS_REGION_LABELS_BY_DATASET:
        raise ValueError(
            f"Unsupported dataset: {dataset}. Use '2023' or '2024'."
        )

    return BRATS_REGION_LABELS_BY_DATASET[dataset]


def initialize_region_metrics(
    dataset: str,
) -> dict[str, dict[str, list[float]]]:
    """Initialize metric storage for the selected BraTS regions.

    Args:
        dataset: Dataset identifier. Expected values are "2023" or "2024".

    Returns:
        Dictionary with one entry per region. Each region stores Dice and
        HD95 values.
    """
    region_labels = get_brats_region_labels(dataset)

    return {
        region_name: {
            "dice_values": [],
            "hd95_values": [],
        }
        for region_name in region_labels
    }


def convert_to_label_map(x: torch.Tensor) -> torch.Tensor:
    """Convert predictions or labels to categorical label maps.

    Supported inputs are:
        - [B, C, H, W, D]&#58; logits, probabilities, or one-hot tensors.
        - [B, 1, H, W, D]&#58; label maps with singleton channel dimension.
        - [B, H, W, D]&#58; label maps.

    Args:
        x: Input tensor containing predictions or ground-truth labels.

    Returns:
        Tensor of shape [B, H, W, D] with integer labels.

    Raises:
        ValueError: If the input tensor shape is not supported.
    """
    if x.ndim == 5 and x.shape[1] > 1:
        return torch.argmax(x, dim=1).long()

    if x.ndim == 5 and x.shape[1] == 1:
        return x[:, 0].long()

    if x.ndim == 4:
        return x.long()

    raise ValueError(
        f"Unsupported tensor shape for label map conversion: {tuple(x.shape)}"
    )


def create_region_mask(
    label_map: torch.Tensor,
    region_values: list[int],
) -> torch.Tensor:
    """Create a binary mask for a BraTS region.

    Args:
        label_map: Tensor containing integer segmentation labels.
        region_values: Integer label values that define the region.

    Returns:
        Boolean tensor where True indicates voxels belonging to the region.
    """
    mask = torch.zeros_like(label_map, dtype=torch.bool)

    for value in region_values:
        mask = mask | (label_map == value)

    return mask


def calculate_binary_dice(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
) -> float:
    """Calculate the binary Dice coefficient.

    Args:
        pred_mask: Boolean tensor representing the predicted region.
        gt_mask: Boolean tensor representing the ground-truth region.

    Returns:
        Dice score. Returns 1.0 if both masks are empty and 0.0 if only one
        mask is empty.
    """
    pred_sum = int(pred_mask.sum().item())
    gt_sum = int(gt_mask.sum().item())

    if pred_sum == 0 and gt_sum == 0:
        return 1.0

    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = int((pred_mask & gt_mask).sum().item())

    return (2.0 * intersection) / (pred_sum + gt_sum)


def calculate_binary_hd95(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
) -> float | None:
    """Calculate the 95th percentile Hausdorff distance.

    Args:
        pred_mask: Boolean tensor representing the predicted region.
        gt_mask: Boolean tensor representing the ground-truth region.

    Returns:
        HD95 value. Returns 0.0 if both masks are empty and None if only one
        mask is empty.
    """
    pred_sum = int(pred_mask.sum().item())
    gt_sum = int(gt_mask.sum().item())

    if pred_sum == 0 and gt_sum == 0:
        return 0.0

    if pred_sum == 0 or gt_sum == 0:
        return None

    metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,
        reduction=MetricReduction.MEAN,
    )

    y_pred = pred_mask.unsqueeze(0).unsqueeze(0).float()
    y = gt_mask.unsqueeze(0).unsqueeze(0).float()

    metric(y_pred=y_pred, y=y)
    value = metric.aggregate().item()
    metric.reset()

    if math.isnan(value) or not math.isfinite(value):
        return None

    return float(value)


def update_region_metrics(
    stats: dict[str, dict[str, list[float]]],
    pred_labels: torch.Tensor,
    gt_labels: torch.Tensor,
    dataset: str,
) -> None:
    """Update region metrics using a prediction and ground-truth batch.

    Args:
        stats: Dictionary returned by `initialize_region_metrics`.
        pred_labels: Predicted labels or logits.
        gt_labels: Ground-truth labels.
        dataset: Dataset identifier. Expected values are "2023" or "2024".
    """
    region_labels = get_brats_region_labels(dataset)

    pred_labels = convert_to_label_map(pred_labels)
    gt_labels = convert_to_label_map(gt_labels)

    batch_size = pred_labels.shape[0]

    for batch_index in range(batch_size):
        pred_case = pred_labels[batch_index]
        gt_case = gt_labels[batch_index]

        for region_name, region_values in region_labels.items():
            pred_mask = create_region_mask(
                label_map=pred_case,
                region_values=region_values,
            )

            gt_mask = create_region_mask(
                label_map=gt_case,
                region_values=region_values,
            )

            dice = calculate_binary_dice(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
            )

            hd95 = calculate_binary_hd95(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
            )

            stats[region_name]["dice_values"].append(float(dice))

            if hd95 is not None:
                stats[region_name]["hd95_values"].append(float(hd95))


def summarize_region_metrics(
    stats: dict[str, dict[str, list[float]]],
    dataset: str,
) -> dict[str, Any]:
    """Summarize Dice and HD95 metrics for all selected BraTS regions.

    Args:
        stats: Dictionary containing accumulated Dice and HD95 values.
        dataset: Dataset identifier. Expected values are "2023" or "2024".

    Returns:
        Dictionary containing region-wise metrics and global averages.
    """
    region_results = {}

    dice_means = []
    hd95_means = []

    for region_name, values in stats.items():
        dice_values = values["dice_values"]
        hd95_values = values["hd95_values"]

        dice_mean = (
            sum(dice_values) / len(dice_values)
            if len(dice_values) > 0
            else None
        )

        hd95_mean = (
            sum(hd95_values) / len(hd95_values)
            if len(hd95_values) > 0
            else None
        )

        if dice_mean is not None:
            dice_mean = float(dice_mean)
            dice_means.append(dice_mean)

        if hd95_mean is not None:
            hd95_mean = float(hd95_mean)
            hd95_means.append(hd95_mean)

        region_results[region_name] = {
            "dice": dice_mean,
            "hd95": hd95_mean,
            "num_cases": len(dice_values),
            "num_valid_hd95_cases": len(hd95_values),
        }

    mean_dice = (
        float(sum(dice_means) / len(dice_means))
        if len(dice_means) > 0
        else None
    )

    mean_hd95 = (
        float(sum(hd95_means) / len(hd95_means))
        if len(hd95_means) > 0
        else None
    )

    selected_regions = ", ".join(stats.keys())

    return {
        "regions": region_results,
        "mean_dice": mean_dice,
        "mean_hd95": mean_hd95,
        "metric_note": (
            f"Voxel-wise validation metrics for BraTS {dataset} regions "
            f"({selected_regions}). This does not replicate the official "
            "lesion-wise challenge evaluator."
        ),
    }


