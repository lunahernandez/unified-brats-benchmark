import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from src.utils.metrics_GLI import (
    get_LesionWiseResults as get_gli_lesionwise_results,
)
from src.utils.metrics_MEN import (
    get_LesionWiseResults as get_men_lesionwise_results,
)


LesionWiseFn = Callable[..., Any]


@dataclass(frozen=True)
class LesionWiseEvaluatorConfig:
    """Configuration for a BraTS lesion-wise evaluator.

    Args:
        dataset: Dataset identifier.
        challenge_name: Challenge name expected by the evaluator.
        labels: Labels expected in the lesion-wise evaluator output.
        class_labels: Individual foreground labels.
        region_labels: Composite BraTS regions.
        region_values: Mapping between labels or regions and integer
            segmentation values.
        evaluator_fn: Function used to compute lesion-wise metrics.
    """

    dataset: str
    challenge_name: str
    labels: list[str]
    class_labels: list[str]
    region_labels: list[str]
    region_values: OrderedDict[str, list[int]]
    evaluator_fn: LesionWiseFn


def get_mean(values: Iterable[float | int | None]) -> float | None:
    """Calculate the mean of a sequence, ignoring None and NaN values.

    Args:
        values: Iterable containing numeric values or None entries.

    Returns:
        Mean value as a float. Returns None if there are no valid values.
    """
    valid_values = []

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


def get_lesionwise_evaluator_config(
    dataset: str,
) -> LesionWiseEvaluatorConfig:
    """Return the lesion-wise evaluator configuration for a dataset.

    Args:
        dataset: Dataset identifier. Expected values are "2023" or "2024".

    Returns:
        Lesion-wise evaluator configuration.

    Raises:
        ValueError: If the dataset is not supported.
    """
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
            labels=["WT", "TC", "NETC", "SNFH", "ET"],
            class_labels=["NETC", "SNFH", "ET"],
            region_labels=["ET", "TC", "WT"],
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
            labels=["WT", "TC", "NETC", "SNFH", "ET", "RC"],
            class_labels=["NETC", "SNFH", "ET", "RC"],
            region_labels=["ET", "TC", "WT"],
            region_values=region_values,
            evaluator_fn=get_gli_lesionwise_results,
        )

    raise ValueError(
        f"Unsupported dataset: {dataset}. Use '2023' or '2024'."
    )


def initialize_lesionwise_metric_store(
    labels: list[str],
) -> dict[str, dict[str, list[float]]]:
    """Initialize lesion-wise metric storage.

    Args:
        labels: Labels that will be collected.

    Returns:
        Dictionary with Dice, IoU, and HD95 lists for each label.
    """
    return {
        label: {
            "dice": [],
            "iou": [],
            "hd95": [],
        }
        for label in labels
    }


def call_lesionwise_evaluator(
    config: LesionWiseEvaluatorConfig,
    pred_path: Path,
    gt_path: Path,
    work_dir: Path,
) -> pd.DataFrame:
    """Call the selected BraTS lesion-wise evaluator.

    The BraTS evaluator scripts may create temporary folders relative to the
    current working directory. This function temporarily changes the working
    directory so all evaluator-generated files remain inside `work_dir`.

    Args:
        config: Lesion-wise evaluator configuration.
        pred_path: Path to the predicted segmentation NIfTI file.
        gt_path: Path to the ground-truth segmentation NIfTI file.
        work_dir: Temporary working directory.

    Returns:
        Summary metrics dataframe.
    """
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
    """Update metric storage from a lesion-wise results dataframe.

    Args:
        results_df: Dataframe returned by a BraTS lesion-wise evaluator.
        metric_store: Metric storage dictionary to update.
    """
    for _, row in results_df.iterrows():
        label = row["Labels"]

        if label not in metric_store:
            continue

        dice_value = row["LesionWise_Score_Dice"]
        iou_value = row.get("LesionWise_Score_IoU", np.nan)
        hd95_value = row["LesionWise_Score_HD95"]

        if not pd.isna(dice_value):
            metric_store[label]["dice"].append(float(dice_value))

        if not pd.isna(iou_value):
            metric_store[label]["iou"].append(float(iou_value))

        if not pd.isna(hd95_value):
            metric_store[label]["hd95"].append(float(hd95_value))


def summarize_lesionwise_metric_store(
    metric_store: dict[str, dict[str, list[float]]],
) -> tuple[
    dict[str, dict[str, float | int | None]],
    float | None,
    float | None,
    float | None,
]:
    """Summarize accumulated lesion-wise metrics.

    Args:
        metric_store: Metric storage dictionary.

    Returns:
        Tuple containing the per-label summary, global mean Dice, global
        mean IoU, and global mean HD95.
    """
    by_label_mean = {}

    global_dice = []
    global_iou = []
    global_hd95 = []

    for label, values in metric_store.items():
        mean_dice = get_mean(values["dice"])
        mean_iou = get_mean(values["iou"])
        mean_hd95 = get_mean(values["hd95"])

        by_label_mean[label] = {
            "lesionwise_dice": mean_dice,
            "lesionwise_iou": mean_iou,
            "lesionwise_hd95": mean_hd95,
            "num_cases_with_dice": len(values["dice"]),
            "num_cases_with_iou": len(values["iou"]),
            "num_cases_with_hd95": len(values["hd95"]),
        }

        if mean_dice is not None:
            global_dice.append(mean_dice)

        if mean_iou is not None:
            global_iou.append(mean_iou)

        if mean_hd95 is not None:
            global_hd95.append(mean_hd95)

    return (
        by_label_mean,
        get_mean(global_dice),
        get_mean(global_iou),
        get_mean(global_hd95),
    )
