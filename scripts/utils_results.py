from collections.abc import Sequence
from pathlib import Path
from numbers import Real
from typing import Any

import csv
import json
import math

import numpy as np
from scipy.stats import pearsonr


REGION_NAMES = ["NETC", "SNFH", "ET", "RC", "TC", "WT"]
REGION_TEST_NAMES = ["NETC", "SNFH", "ET", "RC"]

MODEL_COLORS = {
    "UNet 3D": "#9AD776",
    "ResUNet 3D": "#4CA1D6",
    "Swin UNETR": "#FFCC6E",
    "SegMamba": "#DF3B49",
}

REGION_COLORS = {
    "NETC": "#ff595e",
    "SNFH": "#5ee35e",
    "ET": "#6c6bd6",
    "RC": "#fff45c",
    "TC": "#ffa64d",
    "WT": "#8ecae6",
}


def get_model_color(model: str) -> str:
    """Return the color assigned to a model.

    Args:
        model: The display name of the model.

    Returns:
        The model color as a hexadecimal string.

    Raises:
        ValueError: If the model does not have a defined color.
    """
    if model not in MODEL_COLORS:
        raise ValueError(f"No color defined for model: {model}")

    return MODEL_COLORS[model]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file.

    Args:
        path: The path to the JSON file.

    Returns:
        A dictionary with the JSON contents.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_valid_number(value: Any) -> bool:
    """Check whether a value is numeric and not NaN.

    Args:
        value: The value to check.

    Returns:
        ``True`` if the value is numeric and can be used in calculations;
        otherwise, ``False``.
    """
    if isinstance(value, bool):
        return False

    return isinstance(value, Real) and not math.isnan(float(value))


def safe_mean(values: Sequence[Real | None]) -> float | None:
    """Calculate the mean while ignoring invalid values.

    Args:
        values: A sequence of numeric values that may include ``None`` or ``NaN``.

    Returns:
        The mean of the valid values, or ``None`` if there are no valid values.
    """
    valid_values = [float(value) for value in values if is_valid_number(value)]

    if not valid_values:
        return None

    return float(sum(valid_values) / len(valid_values))


def safe_std(values: Sequence[Real | None]) -> float | None:
    """Calculate the sample standard deviation while ignoring invalid values.

    Args:
        values: A sequence of numeric values that may include ``None`` or ``NaN``.

    Returns:
        The sample standard deviation of the valid values. Returns ``0.0`` if
        there is only one valid value, and ``None`` if there are no valid values.
    """
    valid_values = [float(value) for value in values if is_valid_number(value)]

    if len(valid_values) == 0:
        return None

    if len(valid_values) == 1:
        return 0.0

    return float(np.std(valid_values, ddof=1))


def none_to_nan(values: Sequence[Real | None]) -> list[float]:
    """Convert ``None`` values to ``NaN``.

    Args:
        values: A sequence of numeric values that may include ``None``.

    Returns:
        A list where numeric values are converted to ``float`` and ``None``
        values are converted to ``np.nan``.
    """
    return [float(value) if value is not None else np.nan for value in values]


def get_model_name(data: dict[str, Any], path: Path) -> str:
    """Get a clean display name for a model.

    Args:
        data: The cross-validation data loaded from a JSON file.
        path: The path to the JSON file..

    Returns:
        The clean model name used in plots and tables.
    """
    raw_name = str(data.get("model") or path.parent.name).lower()

    if "resunet3d" in raw_name:
        return "ResUNet 3D"

    if "unet3d" in raw_name:
        return "UNet 3D"

    if "swin_unetr" in raw_name:
        return "Swin UNETR"

    if "segmamba" in raw_name:
        return "SegMamba"

    return str(data.get("model") or path.parent.name)


def collect_results(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Collect summarized results from several cross-validation files.

    Args:
        paths: The paths to the ``crossval_results.json`` files.

    Returns:
        A list of dictionaries with model-level metrics, regional metrics,
        timing information, memory usage and fold-level data.
    """
    results: list[dict[str, Any]] = []

    for path in paths:
        data = load_json(path)
        model = get_model_name(data, path)
        folds = data.get("folds", [])

        train_times_h = [
            fold.get("train_time_sec") / 3600
            for fold in folds
            if is_valid_number(fold.get("train_time_sec"))
        ]

        inference_times = [
            fold.get("inference_time_per_case_sec")
            for fold in folds
            if is_valid_number(fold.get("inference_time_per_case_sec"))
        ]

        train_memory_gb = [
            fold.get("train_memory_mb") / 1024
            for fold in folds
            if is_valid_number(fold.get("train_memory_mb"))
        ]

        test_memory_gb = [
            fold.get("test_memory_mb") / 1024
            for fold in folds
            if is_valid_number(fold.get("test_memory_mb"))
        ]

        results.append(
            {
                "model": model,
                "folds": folds,
                "dice_mean": data.get("cv_test_lesionwise_mean_dice_mean"),
                "dice_std": data.get("cv_test_lesionwise_mean_dice_std"),
                "hd95_mean": data.get("cv_test_lesionwise_mean_hd95_mean"),
                "hd95_std": data.get("cv_test_lesionwise_mean_hd95_std"),
                "regions": data.get("regions", {}),
                "train_time_h_mean": safe_mean(train_times_h),
                "train_time_h_std": safe_std(train_times_h),
                "inference_time_s_mean": safe_mean(inference_times),
                "inference_time_s_std": safe_std(inference_times),
                "train_memory_gb_mean": safe_mean(train_memory_gb),
                "train_memory_gb_std": safe_std(train_memory_gb),
                "test_memory_gb_mean": safe_mean(test_memory_gb),
                "test_memory_gb_std": safe_std(test_memory_gb),
            }
        )

    return results


def collect_fold_level_points(
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect fold-level performance and computational cost values.

    Args:
        results: The model-level results produced by ``collect_results``.

    Returns:
        A list of fold-level rows containing performance, timing and memory
        values.
    """
    rows: list[dict[str, Any]] = []

    for result in results:
        model = result["model"]

        for fold in result["folds"]:
            train_time_sec = fold.get("train_time_sec")
            train_memory_mb = fold.get("train_memory_mb")
            test_memory_mb = fold.get("test_memory_mb")

            rows.append(
                {
                    "model": model,
                    "fold": fold.get("fold"),
                    "test_dice": fold.get("test_lesionwise_mean_dice"),
                    "test_hd95": fold.get("test_lesionwise_mean_hd95"),
                    "train_time_h": (
                        train_time_sec / 3600
                        if is_valid_number(train_time_sec)
                        else None
                    ),
                    "inference_time_s": fold.get("inference_time_per_case_sec"),
                    "train_memory_gb": (
                        train_memory_mb / 1024
                        if is_valid_number(train_memory_mb)
                        else None
                    ),
                    "test_memory_gb": (
                        test_memory_mb / 1024
                        if is_valid_number(test_memory_mb)
                        else None
                    ),
                }
            )

    return rows


def collect_epoch_stats(
    folds: Sequence[dict[str, Any]],
    metric_key: str,
) -> tuple[list[int], list[float], list[float]]:
    """Collect the mean and standard deviation of a history metric by epoch.

    Args:
        folds: The fold-level dictionaries containing the training history.
        metric_key: The name of the history metric to collect.

    Returns:
        A tuple with three lists: the epochs, the mean values and the standard
        deviation values. Missing values are represented as ``np.nan``.
    """
    values_by_epoch: dict[int, list[float]] = {}

    for fold in folds:
        history = fold.get("history") or []

        for epoch_info in history:
            epoch = epoch_info.get("epoch")
            value = epoch_info.get(metric_key)

            if not isinstance(epoch, int):
                continue

            if not is_valid_number(value):
                continue

            values_by_epoch.setdefault(epoch, []).append(float(value))

    epochs = sorted(values_by_epoch.keys())
    means = [safe_mean(values_by_epoch[epoch]) for epoch in epochs]
    stds = [safe_std(values_by_epoch[epoch]) for epoch in epochs]

    return epochs, none_to_nan(means), none_to_nan(stds)


def get_valid_pairs(
    rows: Sequence[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> tuple[list[float], list[float]]:
    """Get valid paired values from a list of rows.

    Args:
        rows: The rows containing the variables to pair.
        x_key: The key used to read the x-axis variable.
        y_key: The key used to read the y-axis variable.

    Returns:
        A tuple containing the valid x and y values.
    """
    x_values: list[float] = []
    y_values: list[float] = []

    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)

        if is_valid_number(x) and is_valid_number(y):
            x_values.append(float(x))
            y_values.append(float(y))

    return x_values, y_values


def calculate_pearson_correlation(
    x_values: Sequence[Real],
    y_values: Sequence[Real],
) -> tuple[float | None, float | None]:
    """Calculate the Pearson correlation coefficient and p-value.

    Args:
        x_values: The values of the first variable.
        y_values: The values of the second variable.

    Returns:
        A tuple containing Pearson's r and the associated p-value. Returns
        ``(None, None)`` if the correlation cannot be calculated.
    """
    if len(x_values) < 3 or len(y_values) < 3:
        return None, None

    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return None, None

    r_value, p_value = pearsonr(x_values, y_values)
    return float(r_value), float(p_value)


def get_significance_stars(p_value: float | None) -> str:
    """Return the significance notation from a p-value.

    Args:
        p_value: The statistical p-value.

    Returns:
        The significance notation using ``***``, ``**``, ``*`` or ``n.s.``.
    """
    if p_value is None:
        return ""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "n.s."


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a list of dictionaries to a CSV file.

    Args:
        path: The path where the CSV file is saved.
        rows: The rows to write. Each row must be a dictionary with the same
            keys.
    """
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


