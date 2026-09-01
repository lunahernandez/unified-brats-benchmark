from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENTS_DIR = Path("experiments")
OUTPUT_DIR = Path("results/statistical_analysis")

ALPHA = 0.05

# Two-sided is the appropriate choice unless a directional
# hypothesis was explicitly pre-specified before looking at results.
ALTERNATIVE = "two-sided"

# Metrics to analyze automatically.
METRIC_SUFFIXES = (
    "_dice",
    "_iou",
    "_hd95",
)

# Only voxel-wise metrics.
# Change to None if you also want to include other metric families.
METRIC_PREFIX = "voxel_"


# ------------------------------------------------------------
# Experiment directories
# ------------------------------------------------------------
#
# The wildcards avoid having to write batch size, number of workers,
# etc. If more than one directory matches a pattern, the script
# deliberately stops so that results from different experiments are
# never mixed accidentally.
#
# Modify these patterns only if your experiment names differ.

EXPERIMENT_PATTERNS = {
    "BraTS 2023": {
        "3D U-Net": (
            "brats2023_unet3d_*cv5_without_background"
        ),
        "SegResNet": (
            "brats2023_segresnet_*cv5_without_background"
        ),
        "Swin UNETR": (
            "brats2023_swin_unetr_*cv5_without_background"
        ),
        "SegMamba": (
            "brats2023_segmamba_*cv5_without_background"
        ),
        "SegMambaV2": (
            "brats2023_segmambav2_*cv5_without_background"
        ),
    },
    "BraTS 2024": {
        "3D U-Net": (
            "brats2024_unet3d_*cv5_without_background"
        ),
        "SegResNet": (
            "brats2024_segresnet_*cv5_without_background"
        ),
        "Swin UNETR": (
            "brats2024_swin_unetr_*cv5_without_background"
        ),
        "SegMamba": (
            "brats2024_segmamba_*cv5_without_background"
        ),
        "SegMambaV2": (
            "brats2024_segmambav2_*cv5_without_background"
        ),
    },
}


# ------------------------------------------------------------
# Statistical comparisons
# ------------------------------------------------------------
#
# These are the comparisons directly motivated by the reviewers.
#
# candidate = model whose performance is being compared against
# reference.
#
# The last comparison explicitly addresses the very small
# SegMambaV2-vs-SegMamba differences mentioned by the reviewers.

COMPARISONS = [
    ("SegMamba", "3D U-Net"),
    ("SegMamba", "SegResNet"),
    ("SegMamba", "Swin UNETR"),
    ("SegMambaV2", "3D U-Net"),
    ("SegMambaV2", "SegResNet"),
    ("SegMambaV2", "Swin UNETR"),
    ("SegMambaV2", "SegMamba"),
]


# ============================================================
# INPUT
# ============================================================


def resolve_experiment_dir(pattern: str) -> Path:
    """
    Resolve an experiment directory from a glob pattern.

    Exactly one matching directory is required to avoid accidentally
    combining results from different experimental configurations.
    """
    matches = sorted(
        path
        for path in EXPERIMENTS_DIR.glob(pattern)
        if path.is_dir()
    )

    if not matches:
        raise FileNotFoundError(
            f"No experiment directory matched:\n"
            f"  {EXPERIMENTS_DIR / pattern}"
        )

    if len(matches) > 1:
        formatted = "\n".join(
            f"  - {path}" for path in matches
        )
        raise RuntimeError(
            f"More than one experiment matched pattern:\n"
            f"  {pattern}\n\n"
            f"Matches:\n{formatted}\n\n"
            f"Use a more specific pattern so that only one "
            f"experiment is selected."
        )

    return matches[0]


def get_fold_number(csv_path: Path) -> int:
    """
    Extract fold number from a path containing fold_X.
    """
    for parent in csv_path.parents:
        if parent.name.startswith("fold_"):
            try:
                return int(parent.name.split("_")[1])
            except (IndexError, ValueError):
                pass

    raise ValueError(
        f"Could not extract fold number from path: {csv_path}"
    )


def load_model_results(
    dataset: str,
    model: str,
    experiment_dir: Path,
) -> pd.DataFrame:
    """
    Load and concatenate per-case test results from all five folds.
    """
    csv_files = sorted(
        experiment_dir.glob(
            "fold_*/test_eval/per_case_metrics.csv"
        )
    )

    if len(csv_files) != 5:
        raise RuntimeError(
            f"{dataset} / {model}: expected 5 "
            f"per_case_metrics.csv files, found {len(csv_files)} "
            f"in:\n  {experiment_dir}"
        )

    frames: list[pd.DataFrame] = []

    for csv_path in csv_files:
        fold = get_fold_number(csv_path)

        df = pd.read_csv(csv_path)

        if "case_id" not in df.columns:
            raise KeyError(
                f"'case_id' not found in {csv_path}"
            )

        # We deliberately overwrite these fields so that the analysis
        # depends on the configuration above rather than potentially
        # inconsistent strings inside old CSV files.
        df["dataset_stat"] = dataset
        df["model_stat"] = model
        df["fold_stat"] = fold

        frames.append(df)

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    # Each subject must appear exactly once in the test partitions
    # across the complete five-fold CV.
    duplicated = result["case_id"].duplicated(
        keep=False
    )

    if duplicated.any():
        duplicate_ids = (
            result.loc[duplicated, "case_id"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"{dataset} / {model}: some subjects appear in more "
            f"than one test fold.\n"
            f"Examples: {duplicate_ids[:10]}"
        )

    return result


def load_all_results() -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load all datasets and models.
    """
    all_results: dict[
        str,
        dict[str, pd.DataFrame]
    ] = {}

    for dataset, model_patterns in EXPERIMENT_PATTERNS.items():
        all_results[dataset] = {}

        print(f"\nLoading {dataset}")

        for model, pattern in model_patterns.items():
            experiment_dir = resolve_experiment_dir(pattern)

            print(
                f"  {model:<12} -> {experiment_dir}"
            )

            df = load_model_results(
                dataset=dataset,
                model=model,
                experiment_dir=experiment_dir,
            )

            all_results[dataset][model] = df

            print(
                f"      subjects: {len(df)}"
            )

    return all_results


# ============================================================
# VALIDATION
# ============================================================


def validate_pairing(
    dataset: str,
    model_results: dict[str, pd.DataFrame],
) -> None:
    """
    Confirm that all models were evaluated on exactly the same
    subjects and that every subject belongs to the same test fold
    for every model.
    """
    model_names = list(model_results.keys())

    reference_model = model_names[0]
    reference_df = model_results[
        reference_model
    ].copy()

    reference_df["case_id"] = (
        reference_df["case_id"].astype(str)
    )

    reference_cases = set(reference_df["case_id"])

    reference_folds = (
        reference_df
        .set_index("case_id")["fold_stat"]
        .sort_index()
    )

    for model in model_names[1:]:
        df = model_results[model].copy()

        df["case_id"] = df["case_id"].astype(str)

        cases = set(df["case_id"])

        if cases != reference_cases:
            missing = sorted(reference_cases - cases)
            extra = sorted(cases - reference_cases)

            raise RuntimeError(
                f"\nPairing error in {dataset}: "
                f"{model} does not contain exactly the same "
                f"test subjects as {reference_model}.\n"
                f"Missing subjects: {missing[:10]}\n"
                f"Extra subjects: {extra[:10]}"
            )

        folds = (
            df
            .set_index("case_id")["fold_stat"]
            .sort_index()
        )

        if not folds.equals(reference_folds):
            mismatches = (
                pd.DataFrame(
                    {
                        "reference_fold": reference_folds,
                        "model_fold": folds,
                    }
                )
                .query(
                    "reference_fold != model_fold"
                )
            )

            raise RuntimeError(
                f"\nFold-pairing error in {dataset} / {model}.\n"
                f"The same subject is assigned to different test "
                f"folds across models.\n\n"
                f"{mismatches.head(10)}"
            )

    print(
        f"{dataset}: pairing validation successful "
        f"({len(reference_cases)} subjects)."
    )


# ============================================================
# METRIC DETECTION
# ============================================================


def find_metric_columns(
    model_results: dict[str, pd.DataFrame],
) -> list[str]:
    """
    Find metric columns shared by all models.
    """
    column_sets = [
        set(df.columns)
        for df in model_results.values()
    ]

    common_columns = set.intersection(*column_sets)

    metrics = []

    for column in sorted(common_columns):
        if METRIC_PREFIX is not None:
            if not column.startswith(METRIC_PREFIX):
                continue

        if not column.endswith(METRIC_SUFFIXES):
            continue

        metrics.append(column)

    if not metrics:
        raise RuntimeError(
            "No common metric columns were found."
        )

    return metrics


def get_metric_type(metric: str) -> str:
    """
    Return dice, iou or hd95.
    """
    if metric.endswith("_dice"):
        return "dice"

    if metric.endswith("_iou"):
        return "iou"

    if metric.endswith("_hd95"):
        return "hd95"

    raise ValueError(
        f"Unknown metric type: {metric}"
    )


# ============================================================
# STATISTICS
# ============================================================


def prepare_paired_values(
    candidate_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match candidate and reference results by case_id.

    Returns
    -------
    candidate_values
    reference_values
    improvement

    'improvement' is always oriented so that a positive number means
    that the candidate model performed better:

        Dice / IoU:
            candidate - reference

        HD95:
            reference - candidate

    Thus a positive rank-biserial correlation always favors the
    candidate architecture.
    """
    candidate = (
        candidate_df[
            ["case_id", metric]
        ]
        .copy()
    )

    reference = (
        reference_df[
            ["case_id", metric]
        ]
        .copy()
    )

    candidate["case_id"] = (
        candidate["case_id"].astype(str)
    )

    reference["case_id"] = (
        reference["case_id"].astype(str)
    )

    candidate = candidate.rename(
        columns={metric: "candidate"}
    )

    reference = reference.rename(
        columns={metric: "reference"}
    )

    paired = candidate.merge(
        reference,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    paired["candidate"] = pd.to_numeric(
        paired["candidate"],
        errors="coerce",
    )

    paired["reference"] = pd.to_numeric(
        paired["reference"],
        errors="coerce",
    )

    # NaN means the metric is undefined. For example, a class may be
    # absent from the reference segmentation. Because the same ground
    # truth is used for both models, these observations are omitted
    # from the paired test.
    valid = (
        ~paired["candidate"].isna()
        & ~paired["reference"].isna()
    )

    paired = paired.loc[valid].copy()

    candidate_values = (
        paired["candidate"]
        .to_numpy(dtype=float)
    )

    reference_values = (
        paired["reference"]
        .to_numpy(dtype=float)
    )

    metric_type = get_metric_type(metric)

    if metric_type == "hd95":
        # Lower HD95 is better.
        improvement = (
            reference_values
            - candidate_values
        )
    else:
        # Higher Dice / IoU is better.
        improvement = (
            candidate_values
            - reference_values
        )

    # MONAI may return +inf for HD95 when one mask is empty.
    #
    # If both models have +inf for the same subject, they are tied
    # with respect to this metric. Direct subtraction would produce
    # inf - inf = NaN, so explicitly encode it as zero difference.
    both_positive_inf = (
        np.isposinf(candidate_values)
        & np.isposinf(reference_values)
    )

    improvement[both_positive_inf] = 0.0

    return (
        candidate_values,
        reference_values,
        improvement,
    )


def paired_rank_biserial(
    differences: np.ndarray,
) -> float:
    """
    Calculate matched-pairs rank-biserial correlation.

    Positive values favor the candidate model because the differences
    have already been oriented so that positive = candidate better.

    Range:
        -1 = all ranked differences favor reference
         0 = no systematic direction
        +1 = all ranked differences favor candidate
    """
    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        ~np.isnan(differences)
    ]

    nonzero = differences[
        differences != 0
    ]

    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(nonzero),
        method="average",
    )

    positive_rank_sum = ranks[
        nonzero > 0
    ].sum()

    negative_rank_sum = ranks[
        nonzero < 0
    ].sum()

    total = (
        positive_rank_sum
        + negative_rank_sum
    )

    if total == 0:
        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / total
    )


def run_wilcoxon(
    differences: np.ndarray,
) -> dict[str, Any]:
    """
    Run the paired Wilcoxon signed-rank test.
    """
    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        ~np.isnan(differences)
    ]

    n = len(differences)
    n_nonzero = int(
        np.sum(differences != 0)
    )

    if n == 0:
        return {
            "n": 0,
            "n_nonzero": 0,
            "statistic": np.nan,
            "p_raw": np.nan,
        }

    if n_nonzero == 0:
        return {
            "n": n,
            "n_nonzero": 0,
            "statistic": 0.0,
            "p_raw": 1.0,
        }

    result = wilcoxon(
        differences,
        zero_method="wilcox",
        alternative=ALTERNATIVE,
        method="auto",
    )

    return {
        "n": n,
        "n_nonzero": n_nonzero,
        "statistic": float(
            result.statistic
        ),
        "p_raw": float(
            result.pvalue
        ),
    }


def safe_median(
    values: np.ndarray,
) -> float:
    """
    Median excluding NaN while preserving +/- inf where meaningful.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        ~np.isnan(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.median(values)
    )


def run_comparison(
    dataset: str,
    candidate_model: str,
    reference_model: str,
    metric: str,
    model_results: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """
    Perform one paired model comparison for one metric.
    """
    candidate_values, reference_values, improvement = (
        prepare_paired_values(
            candidate_df=model_results[
                candidate_model
            ],
            reference_df=model_results[
                reference_model
            ],
            metric=metric,
        )
    )

    wilcoxon_result = run_wilcoxon(
        improvement
    )

    effect_size = paired_rank_biserial(
        improvement
    )

    wins = int(
        np.sum(improvement > 0)
    )

    ties = int(
        np.sum(improvement == 0)
    )

    losses = int(
        np.sum(improvement < 0)
    )

    median_improvement = safe_median(
        improvement
    )

    candidate_median = safe_median(
        candidate_values
    )

    reference_median = safe_median(
        reference_values
    )

    candidate_inf = int(
        np.sum(
            np.isinf(candidate_values)
        )
    )

    reference_inf = int(
        np.sum(
            np.isinf(reference_values)
        )
    )

    return {
        "dataset": dataset,
        "metric": metric,
        "metric_type": get_metric_type(
            metric
        ),
        "candidate": candidate_model,
        "reference": reference_model,
        "comparison": (
            f"{candidate_model} vs "
            f"{reference_model}"
        ),
        "n_pairs": wilcoxon_result["n"],
        "n_nonzero": (
            wilcoxon_result["n_nonzero"]
        ),
        "candidate_median": (
            candidate_median
        ),
        "reference_median": (
            reference_median
        ),
        "median_improvement": (
            median_improvement
        ),
        "candidate_wins": wins,
        "ties": ties,
        "reference_wins": losses,
        "candidate_inf": candidate_inf,
        "reference_inf": reference_inf,
        "wilcoxon_statistic": (
            wilcoxon_result["statistic"]
        ),
        "p_raw": wilcoxon_result[
            "p_raw"
        ],
        "rank_biserial": effect_size,
    }


# ============================================================
# HOLM-BONFERRONI
# ============================================================


def holm_adjust(
    p_values: np.ndarray,
) -> np.ndarray:
    """
    Holm step-down adjusted p-values.

    This implementation returns adjusted p-values equivalent to the
    Holm-Bonferroni procedure.

    NaN values remain NaN.
    """
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        ~np.isnan(p_values)
    )[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p_values[
        valid_indices
    ]

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    m = len(sorted_p)

    sorted_adjusted = np.empty(
        m,
        dtype=float,
    )

    previous = 0.0

    for i, p_value in enumerate(
        sorted_p
    ):
        multiplier = m - i

        current = min(
            1.0,
            multiplier * p_value,
        )

        # Holm adjusted p-values must be monotonic.
        current = max(
            current,
            previous,
        )

        sorted_adjusted[i] = current
        previous = current

    # Undo sorting.
    unsorted_adjusted = np.empty(
        m,
        dtype=float,
    )

    unsorted_adjusted[order] = (
        sorted_adjusted
    )

    adjusted[valid_indices] = (
        unsorted_adjusted
    )

    return adjusted


def apply_holm_correction(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply Holm-Bonferroni correction within each:

        dataset x metric

    Therefore, for each statistical endpoint there are seven
    corrected pairwise comparisons:

        SegMamba vs three baselines
        SegMambaV2 vs three baselines
        SegMambaV2 vs SegMamba

    This corresponds directly to the model comparisons motivated by
    the reviewers while avoiding correction across unrelated
    outcomes such as Dice and HD95.
    """
    results = results.copy()

    results["p_holm"] = np.nan

    grouped = results.groupby(
        ["dataset", "metric"],
        sort=False,
    )

    for _, indices in grouped.groups.items():
        group_indices = list(indices)

        p_values = (
            results.loc[
                group_indices,
                "p_raw",
            ]
            .to_numpy(dtype=float)
        )

        adjusted = holm_adjust(
            p_values
        )

        results.loc[
            group_indices,
            "p_holm",
        ] = adjusted

    results["significant_holm"] = (
        results["p_holm"] < ALPHA
    )

    return results


# ============================================================
# INTERPRETATION
# ============================================================


def classify_effect_size(
    value: float,
) -> str:
    """
    Simple descriptive interpretation of absolute rank-biserial
    correlation magnitude.

    These labels should be treated descriptively rather than as hard
    clinical thresholds.
    """
    if np.isnan(value):
        return "undefined"

    magnitude = abs(value)

    if magnitude < 0.10:
        return "negligible"

    if magnitude < 0.30:
        return "small"

    if magnitude < 0.50:
        return "moderate"

    return "large"


def make_interpretation(
    row: pd.Series,
) -> str:
    """
    Generate a concise interpretation for inspection.
    """
    if pd.isna(row["p_holm"]):
        return "not testable"

    effect = row["rank_biserial"]

    if effect > 0:
        direction = "candidate better"
    elif effect < 0:
        direction = "reference better"
    else:
        direction = "no directional difference"

    if row["significant_holm"]:
        return (
            f"statistically significant; "
            f"{direction}"
        )

    return (
        f"not statistically significant; "
        f"{direction}"
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = load_all_results()

    print(
        "\n"
        "========================================"
    )
    print("VALIDATING PAIRED DESIGN")
    print(
        "========================================"
    )

    for dataset, model_results in (
        all_results.items()
    ):
        validate_pairing(
            dataset=dataset,
            model_results=model_results,
        )

    statistical_results: list[
        dict[str, Any]
    ] = []

    print(
        "\n"
        "========================================"
    )
    print("RUNNING SUBJECT-LEVEL WILCOXON TESTS")
    print(
        "========================================"
    )

    for dataset, model_results in (
        all_results.items()
    ):
        metrics = find_metric_columns(
            model_results
        )

        print(
            f"\n{dataset}: "
            f"{len(metrics)} metrics detected"
        )

        for metric in metrics:
            print(
                f"  {metric}"
            )

            for (
                candidate_model,
                reference_model,
            ) in COMPARISONS:

                if (
                    candidate_model
                    not in model_results
                ):
                    raise KeyError(
                        f"{candidate_model} is "
                        f"missing from {dataset}"
                    )

                if (
                    reference_model
                    not in model_results
                ):
                    raise KeyError(
                        f"{reference_model} is "
                        f"missing from {dataset}"
                    )

                result = run_comparison(
                    dataset=dataset,
                    candidate_model=(
                        candidate_model
                    ),
                    reference_model=(
                        reference_model
                    ),
                    metric=metric,
                    model_results=(
                        model_results
                    ),
                )

                statistical_results.append(
                    result
                )

    results_df = pd.DataFrame(
        statistical_results
    )

    results_df = apply_holm_correction(
        results_df
    )

    results_df["effect_magnitude"] = (
        results_df[
            "rank_biserial"
        ].apply(
            classify_effect_size
        )
    )

    results_df["interpretation"] = (
        results_df.apply(
            make_interpretation,
            axis=1,
        )
    )

    # Sort for easier inspection.
    results_df = results_df.sort_values(
        by=[
            "dataset",
            "metric",
            "candidate",
            "reference",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Full output
    # --------------------------------------------------------

    full_output = (
        OUTPUT_DIR
        / "subject_level_wilcoxon_holm.csv"
    )

    results_df.to_csv(
        full_output,
        index=False,
    )

    # --------------------------------------------------------
    # Only global metrics
    # --------------------------------------------------------

    global_df = results_df[
        results_df["metric"].str.contains(
            "_global_",
            regex=False,
        )
    ].copy()

    global_output = (
        OUTPUT_DIR
        / "global_subject_level_wilcoxon_holm.csv"
    )

    global_df.to_csv(
        global_output,
        index=False,
    )

    # --------------------------------------------------------
    # Significant comparisons
    # --------------------------------------------------------

    significant_df = results_df[
        results_df["significant_holm"]
    ].copy()

    significant_output = (
        OUTPUT_DIR
        / "significant_comparisons.csv"
    )

    significant_df.to_csv(
        significant_output,
        index=False,
    )

    # --------------------------------------------------------
    # SegMambaV2 vs SegMamba only
    # --------------------------------------------------------

    mamba_comparison_df = results_df[
        (
            results_df["candidate"]
            == "SegMambaV2"
        )
        & (
            results_df["reference"]
            == "SegMamba"
        )
    ].copy()

    mamba_output = (
        OUTPUT_DIR
        / "segmambav2_vs_segmamba.csv"
    )

    mamba_comparison_df.to_csv(
        mamba_output,
        index=False,
    )

    # --------------------------------------------------------
    # Console summary: global metrics
    # --------------------------------------------------------

    print(
        "\n"
        "========================================"
    )
    print("GLOBAL RESULTS")
    print(
        "========================================"
    )

    columns_to_print = [
        "dataset",
        "metric",
        "comparison",
        "n_pairs",
        "median_improvement",
        "rank_biserial",
        "p_raw",
        "p_holm",
        "significant_holm",
    ]

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        250,
    ):
        print(
            global_df[
                columns_to_print
            ].to_string(
                index=False
            )
        )

    print(
        "\n"
        "========================================"
    )
    print("FILES SAVED")
    print(
        "========================================"
    )

    print(full_output)
    print(global_output)
    print(significant_output)
    print(mamba_output)


if __name__ == "__main__":
    main()


