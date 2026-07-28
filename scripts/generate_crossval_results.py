from pathlib import Path
import argparse
import json
from typing import Any


from src.config import DATASET, EXPERIMENTS_DIR, EXPERIMENT_NAME, MODEL_NAME
from scripts.utils_results import load_json, safe_mean, safe_std


PREFERRED_REGION_ORDER = [
    "WT",
    "TC",
    "NETC",
    "SNFH",
    "ET",
    "RC",
]



def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Save a dictionary as a JSON file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )



def get_metric_group(
    fold: dict[str, Any],
    metric_group: str,
    label: str,
) -> dict[str, Any] | None:
    group = fold.get(metric_group)

    if not isinstance(group, dict):
        return None

    for sub_key in ("regions", "classes"):
        sub_group = group.get(sub_key)

        if isinstance(sub_group, dict) and label in sub_group:
            return sub_group[label]

    return None


def get_region_metric(
    fold: dict[str, Any],
    region: str,
    metric: str,
    metric_group: str = "test_lesionwise",
) -> float | None:
    label_data = get_metric_group(fold, metric_group, region)

    if label_data is None:
        return None

    return label_data.get(metric)


def collect_fold_results(
    experiment_root: Path,
) -> list[dict[str, Any]]:
    fold_results = []

    for fold_dir in sorted(experiment_root.glob("fold_*")):
        results_path = fold_dir / "results.json"

        if not results_path.exists():
            print(f"Skipping {fold_dir.name}: results.json not found")
            continue

        result = load_json(results_path)

        if result is not None:
            fold_results.append(result)

        print(f"Loaded {results_path}")

    return sorted(fold_results, key=lambda x: x.get("fold", 0))


def get_dataset_from_results(
    fold_results: list[dict[str, Any]],
) -> str:
    datasets = {
        fold.get("dataset")
        for fold in fold_results
        if fold.get("dataset") is not None
    }

    if len(datasets) == 1:
        return str(next(iter(datasets)))

    return str(DATASET)




def get_region_names_from_results(
    fold_results: list[dict[str, Any]],
    metric_group: str = "test_lesionwise",
) -> list[str]:
    available_regions = set()

    for fold in fold_results:
        group = fold.get(metric_group)

        if not isinstance(group, dict):
            continue

        for sub_key in ("regions", "classes"):
            sub_group = group.get(sub_key)

            if isinstance(sub_group, dict):
                available_regions.update(sub_group.keys())

    ordered_regions = [
        region for region in PREFERRED_REGION_ORDER if region in available_regions
    ]

    extra_regions = sorted(
        region for region in available_regions if region not in PREFERRED_REGION_ORDER
    )

    return ordered_regions + extra_regions


def summarize_global_stats(
    fold_results: list[dict[str, Any]],
    metric_group: str,
    global_key: str,
) -> dict[str, float | None]:
    stats = {}

    for metric_name in ("dice", "iou", "hd95"):
        values = [
            ((fold.get(metric_group) or {}).get(global_key) or {}).get(metric_name)
            for fold in fold_results
        ]

        stats[f"{metric_name}_mean"] = safe_mean(values)
        stats[f"{metric_name}_std"] = safe_std(values)

    return stats


def summarize_labels(
    fold_results: list[dict[str, Any]],
    region_names: list[str],
    metric_group: str,
    field_prefix: str,
    output_prefix: str,
) -> dict[str, dict[str, float | None]]:
    summary = {}

    for region in region_names:
        dice_values = [
            get_region_metric(fold, region, f"{field_prefix}dice", metric_group)
            for fold in fold_results
        ]
        iou_values = [
            get_region_metric(fold, region, f"{field_prefix}iou", metric_group)
            for fold in fold_results
        ]
        hd95_values = [
            get_region_metric(fold, region, f"{field_prefix}hd95", metric_group)
            for fold in fold_results
        ]

        summary[region] = {
            f"{output_prefix}dice_mean": safe_mean(dice_values),
            f"{output_prefix}dice_std": safe_std(dice_values),
            f"{output_prefix}iou_mean": safe_mean(iou_values),
            f"{output_prefix}iou_std": safe_std(iou_values),
            f"{output_prefix}hd95_mean": safe_mean(hd95_values),
            f"{output_prefix}hd95_std": safe_std(hd95_values),
        }

    return summary


def aggregate_cv_results(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset = get_dataset_from_results(fold_results)

    summary = {
        "dataset": dataset,
        "model": fold_results[0].get("model", MODEL_NAME),
        "n_folds": len(fold_results),
        "folds": fold_results,
        "cv_best_val_dice_mean": safe_mean(
            [fold.get("best_val_dice") for fold in fold_results]
        ),
        "cv_best_val_dice_std": safe_std(
            [fold.get("best_val_dice") for fold in fold_results]
        ),
    }

    for scalar_key in (
        "train_time_sec",
        "inference_time_per_case_sec",
        "train_memory_mb",
        "test_memory_mb",
    ):
        values = [fold.get(scalar_key) for fold in fold_results]
        summary[f"cv_{scalar_key}_mean"] = safe_mean(values)
        summary[f"cv_{scalar_key}_std"] = safe_std(values)

    for metric_group, global_key, prefix in (
        ("test_voxelwise", "global_classes", "cv_test_voxelwise_global_classes"),
        ("test_voxelwise", "global_regions", "cv_test_voxelwise_global_regions"),
        ("test_lesionwise", "global_classes", "cv_test_lesionwise_global_classes"),
        ("test_lesionwise", "global_regions", "cv_test_lesionwise_global_regions"),
    ):
        stats = summarize_global_stats(fold_results, metric_group, global_key)

        for stat_name, value in stats.items():
            summary[f"{prefix}_{stat_name}"] = value

    lesionwise_region_names = get_region_names_from_results(
        fold_results, metric_group="test_lesionwise"
    )
    voxelwise_region_names = get_region_names_from_results(
        fold_results, metric_group="test_voxelwise"
    )

    summary["regions"] = summarize_labels(
        fold_results, lesionwise_region_names, "test_lesionwise", "lesionwise_", "lesionwise_"
    )
    summary["voxelwise_regions"] = summarize_labels(
        fold_results, voxelwise_region_names, "test_voxelwise", "", "voxelwise_"
    )

    return summary




def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENTS_DIR / EXPERIMENT_NAME,
        help="Experiment folder containing fold_* directories.",
    )


    return parser.parse_args()



def main() -> None:

    args = parse_args()


    fold_results = collect_fold_results(
        args.experiment_root
    )


    if not fold_results:
        raise FileNotFoundError(
            f"No fold results found in {args.experiment_root}"
        )


    cv_results = aggregate_cv_results(
        fold_results
    )


    output_path = (
        args.experiment_root /
        "crossval_results.json"
    )


    save_json(
        output_path,
        cv_results,
    )


    print(
        f"\nSaved: {output_path}"
    )

    print(
        json.dumps(
            cv_results,
            indent=2,
        )
    )



if __name__ == "__main__":
    main()



# Example:
#
# python -m scripts.generate_crossval_results \
# --experiment-root experiments/brats2024_segresnet_roi128_bs1_nworkers4_cv5_without_background/


