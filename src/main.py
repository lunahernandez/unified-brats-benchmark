import sys
import argparse as argparse

def preparse_dataset_model():
    """Lee --dataset/--model de sys.argv y los mete en el entorno
    ANTES de importar src.config, que los consume al importarse."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--dataset", choices=["2023", "2024"], default=None)
    pre.add_argument("--model", default=None)
    known, _ = pre.parse_known_args()

    if known.dataset is not None:
        os.environ["BRATS_DATASET"] = known.dataset
    if known.model is not None:
        os.environ["BRATS_MODEL"] = known.model.lower()

import os
preparse_dataset_model()

from pathlib import Path
import gc
import json
import math
import statistics
from typing import Any, Iterable

import torch
import torch.multiprocessing as mp
from monai.data import DataLoader, PersistentDataset

from src.config import *   # ahora ya ve las env vars seteadas
from src.data.splits import make_kfold_splits, save_split, split_train_val
from src.data.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)
from src.evaluate import evaluate_test
from src.models.get_model import get_model
from src.train import train_model
from src.utils.checkpoints import load_checkpoint
from src.utils.seed import set_seed
from src.data.dataset import get_cases_from_dirs


def safe_mean(values: Iterable[int | float | None]) -> float | None:
    """Calculate the mean while ignoring invalid values."""
    valid_values: list[float] = []

    for value in values:
        if value is None:
            continue

        value = float(value)

        if math.isnan(value):
            continue

        valid_values.append(value)

    if len(valid_values) == 0:
        return None

    return float(sum(valid_values) / len(valid_values))


def safe_std(values: Iterable[int | float | None]) -> float | None:
    """Calculate the standard deviation while ignoring invalid values."""
    valid_values: list[float] = []

    for value in values:
        if value is None:
            continue

        value = float(value)

        if math.isnan(value):
            continue

        valid_values.append(value)

    if len(valid_values) < 2:
        return 0.0 if len(valid_values) == 1 else None

    return float(statistics.stdev(valid_values))


def build_cache_dirs() -> dict[str, Path]:
    """Build the persistent cache directories."""
    base = PERSISTENT_CACHE_DIR

    return {
        "train": base / "train",
        "eval": base / "eval",
    }


def build_train_val_loaders(
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    cache_dirs: dict[str, Path],
) -> tuple[DataLoader, DataLoader]:
    """Build the training and validation data loaders."""
    train_ds = PersistentDataset(
        data=train_cases,
        transform=get_train_transforms(
            roi_size=ROI_SIZE,
            spacing=SPACING,
        ),
        cache_dir=cache_dirs["train"],
    )

    val_ds = PersistentDataset(
        data=val_cases,
        transform=get_val_transforms(
            roi_size=ROI_SIZE,
            spacing=SPACING,
        ),
        cache_dir=cache_dirs["eval"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        multiprocessing_context="spawn" if NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        multiprocessing_context="spawn" if NUM_WORKERS > 0 else None,
    )

    return train_loader, val_loader


def build_test_loader(
    test_cases: list[dict[str, Any]],
    cache_dirs: dict[str, Path],
) -> DataLoader:
    """Build the test data loader."""
    test_ds = PersistentDataset(
        data=test_cases,
        transform=get_test_transforms(
            roi_size=ROI_SIZE,
            spacing=SPACING,
        ),
        cache_dir=cache_dirs["eval"],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        multiprocessing_context="spawn" if NUM_WORKERS > 0 else None,
    )

    return test_loader


def cleanup_memory() -> None:
    """Release unused memory after a training or testing phase."""
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        torch.cuda.empty_cache()


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Save a dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists."""
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_base_results(
    fold_idx: int,
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    best_val: float | None = None,
    best_epoch: int | None = None,
    train_time_sec: float | None = None,
    train_memory_mb: float | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the base result structure for a fold."""
    return {
        "fold": fold_idx,
        "dataset": DATASET,
        "challenge_name": None,
        "model": MODEL_NAME,
        "best_val_dice": best_val,
        "best_epoch": best_epoch,
        "test_voxelwise": None,
        "test_lesionwise": None,
        "train_time_sec": train_time_sec,
        "inference_time_per_case_sec": None,
        "train_memory_mb": train_memory_mb,
        "test_memory_mb": None,
        "test_error": None,
        "config": {
            "dataset": DATASET,
            "train_dirs": [str(path) for path in TRAIN_DIRS],
            "roi_size": ROI_SIZE,
            "spacing": SPACING,
            "in_channels": IN_CHANNELS,
            "out_channels": OUT_CHANNELS,
            "batch_size": BATCH_SIZE,
            "val_batch_size": VAL_BATCH_SIZE,
            "sw_batch_size": SW_BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "clip_grad": CLIP_GRAD,
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
            "use_checkpoint": USE_CHECKPOINT,
            "num_workers": NUM_WORKERS,
            "persistent_workers": False,
            "seed": SEED + fold_idx,
            "n_folds": N_FOLDS,
            "inner_val_ratio": INNER_VAL_RATIO,
        },
        "history": history,
        "sizes": {
            "train": len(train_cases),
            "val": len(val_cases),
            "test": len(test_cases),
        },
    }


def train_fold(
    fold_idx: int,
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    include_background: bool = True,
) -> dict[str, Any]:
    """Train a model on a specific fold."""
    set_seed(SEED + fold_idx)

    print(f"DATASET: {DATASET}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"DEVICE: {DEVICE}")
    print(f"DEVICE type: {DEVICE.type}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    fold_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    save_split(
        train_cases=train_cases,
        val_cases=val_cases,
        test_cases=test_cases,
        save_path=fold_dir / "split.json",
        fold=fold_idx,
    )

    cache_dirs = build_cache_dirs()
    cache_dirs["train"].mkdir(parents=True, exist_ok=True)
    cache_dirs["eval"].mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_train_val_loaders(
        train_cases=train_cases,
        val_cases=val_cases,
        cache_dirs=cache_dirs,
    )

    model = get_model(
        model_name=MODEL_NAME,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        use_checkpoint=USE_CHECKPOINT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        optimizer=optimizer,
        max_epochs=MAX_EPOCHS,
        val_every=VAL_EVERY,
        experiment_dir=fold_dir,
        roi_size=ROI_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        clip_grad=CLIP_GRAD,
        grad_clip_max_norm=GRAD_CLIP_MAX_NORM or 1.0,
        include_background=include_background,
    )

    train_mem = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else None
    )

    model, _, best_epoch, best_val = load_checkpoint(
        model=model,
        optimizer=None,
        checkpoint_path=fold_dir / "best_model.pt",
        device=DEVICE,
    )

    fold_results = build_base_results(
        fold_idx=fold_idx,
        train_cases=train_cases,
        val_cases=val_cases,
        test_cases=test_cases,
        best_val=best_val,
        best_epoch=best_epoch,
        train_time_sec=train_info["total_train_time_sec"],
        train_memory_mb=train_mem,
        history=train_info["history"],
    )

    save_json(fold_dir / "results.json", fold_results)

    del model
    del optimizer
    del train_loader
    del val_loader
    cleanup_memory()

    return fold_results


def test_fold(
    fold_idx: int,
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the best model of a fold on the test set."""
    set_seed(SEED + fold_idx)

    fold_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    cache_dirs = build_cache_dirs()
    cache_dirs["train"].mkdir(parents=True, exist_ok=True)
    cache_dirs["eval"].mkdir(parents=True, exist_ok=True)

    test_loader = build_test_loader(
        test_cases=test_cases,
        cache_dirs=cache_dirs,
    )

    model = get_model(
        model_name=MODEL_NAME,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        use_checkpoint=USE_CHECKPOINT,
    ).to(DEVICE)

    model, _, best_epoch, best_val = load_checkpoint(
        model=model,
        optimizer=None,
        checkpoint_path=fold_dir / "best_model.pt",
        device=DEVICE,
    )

    existing_results = load_json(fold_dir / "results.json")

    if existing_results is None:
        existing_results = build_base_results(
            fold_idx=fold_idx,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
            best_val=best_val,
            best_epoch=best_epoch,
        )
        save_json(fold_dir / "results.json", existing_results)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        test_metrics = evaluate_test(
            model=model,
            test_loader=test_loader,
            device=DEVICE,
            dataset=DATASET,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            output_dir=fold_dir / "test_eval",
            model_name=MODEL_NAME,
            fold_idx=fold_idx,
        )


        test_mem = (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else None
        )

        existing_results["dataset"] = DATASET
        existing_results["challenge_name"] = test_metrics["challenge_name"]
        existing_results["best_val_dice"] = best_val
        existing_results["best_epoch"] = best_epoch
        existing_results["test_voxelwise"] = test_metrics["voxelwise"]
        existing_results["test_lesionwise"] = test_metrics["lesionwise"]
        existing_results["inference_time_per_case_sec"] = test_metrics[
            "avg_inference_time_sec"
        ]
        existing_results["test_memory_mb"] = test_mem
        existing_results["test_error"] = None

    except Exception as e:
        existing_results["test_error"] = repr(e)
        save_json(fold_dir / "results.json", existing_results)
        raise

    save_json(fold_dir / "results.json", existing_results)

    del model
    del test_loader
    cleanup_memory()

    return existing_results


def run_fold(
    fold_idx: int,
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    mode: str,
    include_background: bool = True,
) -> dict[str, Any]:
    """Run a fold according to the selected mode."""
    if mode == "train":
        return train_fold(
            fold_idx=fold_idx,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
            include_background=include_background,
        )

    if mode == "test":
        return test_fold(
            fold_idx=fold_idx,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
        )

    if mode == "all":
        train_fold(
            fold_idx=fold_idx,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
            include_background=include_background,
        )
        cleanup_memory()
        return test_fold(
            fold_idx=fold_idx,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def get_nested_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def summarize_fold_scalar(
    fold_results: list[dict[str, Any]],
    key: str,
    prefix: str = "cv",
) -> dict[str, float | None]:
    values = [fold.get(key) for fold in fold_results]

    return {
        f"{prefix}_{key}_mean": safe_mean(values),
        f"{prefix}_{key}_std": safe_std(values),
    }


def summarize_global_stat(
    fold_results: list[dict[str, Any]],
    metric_group: str,
    global_key: str,
) -> dict[str, float | None]:
    stats: dict[str, float | None] = {}

    for metric_name in ("dice", "iou", "hd95"):
        values = [
            get_nested_value(fold, metric_group, global_key, metric_name)
            for fold in fold_results
        ]

        stats[f"{metric_name}_mean"] = safe_mean(values)
        stats[f"{metric_name}_std"] = safe_std(values)

    return stats


def get_region_names_from_results(
    fold_results: list[dict[str, Any]],
) -> list[str]:
    preferred_order = ["WT", "TC", "NETC", "SNFH", "ET", "RC"]
    available_regions = set()

    for fold_result in fold_results:
        lesionwise = fold_result.get("test_lesionwise") or {}

        for group_key in ("regions", "classes"):
            group = lesionwise.get(group_key) or {}
            available_regions.update(group.keys())

    return [
        region
        for region in preferred_order
        if region in available_regions
    ]


def get_voxelwise_label_names(
    fold_results: list[dict[str, Any]],
) -> list[str]:
    preferred_order = ["WT", "TC", "NETC", "SNFH", "ET", "RC"]
    available_labels = set()

    for fold_result in fold_results:
        voxelwise = fold_result.get("test_voxelwise") or {}

        for group_key in ("regions", "classes"):
            group = voxelwise.get(group_key) or {}
            available_labels.update(group.keys())

    return [
        label
        for label in preferred_order
        if label in available_labels
    ]


def summarize_lesionwise_labels(
    fold_results: list[dict[str, Any]],
    label_names: list[str],
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}

    for label in label_names:
        dice_values = []
        hd95_values = []
        iou_values = []

        for fold_result in fold_results:
            lesionwise = fold_result.get("test_lesionwise") or {}
            by_label = lesionwise.get("regions") or {}

            if label not in by_label:
                by_label = lesionwise.get("classes") or {}

            if label not in by_label:
                continue

            dice_values.append(by_label[label].get("lesionwise_dice"))
            hd95_values.append(by_label[label].get("lesionwise_hd95"))
            iou_values.append(by_label[label].get("lesionwise_iou"))

        summary[label] = {
            "lesionwise_dice_mean": safe_mean(dice_values),
            "lesionwise_dice_std": safe_std(dice_values),
            "lesionwise_hd95_mean": safe_mean(hd95_values),
            "lesionwise_hd95_std": safe_std(hd95_values),
            "lesionwise_iou_mean": safe_mean(iou_values),
            "lesionwise_iou_std": safe_std(iou_values),
        }

    return summary


def summarize_voxelwise_labels(
    fold_results: list[dict[str, Any]],
    label_names: list[str],
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}

    for label in label_names:
        dice_values = []
        hd95_values = []
        iou_values = []

        for fold_result in fold_results:
            voxelwise = fold_result.get("test_voxelwise") or {}
            by_label = voxelwise.get("regions") or {}

            if label not in by_label:
                by_label = voxelwise.get("classes") or {}

            if label not in by_label:
                continue

            dice_values.append(by_label[label].get("dice"))
            hd95_values.append(by_label[label].get("hd95"))
            iou_values.append(by_label[label].get("iou"))

        summary[label] = {
            "voxelwise_dice_mean": safe_mean(dice_values),
            "voxelwise_dice_std": safe_std(dice_values),
            "voxelwise_hd95_mean": safe_mean(hd95_values),
            "voxelwise_hd95_std": safe_std(hd95_values),
            "voxelwise_iou_mean": safe_mean(iou_values),
            "voxelwise_iou_std": safe_std(iou_values),
        }

    return summary


def aggregate_cv_results(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "dataset": DATASET,
        "model": MODEL_NAME,
        "n_folds": N_FOLDS,
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
        summary.update(summarize_fold_scalar(fold_results, scalar_key))

    for metric_group, global_key, prefix in (
        ("test_voxelwise", "global_classes", "cv_test_voxelwise_global_classes"),
        ("test_voxelwise", "global_regions", "cv_test_voxelwise_global_regions"),
        ("test_lesionwise", "global_classes", "cv_test_lesionwise_global_classes"),
        ("test_lesionwise", "global_regions", "cv_test_lesionwise_global_regions"),
    ):
        stats = summarize_global_stat(fold_results, metric_group, global_key)

        for stat_name, value in stats.items():
            summary[f"{prefix}_{stat_name}"] = value

    lesionwise_label_names = get_region_names_from_results(fold_results)
    voxelwise_label_names = get_voxelwise_label_names(fold_results)

    summary["regions"] = summarize_lesionwise_labels(
        fold_results, lesionwise_label_names
    )
    summary["voxelwise_regions"] = summarize_voxelwise_labels(
        fold_results, voxelwise_label_names
    )

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["2023", "2024"],
        default=DATASET,
        help="Dataset a usar (ya aplicado vía variable de entorno).",
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Modelo a usar (ya aplicado vía variable de entorno).",
    )

    parser.add_argument(
        "--mode",
        choices=["train", "test", "all"],
        default="all",
        help=(
            "train: only trains, test: only evaluates, "
            "all: trains and evaluates."
        ),
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If specified, executes only that fold.",
    )

    parser.add_argument(
        "--include-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the background class in the loss function.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the main experimental workflow."""
    args = parse_args()
    set_seed(SEED)

    experiment_root = EXPERIMENTS_DIR / EXPERIMENT_NAME
    experiment_root.mkdir(parents=True, exist_ok=True)

    all_cases = get_cases_from_dirs(
        TRAIN_DIRS,
        include_label=True,
    )

    folds = make_kfold_splits(
        all_cases,
        n_splits=N_FOLDS,
        seed=SEED,
    )

    fold_results = []

    for fold_idx, (train_val_cases, test_cases) in enumerate(
        folds,
        start=1,
    ):
        if args.fold is not None and fold_idx != args.fold:
            continue

        cleanup_memory()

        train_cases, val_cases = split_train_val(
            train_val_cases,
            val_ratio=INNER_VAL_RATIO,
            seed=SEED + fold_idx,
        )

        print("=" * 70)
        print(f"Dataset: {DATASET}")
        print(f"Fold {fold_idx}/{N_FOLDS}")
        print(f"Mode: {args.mode}")
        print(
            f"Train: {len(train_cases)} | "
            f"Val: {len(val_cases)} | "
            f"Test: {len(test_cases)}"
        )

        fold_result = run_fold(
            fold_idx=fold_idx,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
            mode=args.mode,
            include_background=args.include_background,
        )

        fold_results.append(fold_result)
        cleanup_memory()

    if args.fold is None:
        cv_results = aggregate_cv_results(fold_results)

        with open(
            experiment_root / "crossval_results.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(cv_results, f, indent=2)

        print(json.dumps(cv_results, indent=2))

    else:
        print(
            f"\nFold {args.fold} completed. "
            "Individual results are saved in its folder."
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
