import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def save_split(
    train_cases: List[Dict],
    val_cases: List[Dict],
    test_cases: List[Dict],
    save_path: str | Path,
    fold: Optional[int] = None,
) -> None:
    """Saves the partition of a fold to a JSON file.

    Args:
        train_cases: List of training cases.
        val_cases: List of validation cases.
        test_cases: List of testing cases.
        save_path: Path where the JSON file will be saved.
        fold: Fold identifier.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    split_dict = {
        "fold": fold,
        "train": [case["case_id"] for case in train_cases],
        "val": [case["case_id"] for case in val_cases],
        "test": [case["case_id"] for case in test_cases],
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)


def filter_cases_by_ids(cases: List[Dict], case_ids: List[str]) -> List[Dict]:
    """Filters the cases whose identifier is in a given list.

    Args:
        cases: List of cases.
        case_ids: List of case identifiers to keep.

    Returns:
        List of filtered cases.
    """
    case_ids = set(case_ids)
    return [case for case in cases if case["case_id"] in case_ids]


def split_train_val(
    cases: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Splits the cases into training and validation.

    Args:
        cases: List of cases to split.
        val_ratio: Proportion of validation cases.
        seed: Seed used to shuffle the cases.

    Returns:
        A tuple with the training and validation lists.

    Raises:
        ValueError: If ``val_ratio`` is not between 0 and 1.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    random_generator = random.Random(seed)
    shuffled_cases = cases.copy()
    random_generator.shuffle(shuffled_cases)

    n_val = max(1, int(len(shuffled_cases) * val_ratio))
    n_val = min(n_val, len(shuffled_cases) - 1) if len(shuffled_cases) > 1 else 0

    val_cases = shuffled_cases[:n_val]
    train_cases = shuffled_cases[n_val:]

    return train_cases, val_cases


def make_kfold_splits(
    cases: List[Dict],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Dict], List[Dict]]]:
    """Generates partitions for k-fold cross-validation.

    Args:
        cases: List of cases to split.
        n_splits: Number of folds.
        seed: Seed used to shuffle the cases.

    Returns:
        A list of tuples ``(train_val_cases, test_cases)`` for each fold,
        where train_val_cases is the list of cases used for training
        and validation, and test_cases is the list of cases used for testing.

    Raises:
        ValueError: If ``n_splits`` is less than 2.
        ValueError: If there are not enough cases to generate the folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if len(cases) < n_splits:
        raise ValueError("There are not enough cases for so many folds.")

    random_generator = random.Random(seed)
    indices = list(range(len(cases)))
    random_generator.shuffle(indices)

    fold_sizes = [len(cases) // n_splits] * n_splits
    for i in range(len(cases) % n_splits):
        fold_sizes[i] += 1

    folds = []
    current = 0

    for fold_size in fold_sizes:
        test_idx = set(indices[current:current + fold_size])
        current += fold_size

        train_val_cases = [cases[i] for i in range(len(cases)) if i not in test_idx]
        test_cases = [cases[i] for i in range(len(cases)) if i in test_idx]

        folds.append((train_val_cases, test_cases))

    return folds