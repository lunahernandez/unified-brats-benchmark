from pathlib import Path
from typing import Dict, List, Optional


def get_case_data(case_dir: Path, include_label: bool = True) -> Optional[Dict]:
    """Retrieves information for a case from its directory.
    
    Args:
        case_dir: Path to the case folder.
        include_label: Indicates whether the segmentation mask should be included.

    Returns:
        A dictionary with the case identifier, the image paths,
        and, optionally, the label path. Returns ``None`` if any
        file is missing.
    """
    case_id = case_dir.name

    image_paths = []
    for seq in ["t1c", "t1n", "t2f", "t2w"]:
        image_path = case_dir / f"{case_id}-{seq}.nii.gz"
        if not image_path.exists():
            return None
        image_paths.append(str(image_path))

    case_data = {
        "case_id": case_id,
        "image": image_paths,
    }

    if include_label:
        label_path = case_dir / f"{case_id}-seg.nii.gz"
        if not label_path.exists():
            return None
        case_data["label"] = str(label_path)
        case_data["label_path"] = str(label_path)

    return case_data


def get_cases_from_dirs(directories: List[Path], include_label: bool = True) -> List[Dict]:
    """Retrieves cases information from a list of directories.
    
    Args:
        directories: List of paths to directories containing case folders.
        include_label: Indicates whether the segmentation mask should be included.

    Returns:
        A list of dictionaries with the information of each case.
    """
    cases = []
    for directory in directories:
        if not directory.exists():
            continue

        for case_dir in sorted(directory.iterdir()):
            if not case_dir.is_dir():
                continue

            case_data = get_case_data(case_dir, include_label=include_label)
            if case_data is not None:
                cases.append(case_data)

    return cases
