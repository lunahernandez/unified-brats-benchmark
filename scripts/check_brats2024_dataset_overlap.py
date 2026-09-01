from pathlib import Path
import hashlib
import csv


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_DATA1_V2 = Path(
    "data/training_data1_v2"
)

TRAINING_DATA_ADDITIONAL = Path(
    "data/training_data_additional"
)

OUTPUT_CSV = Path("results/brats2023_overlap_report.csv")


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def get_patient_folders(directory: Path) -> dict[str, Path]:
    """
    Get all patient folders contained directly inside a directory.

    The folder name is assumed to be the patient/case identifier.

    Returns:
        Dictionary:
            patient_id -> patient folder path
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    folders = {
        folder.name: folder
        for folder in directory.iterdir()
        if folder.is_dir()
    }

    return folders


def sha256_file(path: Path) -> str:
    """
    Compute SHA-256 hash of a file.
    """
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_files_relative(folder: Path) -> dict[str, Path]:
    """
    Get all files recursively inside a patient folder.

    Returns:
        Dictionary:
            relative_path -> absolute_path
    """
    return {
        str(path.relative_to(folder)): path
        for path in folder.rglob("*")
        if path.is_file()
    }


def compare_patient_folders(
    folder_1: Path,
    folder_2: Path,
) -> dict:
    """
    Compare two folders corresponding to the same patient ID.

    Checks:
        - Same file names
        - Missing files
        - SHA-256 equality for common files

    Returns:
        Dictionary with comparison information.
    """
    files_1 = get_files_relative(folder_1)
    files_2 = get_files_relative(folder_2)

    names_1 = set(files_1)
    names_2 = set(files_2)

    common_files = names_1 & names_2
    only_1 = names_1 - names_2
    only_2 = names_2 - names_1

    identical_files = []
    different_files = []

    for filename in sorted(common_files):
        hash_1 = sha256_file(files_1[filename])
        hash_2 = sha256_file(files_2[filename])

        if hash_1 == hash_2:
            identical_files.append(filename)
        else:
            different_files.append(filename)

    folders_identical = (
        not only_1
        and not only_2
        and not different_files
    )

    return {
        "folders_identical": folders_identical,
        "identical_files": identical_files,
        "different_files": different_files,
        "only_folder_1": sorted(only_1),
        "only_folder_2": sorted(only_2),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("BraTS 2024 dataset overlap analysis")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Read patient IDs
    # --------------------------------------------------------

    patients_1 = get_patient_folders(TRAINING_DATA1_V2)
    patients_2 = get_patient_folders(TRAINING_DATA_ADDITIONAL)

    ids_1 = set(patients_1)
    ids_2 = set(patients_2)

    # --------------------------------------------------------
    # 2. Compute overlap
    # --------------------------------------------------------

    overlap = ids_1 & ids_2
    only_1 = ids_1 - ids_2
    only_2 = ids_2 - ids_1
    unique_ids = ids_1 | ids_2

    print()
    print("DATASET COUNTS")
    print("-" * 70)

    print(
        f"training_data1_v2:        "
        f"{len(ids_1):>6} patient folders"
    )

    print(
        f"training_data_additional: "
        f"{len(ids_2):>6} patient folders"
    )

    print(
        f"Raw sum:                  "
        f"{len(ids_1) + len(ids_2):>6}"
    )

    print(
        f"Overlapping IDs:          "
        f"{len(overlap):>6}"
    )

    print(
        f"Unique patient IDs:       "
        f"{len(unique_ids):>6}"
    )

    print(
        f"Only in data1_v2:         "
        f"{len(only_1):>6}"
    )

    print(
        f"Only in additional:       "
        f"{len(only_2):>6}"
    )

    # --------------------------------------------------------
    # 3. Show overlapping IDs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERLAPPING PATIENT IDs")
    print("=" * 70)

    if not overlap:
        print()
        print("No overlapping patient IDs were found.")
        print()
        print(
            "Therefore, if the folder name corresponds to the unique "
            "patient identifier, the two datasets contain distinct cases."
        )

    else:
        print()
        print(
            f"{len(overlap)} patient IDs are present in BOTH folders:"
        )
        print()

        for patient_id in sorted(overlap):
            print(patient_id)

    # --------------------------------------------------------
    # 4. Compare overlapping folders file by file
    # --------------------------------------------------------

    comparison_results = []

    if overlap:
        print()
        print("=" * 70)
        print("COMPARISON OF OVERLAPPING CASES")
        print("=" * 70)

        for patient_id in sorted(overlap):

            print()
            print(f"Patient: {patient_id}")

            result = compare_patient_folders(
                patients_1[patient_id],
                patients_2[patient_id],
            )

            comparison_results.append(
                {
                    "patient_id": patient_id,
                    **result,
                }
            )

            if result["folders_identical"]:
                print("  Result: IDENTICAL")

            else:
                print("  Result: DIFFERENT")

                if result["different_files"]:
                    print("  Files with different contents:")

                    for filename in result["different_files"]:
                        print(f"    {filename}")

                if result["only_folder_1"]:
                    print("  Files only in training_data1_v2:")

                    for filename in result["only_folder_1"]:
                        print(f"    {filename}")

                if result["only_folder_2"]:
                    print("  Files only in training_data_additional:")

                    for filename in result["only_folder_2"]:
                        print(f"    {filename}")

    # --------------------------------------------------------
    # 5. Summary of overlapping cases
    # --------------------------------------------------------

    if overlap:

        identical_cases = [
            result
            for result in comparison_results
            if result["folders_identical"]
        ]

        different_cases = [
            result
            for result in comparison_results
            if not result["folders_identical"]
        ]

        print()
        print("=" * 70)
        print("OVERLAP SUMMARY")
        print("=" * 70)

        print(
            f"Overlapping IDs:                  "
            f"{len(overlap)}"
        )

        print(
            f"Completely identical cases:       "
            f"{len(identical_cases)}"
        )

        print(
            f"Cases containing differences:     "
            f"{len(different_cases)}"
        )

        if different_cases:
            print()
            print("Patients with different versions:")

            for result in different_cases:
                print(f"  {result['patient_id']}")

    # --------------------------------------------------------
    # 6. Save CSV report
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        fieldnames = [
            "patient_id",
            "in_training_data1_v2",
            "in_training_data_additional",
            "status",
            "folders_identical",
            "different_files",
            "only_data1_v2",
            "only_additional",
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        comparison_by_id = {
            result["patient_id"]: result
            for result in comparison_results
        }

        for patient_id in sorted(unique_ids):

            in_1 = patient_id in ids_1
            in_2 = patient_id in ids_2

            if in_1 and in_2:
                status = "OVERLAP"
                comparison = comparison_by_id[patient_id]

                folders_identical = comparison["folders_identical"]

                different_files = "; ".join(
                    comparison["different_files"]
                )

                only_data1 = "; ".join(
                    comparison["only_folder_1"]
                )

                only_additional = "; ".join(
                    comparison["only_folder_2"]
                )

            elif in_1:
                status = "ONLY_DATA1_V2"
                folders_identical = ""
                different_files = ""
                only_data1 = ""
                only_additional = ""

            else:
                status = "ONLY_ADDITIONAL"
                folders_identical = ""
                different_files = ""
                only_data1 = ""
                only_additional = ""

            writer.writerow(
                {
                    "patient_id": patient_id,
                    "in_training_data1_v2": in_1,
                    "in_training_data_additional": in_2,
                    "status": status,
                    "folders_identical": folders_identical,
                    "different_files": different_files,
                    "only_data1_v2": only_data1,
                    "only_additional": only_additional,
                }
            )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print()
    print(
        f"Raw number of folders: "
        f"{len(ids_1) + len(ids_2)}"
    )

    print(
        f"Number of duplicated patient IDs: "
        f"{len(overlap)}"
    )

    print(
        f"Final number of unique patient IDs: "
        f"{len(unique_ids)}"
    )

    print()
    print(f"Detailed report saved to:")
    print(OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()


