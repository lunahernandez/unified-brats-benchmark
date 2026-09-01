"""
Generate publication-ready visualizations for BraTS 2023 or BraTS 2024.

For each dataset, the script generates:
1. MRI modalities + binary segmentation mask
2. Sagittal, coronal, and axial views + labeled segmentation mask
3. Sex and age distributions
4. Meningioma grade distribution or glioma type distribution
5. Tumor subregion frequency
6. Tumor subregion volume distribution
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba


# ---------------------------------------------------------------------
# PUBLICATION STYLE
# ---------------------------------------------------------------------


def configure_publication_style() -> None:
    """Configure readable typography for exported figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 22,
            "axes.titlesize": 28,
            "axes.titleweight": "semibold",
            "axes.titlepad": 12,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "figure.titlesize": 32,
            "figure.titleweight": "bold",
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------
# GENERAL UTILITIES
# ---------------------------------------------------------------------


def load_volume(
    case_dir: Path,
    patient_id: str,
    suffix: str,
) -> np.ndarray:
    """
    Load a NIfTI volume following the BraTS naming convention.

    Expected filename:
        <patient_id>-<suffix>.nii.gz
    """
    file_path = case_dir / f"{patient_id}-{suffix}.nii.gz"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return nib.load(file_path).get_fdata()


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D image slice using the 1st and 99th percentiles.
    """
    valid_values = image[np.isfinite(image)]

    if valid_values.size == 0:
        return np.zeros_like(image, dtype=float)

    lower, upper = np.percentile(valid_values, [1, 99])

    if upper <= lower:
        return np.zeros_like(image, dtype=float)

    normalized = (image - lower) / (upper - lower)

    return np.clip(normalized, 0, 1)


def normalize_key(value: str) -> str:
    """
    Normalize strings to improve clinical-column matching.
    """
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("utf-8")
    value = re.sub(r"[^a-z0-9]+", "", value)

    return value


def find_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str:
    """
    Find a dataframe column using normalized case-insensitive matching.
    """
    normalized_columns = {
        normalize_key(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        candidate_key = normalize_key(candidate)

        if candidate_key in normalized_columns:
            return normalized_columns[candidate_key]

    raise KeyError(
        "Could not identify a suitable column.\n"
        f"Expected one of: {list(candidates)}\n"
        f"Available columns: {list(dataframe.columns)}"
    )


def read_clinical_file(file_path: Path) -> pd.DataFrame:
    """
    Read clinical metadata from CSV, XLSX, or XLS.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Clinical file not found: {file_path}"
        )

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raise ValueError(
        "The clinical file must be CSV, XLSX, or XLS."
    )


def save_figure(
    fig: plt.Figure,
    output_path: Path,
) -> None:
    """
    Save a figure as SVG and PDF.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path.with_suffix(".svg"),
        format="svg",
        bbox_inches="tight",
        pad_inches=0.12,
    )

    fig.savefig(
        output_path.with_suffix(".pdf"),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.12,
    )

    plt.close(fig)


def get_axial_tumor_slice(mask: np.ndarray) -> int:
    """
    Select the axial slice containing the greatest number
    of segmented voxels.
    """
    segmented_voxels_per_slice = np.sum(
        mask > 0,
        axis=(0, 1),
    )

    if segmented_voxels_per_slice.max() == 0:
        return mask.shape[2] // 2

    return int(
        np.argmax(segmented_voxels_per_slice)
    )


def get_plane_indices(
    mask: np.ndarray,
) -> tuple[int, int, int]:
    """
    Select sagittal, coronal, and axial slices with
    the largest tumor area.
    """
    sagittal_idx = int(
        np.argmax(
            np.sum(mask > 0, axis=(1, 2))
        )
    )

    coronal_idx = int(
        np.argmax(
            np.sum(mask > 0, axis=(0, 2))
        )
    )

    axial_idx = int(
        np.argmax(
            np.sum(mask > 0, axis=(0, 1))
        )
    )

    return sagittal_idx, coronal_idx, axial_idx


def get_label_configuration(
    dataset: str,
) -> tuple[dict[int, str], dict[int, str]]:
    """
    Return segmentation label names and colors
    for each BraTS dataset.
    """
    if dataset == "2023":
        label_names = {
            1: "NETC",
            2: "SNFH",
            3: "ET",
        }

        label_colors = {
            1: "#ff0000",
            2: "#00ff00",
            3: "#0000ff",
        }

        return label_names, label_colors

    if dataset == "2024":
        label_names = {
            1: "NETC",
            2: "SNFH",
            3: "ET",
            4: "RC",
        }

        label_colors = {
            1: "#FF5C5C",
            2: "#66F466",
            3: "#7070DC",
            4: "#FFFF5C",
        }

        return label_names, label_colors

    raise ValueError(
        "Dataset must be '2023' or '2024'."
    )


# ---------------------------------------------------------------------
# DATASET / CLINICAL METADATA CONSISTENCY
# ---------------------------------------------------------------------


def get_dataset_patient_ids(
    data_dirs: list[Path],
) -> set[str]:
    """
    Collect unique patient IDs from the dataset directories.

    Patient IDs are assumed to correspond to the patient-folder names.
    """
    patient_ids: set[str] = set()
    duplicated_ids: set[str] = set()

    total_folders = 0

    print()
    print("=" * 70)
    print("DATASET PATIENT-ID CHECK")
    print("=" * 70)

    for data_dir in data_dirs:
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {data_dir}"
            )

        directory_count = 0

        for case_dir in data_dir.iterdir():
            if not case_dir.is_dir():
                continue

            directory_count += 1
            total_folders += 1

            patient_id = case_dir.name.strip()

            if patient_id in patient_ids:
                duplicated_ids.add(patient_id)

            patient_ids.add(patient_id)

        print(
            f"{data_dir.name}: {directory_count} patient folders"
        )

    print(f"Raw patient folders: {total_folders}")
    print(
        f"Unique patient IDs: {len(patient_ids)}"
    )
    print(
        "Patient IDs appearing in more than one directory: "
        f"{len(duplicated_ids)}"
    )

    if duplicated_ids:
        print("\nDuplicated IDs:")

        for patient_id in sorted(duplicated_ids):
            print(f"  {patient_id}")

    return patient_ids


def filter_clinical_to_dataset(
    clinical_df: pd.DataFrame,
    data_dirs: list[Path],
    patient_id_column: str | None = None,
) -> pd.DataFrame:
    """
    Restrict clinical metadata to subjects actually present
    in the dataset directories.

    Also report:
        - total clinical rows
        - unique clinical IDs
        - IDs present only in the clinical file
        - dataset IDs missing from the clinical file
        - duplicated IDs inside the final cohort
    """
    if patient_id_column is None:
        patient_id_column = find_column(
            clinical_df,
            [
                "Patient ID",
                "PatientID",
                "Subject ID",
                "SubjectID",
                "BraTS ID",
                "BraTS Subject ID",
                "BraTS 2024 Subject ID",
                "Case ID",
            ],
        )

    dataset_ids = get_dataset_patient_ids(
        data_dirs
    )

    metadata = clinical_df.copy()

    metadata[patient_id_column] = (
        metadata[patient_id_column]
        .astype("string")
        .str.strip()
    )

    metadata_with_id = metadata[
        metadata[patient_id_column].notna()
        & (metadata[patient_id_column] != "")
    ].copy()

    clinical_ids = set(
        metadata_with_id[
            patient_id_column
        ].tolist()
    )

    extra_clinical_ids = (
        clinical_ids - dataset_ids
    )

    missing_clinical_ids = (
        dataset_ids - clinical_ids
    )

    filtered_df = metadata_with_id[
        metadata_with_id[
            patient_id_column
        ].isin(dataset_ids)
    ].copy()

    duplicated_filtered = filtered_df[
        filtered_df[
            patient_id_column
        ].duplicated(keep=False)
    ].sort_values(
        patient_id_column
    )

    print()
    print("=" * 70)
    print("CLINICAL METADATA CHECK")
    print("=" * 70)

    print(
        f"Clinical ID column: "
        f"{patient_id_column}"
    )

    print(
        f"Rows in original clinical file: "
        f"{len(clinical_df)}"
    )

    print(
        f"Rows with a valid patient ID: "
        f"{len(metadata_with_id)}"
    )

    print(
        f"Unique patient IDs in clinical file: "
        f"{len(clinical_ids)}"
    )

    print(
        f"Unique patient IDs in dataset: "
        f"{len(dataset_ids)}"
    )

    print(
        f"Clinical IDs not present in dataset: "
        f"{len(extra_clinical_ids)}"
    )

    print(
        f"Dataset IDs without clinical metadata: "
        f"{len(missing_clinical_ids)}"
    )

    print(
        f"Rows after restricting metadata to dataset: "
        f"{len(filtered_df)}"
    )

    print(
        "Duplicated patient IDs inside filtered cohort: "
        f"{duplicated_filtered[patient_id_column].nunique()}"
    )

    if extra_clinical_ids:
        print()
        print(
            "Clinical IDs outside the experimental cohort:"
        )

        for patient_id in sorted(
            extra_clinical_ids
        ):
            print(f"  {patient_id}")

    if missing_clinical_ids:
        print()
        print(
            "Dataset IDs without clinical metadata:"
        )

        for patient_id in sorted(
            missing_clinical_ids
        ):
            print(f"  {patient_id}")

    if not duplicated_filtered.empty:
        print()
        print(
            "Duplicated IDs inside the experimental cohort:"
        )

        print(
            duplicated_filtered[
                [patient_id_column]
            ].to_string(index=False)
        )

        raise ValueError(
            "Duplicated patient IDs were found in the "
            "clinical metadata for subjects in the "
            "experimental cohort. Inspect these records "
            "before generating the clinical figures."
        )

    return filtered_df


# ---------------------------------------------------------------------
# FIGURE 1: MRI MODALITIES + BINARY MASK
# ---------------------------------------------------------------------


def find_case_dir(
    data_dirs: list[Path],
    patient_id: str,
) -> Path:
    """
    Find the folder of a patient across one or more dataset directories.
    """
    for data_dir in data_dirs:
        case_dir = data_dir / patient_id

        if case_dir.is_dir():
            return case_dir

    searched_dirs = "\n".join(
        f" - {path}"
        for path in data_dirs
    )

    raise FileNotFoundError(
        f"Patient folder not found for: {patient_id}\n"
        f"Searched directories:\n{searched_dirs}"
    )


def plot_modalities_with_mask(
    data_dirs: list[Path],
    patient_id: str,
    output_dir: Path,
    dataset: str,
) -> None:
    """
    Create a 2x3 figure with MRI modalities, binary mask,
    and a single-color overlay on post-contrast T1w.
    """
    case_dir = find_case_dir(
        data_dirs,
        patient_id,
    )

    volumes = {
        "Pre-contrast T1w": load_volume(
            case_dir,
            patient_id,
            "t1n",
        ),
        "Post-contrast T1w": load_volume(
            case_dir,
            patient_id,
            "t1c",
        ),
        "T2w": load_volume(
            case_dir,
            patient_id,
            "t2w",
        ),
        "T2-FLAIR": load_volume(
            case_dir,
            patient_id,
            "t2f",
        ),
        "Segmentation mask": load_volume(
            case_dir,
            patient_id,
            "seg",
        ),
    }

    segmentation = volumes[
        "Segmentation mask"
    ]

    slice_idx = get_axial_tumor_slice(
        segmentation
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(30, 18),
    )

    axes = axes.ravel()

    display_names = [
        "Pre-contrast T1w",
        "Post-contrast T1w",
        "T2w",
        "T2-FLAIR",
        "Segmentation mask",
    ]

    for index, name in enumerate(
        display_names
    ):
        image_slice = volumes[name][
            :,
            :,
            slice_idx,
        ]

        if name == "Segmentation mask":
            binary_mask = (
                image_slice > 0
            ).astype(float)

            axes[index].imshow(
                np.rot90(binary_mask),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
                origin="upper",
            )

        else:
            axes[index].imshow(
                np.rot90(
                    normalize_for_display(
                        image_slice
                    )
                ),
                cmap="gray",
                origin="upper",
            )

        axes[index].set_title(
            name,
            fontsize=40,
        )

        axes[index].axis("off")

    t1c_slice = volumes[
        "Post-contrast T1w"
    ][:, :, slice_idx]

    mask_slice = segmentation[
        :,
        :,
        slice_idx,
    ]

    axes[5].imshow(
        np.rot90(
            normalize_for_display(
                t1c_slice
            )
        ),
        cmap="gray",
        origin="upper",
    )

    overlay_rgba = np.zeros(
        mask_slice.shape + (4,),
        dtype=float,
    )

    overlay_rgba[..., :3] = (
        to_rgba("#9191E9")[:3]
    )

    overlay_rgba[..., 3] = np.where(
        mask_slice > 0,
        0.70,
        0.0,
    )

    axes[5].imshow(
        np.rot90(overlay_rgba),
        interpolation="nearest",
        origin="upper",
        zorder=2,
    )

    axes[5].set_title(
        "Mask over post-contrast T1w",
        fontsize=40,
    )

    axes[5].axis("off")

    # No global "BraTS 2023/2024" title.

    fig.subplots_adjust(
        left=0.03,
        right=0.98,
        bottom=0.05,
        top=0.89,
        wspace=0.15,
        hspace=0.15,
    )

    save_figure(
        fig,
        output_dir
        / f"{patient_id}_modalities_mask",
    )


# ---------------------------------------------------------------------
# FIGURE 2: SAGITTAL, CORONAL, AXIAL VIEWS + MASK
# ---------------------------------------------------------------------


def plot_planes_with_mask(
    data_dirs: list[Path],
    patient_id: str,
    output_dir: Path,
    dataset: str,
) -> None:
    """
    Create sagittal, coronal, and axial post-contrast T1w views
    with segmentation masks.
    """
    case_dir = find_case_dir(
        data_dirs,
        patient_id,
    )

    segmentation = load_volume(
        case_dir,
        patient_id,
        "seg",
    )

    t1c = load_volume(
        case_dir,
        patient_id,
        "t1c",
    )

    (
        sagittal_idx,
        coronal_idx,
        axial_idx,
    ) = get_plane_indices(
        segmentation
    )

    views = [
        (
            "Sagittal view",
            sagittal_idx,
            t1c[
                sagittal_idx,
                :,
                :,
            ].T,
            segmentation[
                sagittal_idx,
                :,
                :,
            ].T,
        ),
        (
            "Coronal view",
            coronal_idx,
            t1c[
                :,
                coronal_idx,
                :,
            ].T,
            segmentation[
                :,
                coronal_idx,
                :,
            ].T,
        ),
        (
            "Axial view",
            axial_idx,
            t1c[
                :,
                :,
                axial_idx,
            ].T,
            segmentation[
                :,
                :,
                axial_idx,
            ].T,
        ),
    ]

    (
        label_names,
        label_colors,
    ) = get_label_configuration(
        dataset
    )

    sorted_labels = sorted(
        label_colors.keys()
    )

    colormap = ListedColormap(
        [
            label_colors[label]
            for label in sorted_labels
        ]
    )

    boundaries = np.arange(
        min(sorted_labels) - 0.5,
        max(sorted_labels) + 1.5,
        1,
    )

    norm = BoundaryNorm(
        boundaries,
        colormap.N,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 10),
    )

    for (
        ax,
        (
            title,
            index,
            image_slice,
            mask_slice,
        ),
    ) in zip(
        axes,
        views,
    ):
        ax.imshow(
            normalize_for_display(
                image_slice
            ),
            cmap="gray",
            origin="lower",
        )

        masked_segmentation = (
            np.ma.masked_where(
                mask_slice == 0,
                mask_slice,
            )
        )

        ax.imshow(
            masked_segmentation,
            cmap=colormap,
            norm=norm,
            alpha=0.58,
            origin="lower",
            interpolation="none",
        )

        ax.set_title(
            f"{title} ({index})",
            fontsize=34,
        )

        ax.axis("off")

    legend_handles = [
        mpatches.Patch(
            color=label_colors[label],
            label=label_names[label],
            alpha=0.75,
        )
        for label in sorted_labels
    ]

    fig.legend(
        handles=legend_handles,
        title="Tumor subregions",
        loc="lower center",
        ncol=len(sorted_labels),
        fontsize=34,
        title_fontsize=36,
        frameon=False,
    )

    # No global "BraTS 2023/2024" title.

    fig.subplots_adjust(
        left=0.03,
        right=0.98,
        bottom=0.18,
        top=0.86,
        wspace=0.06,
    )

    save_figure(
        fig,
        output_dir
        / f"{patient_id}_multiplanar_mask",
    )


# ---------------------------------------------------------------------
# FIGURE 3: SEX AND AGE DISTRIBUTIONS
# ---------------------------------------------------------------------


def translate_sex_values(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize common sex labels to English.
    """
    mapping = {
        "f": "Female",
        "female": "Female",
        "m": "Male",
        "male": "Male",
        "woman": "Female",
        "man": "Male",
        "mujer": "Female",
        "hombre": "Male",
    }

    return (
        series
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            lambda value: mapping.get(
                value,
                value.title(),
            )
        )
    )


def plot_sex_and_age(
    clinical_df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    sex_column: str | None = None,
    age_column: str | None = None,
) -> None:
    """
    Create sex distribution and age distribution plots.
    """
    if sex_column is None:
        sex_column = find_column(
            clinical_df,
            [
                "Sex",
                "Gender",
                "Patient's Sex",
                "Patient Sex",
            ],
        )

    if age_column is None:
        age_column = find_column(
            clinical_df,
            [
                "Age",
                "Patient's Age",
                "Patient Age",
            ],
        )

    sex_data = translate_sex_values(
        clinical_df[sex_column]
    )

    age_data = pd.to_numeric(
        clinical_df[age_column],
        errors="coerce",
    ).dropna()

    fig, (
        ax_sex,
        ax_age,
    ) = plt.subplots(
        1,
        2,
        figsize=(15, 6.5),
    )

    sex_counts = (
        sex_data.value_counts()
    )

    ax_sex.pie(
        sex_counts.values,
        labels=sex_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={
            "fontsize": 22,
        },
    )

    ax_sex.set_title(
        "Sex distribution",
        fontsize=22,
    )

    ax_sex.axis("equal")

    ax_age.hist(
        age_data,
        bins=15,
        edgecolor="white",
        linewidth=1.0,
    )

    ax_age.set_title(
        "Age distribution",
        fontsize=22,
    )

    ax_age.set_xlabel(
        "Age (years)"
    )

    ax_age.set_ylabel(
        "Number of cases"
    )

    ax_age.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    ax_age.set_axisbelow(True)

    # No global "BraTS 2023/2024" title.

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.12,
        top=0.86,
        wspace=0.28,
    )

    filename = (
        f"{dataset_name.lower().replace(' ', '_')}"
        "_sex_age"
    )

    save_figure(
        fig,
        output_dir / filename,
    )


# ---------------------------------------------------------------------
# FIGURE 4: MENINGIOMA GRADE OR GLIOMA TYPE
# ---------------------------------------------------------------------


def plot_category_distribution(
    clinical_df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    category_column: str,
    title: str,
    category_type: str,
) -> None:
    """
    Create a bar chart for meningioma grade or glioma type.
    """
    if category_type == "grade":
        data = pd.to_numeric(
            clinical_df[
                category_column
            ],
            errors="coerce",
        ).dropna().astype(int)

        counts = (
            data
            .value_counts()
            .sort_index()
        )

        x_labels = [
            f"Grade {grade}"
            for grade in counts.index
        ]

        filename = "meningioma_grade"

    elif category_type == "type":
        data = (
            clinical_df[
                category_column
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        data = (
            clinical_df[category_column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        data = data.replace(
            {
                "glioma NOS": "Glioma NOS",
                "Glioma nos": "Glioma NOS",
                "glioma nos": "Glioma NOS",
            }
        )

        counts = data.value_counts()

        x_labels = (
            counts.index.tolist()
        )

        filename = "glioma_type"

    else:
        raise ValueError(
            "category_type must be "
            "'grade' or 'type'."
        )

    discrete_palette = [
        "#f25f5c",
        "#ffe066",
        "#247ba0",
        "#70c1b3",
        "#d9bbf9",
        "#5d4954",
    ]

    bar_colors = [
        discrete_palette[
            index % len(discrete_palette)
        ]
        for index in range(
            len(counts)
        )
    ]

    # Keep the same publication size as the previous figure.
    fig, ax = plt.subplots(
        figsize=(15, 9.5)
    )

    bars = ax.bar(
        x_labels,
        counts.values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )

    # Only the actual figure title.
    # No "BraTS 2024" suptitle.
    ax.set_title(
        title,
        fontsize=32,
    )

    ax.set_ylabel(
        "Number of cases",
        fontsize=24,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    ax.set_axisbelow(True)

    maximum = max(
        counts.values
    )

    for bar, value in zip(
        bars,
        counts.values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + maximum * 0.025,
            str(value),
            ha="center",
            va="bottom",
            fontsize=28,
            fontweight="semibold",
        )

    ax.set_ylim(
        0,
        maximum * 1.20,
    )

    if category_type == "type":
        ax.tick_params(
            axis="x",
            labelrotation=20,
            labelsize=28,
        )

        for label in (
            ax.get_xticklabels()
        ):
            label.set_horizontalalignment(
                "right"
            )

    else:
        ax.tick_params(
            axis="x",
            labelsize=28,
        )

    ax.tick_params(
        axis="y",
        labelsize=28,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=(
            0.18
            if category_type == "type"
            else 0.12
        ),
        top=0.85,
    )

    save_figure(
        fig,
        output_dir / filename,
    )


# ---------------------------------------------------------------------
# FIGURES 5 AND 6:
# TUMOR SUBREGION FREQUENCY AND VOLUME DISTRIBUTION
# ---------------------------------------------------------------------


def collect_subregion_statistics(
    data_dirs: list[Path],
    dataset: str,
) -> pd.DataFrame:
    """
    Compute voxel counts and case-level presence
    across multiple dataset paths.
    """
    label_names, _ = (
        get_label_configuration(
            dataset
        )
    )

    records: list[
        dict[str, int | bool | str]
    ] = []

    for data_dir in data_dirs:
        if not data_dir.exists():
            raise FileNotFoundError(
                "Dataset directory not found: "
                f"{data_dir}"
            )

        for case_dir in sorted(
            data_dir.iterdir()
        ):
            if not case_dir.is_dir():
                continue

            patient_id = (
                case_dir.name
            )

            try:
                segmentation = load_volume(
                    case_dir,
                    patient_id,
                    "seg",
                )

            except FileNotFoundError:
                continue

            record: dict[
                str,
                int | bool | str,
            ] = {
                "Patient ID": patient_id,
                "Source": data_dir.name,
            }

            for (
                label_id,
                label_name,
            ) in label_names.items():
                voxel_count = int(
                    np.count_nonzero(
                        segmentation
                        == label_id
                    )
                )

                record[
                    f"{label_name} voxels"
                ] = voxel_count

                record[
                    f"{label_name} present"
                ] = (
                    voxel_count > 0
                )

            records.append(
                record
            )

    if not records:
        raise ValueError(
            "No valid segmentation masks "
            "were found in the provided "
            "dataset directories."
        )

    return pd.DataFrame(
        records
    )


def add_bar_labels(
    ax: plt.Axes,
    bars,
    values: list[float],
    offset: float,
) -> None:
    """
    Add percentage labels above bars.
    """
    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + offset,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=34,
            fontweight="semibold",
        )


def plot_subregion_frequency(
    subregion_df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> None:
    """
    Plot percentage of cases in which each
    tumor subregion is present.
    """
    (
        label_names,
        label_colors,
    ) = get_label_configuration(
        dataset
    )

    label_ids = sorted(
        label_names
    )

    labels = [
        label_names[label_id]
        for label_id in label_ids
    ]

    colors = [
        label_colors[label_id]
        for label_id in label_ids
    ]

    percentages = [
        100
        * subregion_df[
            f"{label_name} present"
        ].mean()
        for label_name in labels
    ]

    fig, ax = plt.subplots(
        figsize=(15, 9.5)
    )

    bars = ax.bar(
        labels,
        percentages,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title(
        "Tumor subregion frequency",
        fontsize=36,
    )

    ax.set_ylabel(
        "Percentage of cases (%)",
        fontsize=34,
    )

    ax.tick_params(
        axis="both",
        labelsize=34,
    )

    ax.set_ylim(
        0,
        115,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    ax.set_axisbelow(True)

    add_bar_labels(
        ax=ax,
        bars=bars,
        values=percentages,
        offset=1.5,
    )

    # No global BraTS title.

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.12,
        top=0.85,
    )

    save_figure(
        fig,
        output_dir
        / (
            f"brats{dataset}"
            "_subregion_frequency"
        ),
    )


def plot_subregion_volume_distribution(
    subregion_df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> None:
    """
    Plot global volume share represented by each
    tumor subregion.
    """
    (
        label_names,
        label_colors,
    ) = get_label_configuration(
        dataset
    )

    label_ids = sorted(
        label_names
    )

    labels = [
        label_names[label_id]
        for label_id in label_ids
    ]

    colors = [
        label_colors[label_id]
        for label_id in label_ids
    ]

    voxel_totals = [
        int(
            subregion_df[
                f"{label_name} voxels"
            ].sum()
        )
        for label_name in labels
    ]

    total_segmented_voxels = sum(
        voxel_totals
    )

    if total_segmented_voxels == 0:
        percentages = [
            0.0
        ] * len(labels)

    else:
        percentages = [
            100
            * voxel_count
            / total_segmented_voxels
            for voxel_count
            in voxel_totals
        ]

    fig, ax = plt.subplots(
        figsize=(15, 9.5)
    )

    bars = ax.bar(
        labels,
        percentages,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title(
        "Tumor subregion volume distribution",
        fontsize=36,
    )

    ax.set_ylabel(
        "Segmented tumor volume (%)",
        fontsize=34,
    )

    ax.tick_params(
        axis="both",
        labelsize=34,
    )

    ax.set_ylim(
        0,
        max(
            percentages,
            default=0,
        )
        + 12,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    ax.set_axisbelow(True)

    add_bar_labels(
        ax=ax,
        bars=bars,
        values=percentages,
        offset=0.8,
    )

    # No global BraTS title.

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.12,
        top=0.85,
    )

    save_figure(
        fig,
        output_dir
        / (
            f"brats{dataset}"
            "_subregion_volume_distribution"
        ),
    )


def generate_subregion_eda_figures(
    data_dirs: list[Path],
    dataset: str,
    output_dir: Path,
) -> None:
    """
    Generate subregion frequency and volume-distribution figures.
    """
    subregion_df = (
        collect_subregion_statistics(
            data_dirs=data_dirs,
            dataset=dataset,
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subregion_df.to_csv(
        output_dir
        / (
            f"brats{dataset}"
            "_subregion_statistics.csv"
        ),
        index=False,
    )

    plot_subregion_frequency(
        subregion_df=subregion_df,
        dataset=dataset,
        output_dir=output_dir,
    )

    plot_subregion_volume_distribution(
        subregion_df=subregion_df,
        dataset=dataset,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------
# BRA TS 2023 EXECUTION
# ---------------------------------------------------------------------


def run_brats2023(
    args: argparse.Namespace,
) -> None:
    """
    Generate BraTS 2023 visualizations.
    """
    output_dir = (
        args.output_dir
        / "brats2023"
    )

    plot_modalities_with_mask(
        data_dirs=args.data_dir,
        patient_id=args.patient_id,
        output_dir=output_dir,
        dataset="2023",
    )

    plot_planes_with_mask(
        data_dirs=args.data_dir,
        patient_id=args.patient_id,
        output_dir=output_dir,
        dataset="2023",
    )

    generate_subregion_eda_figures(
        data_dirs=args.data_dir,
        dataset="2023",
        output_dir=output_dir,
    )

    if args.clinical_file is not None:
        clinical_df = (
            read_clinical_file(
                args.clinical_file
            )
        )

        plot_sex_and_age(
            clinical_df=clinical_df,
            dataset_name="BraTS 2023",
            output_dir=output_dir,
            sex_column=args.sex_col,
            age_column=args.age_col,
        )

        grade_column = (
            args.category_col
        )

        if grade_column is None:
            grade_column = (
                find_column(
                    clinical_df,
                    [
                        "Grade",
                        "WHO Grade",
                        "Meningioma Grade",
                        "Tumor Grade",
                    ],
                )
            )

        plot_category_distribution(
            clinical_df=clinical_df,
            dataset_name="BraTS 2023",
            output_dir=output_dir,
            category_column=grade_column,
            title="Meningioma grade",
            category_type="grade",
        )

    print(
        "BraTS 2023 figures saved to: "
        f"{output_dir}"
    )


# ---------------------------------------------------------------------
# BRA TS 2024 EXECUTION
# ---------------------------------------------------------------------


def run_brats2024(
    args: argparse.Namespace,
) -> None:
    """
    Generate BraTS 2024 visualizations.

    Clinical figures are restricted to the same patient cohort
    represented by the provided dataset directories.
    """
    output_dir = (
        args.output_dir
        / "brats2024"
    )

    plot_modalities_with_mask(
        data_dirs=args.data_dir,
        patient_id=args.patient_id,
        output_dir=output_dir,
        dataset="2024",
    )

    plot_planes_with_mask(
        data_dirs=args.data_dir,
        patient_id=args.patient_id,
        output_dir=output_dir,
        dataset="2024",
    )

    generate_subregion_eda_figures(
        data_dirs=args.data_dir,
        dataset="2024",
        output_dir=output_dir,
    )

    if args.clinical_file is not None:
        clinical_df = (
            read_clinical_file(
                args.clinical_file
            )
        )

        # IMPORTANT:
        # Restrict metadata to the actual experimental cohort.
        clinical_df = (
            filter_clinical_to_dataset(
                clinical_df=clinical_df,
                data_dirs=args.data_dir,
            )
        )

        plot_sex_and_age(
            clinical_df=clinical_df,
            dataset_name="BraTS 2024",
            output_dir=output_dir,
            sex_column=args.sex_col,
            age_column=args.age_col,
        )

        glioma_type_column = (
            args.category_col
        )

        if glioma_type_column is None:
            glioma_type_column = (
                find_column(
                    clinical_df,
                    [
                        "Type",
                        "Glioma Type",
                        "Tumor Type",
                        "Diagnosis",
                        "Histology",
                        "Glioma Subtype",
                    ],
                )
            )

        diagnosis_counts = (
            clinical_df[
                glioma_type_column
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
        )

        print()
        print("=" * 70)
        print("GLIOMA TYPE CHECK")
        print("=" * 70)

        print(
            diagnosis_counts.to_string()
        )

        print()
        print(
            "Total cases represented in "
            "glioma-type figure: "
            f"{diagnosis_counts.sum()}"
        )

        print(
            "Clinical cohort rows: "
            f"{len(clinical_df)}"
        )

        missing_diagnosis = (
            clinical_df[
                glioma_type_column
            ]
            .isna()
            .sum()
        )

        print(
            "Patients without glioma-type information: "
            f"{missing_diagnosis}"
        )

        plot_category_distribution(
            clinical_df=clinical_df,
            dataset_name="BraTS 2024",
            output_dir=output_dir,
            category_column=glioma_type_column,
            title="Glioma type",
            category_type="type",
        )

    print(
        "BraTS 2024 figures saved to: "
        f"{output_dir}"
    )


# ---------------------------------------------------------------------
# COMMAND LINE ARGUMENTS
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-ready MRI, clinical, "
            "and subregion visualizations for "
            "BraTS 2023 or BraTS 2024."
        )
    )

    parser.add_argument(
        "dataset",
        choices=[
            "2023",
            "2024",
        ],
        help=(
            "Dataset available on the current device."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more directories containing "
            "patient folders."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where figures will be saved."
        ),
    )

    parser.add_argument(
        "--patient-id",
        required=True,
        help=(
            "Patient identifier used for MRI visualizations."
        ),
    )

    parser.add_argument(
        "--clinical-file",
        type=Path,
        default=None,
        help=(
            "CSV/XLSX file containing sex, age, "
            "and grade or tumor-type metadata."
        ),
    )

    parser.add_argument(
        "--sex-col",
        default=None,
        help=(
            "Sex-column name, if automatic detection fails."
        ),
    )

    parser.add_argument(
        "--age-col",
        default=None,
        help=(
            "Age-column name, if automatic detection fails."
        ),
    )

    parser.add_argument(
        "--category-col",
        default=None,
        help=(
            "Grade-column name for BraTS 2023 or "
            "glioma-type column name for BraTS 2024."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main() -> None:
    """
    Execute the selected dataset workflow.
    """
    configure_publication_style()

    args = parse_arguments()

    if args.dataset == "2023":
        run_brats2023(args)

    else:
        run_brats2024(args)


if __name__ == "__main__":
    main()





# python scripts/generate_brats_figures.py 2023 --data-dir data_diego/ --clinical-fil data_diego/Meningioma\ supplementary\ clinical\ data\ and\ imaging\ parameters\ for\ training\ and\ validation\ sets\ \(1\).xlsx --output-dir figures/eda_article/ --patient-id "BraTS-MEN-00891-000"


# python scripts/generate_brats_figures.py 2024 --data-dir data/training_data1_v2 data/training_data_additional --clinical-file data/BraTS-PTG\ supplementary\ demographic\ information\ and\ metadata.xlsx --output-dir figures/eda_article --patient-id "BraTS-GLI-03063-100"

