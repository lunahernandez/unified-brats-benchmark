# A Unified Benchmark of Convolutional, Transformer, and State Space Models for 3D Brain Tumor Segmentation from Magnetic Resonance Imaging

Repository containing the code developed for the research article:

> **A Unified Benchmark of Convolutional, Transformer, and State Space Models for 3D Brain Tumor Segmentation from Magnetic Resonance Imaging**

The project provides a unified experimental framework for comparing modern deep learning architectures for three-dimensional brain tumor segmentation from magnetic resonance imaging (MRI).

Five representative architectures from different deep learning paradigms are evaluated:

* 3D U-Net, representing conventional convolutional neural networks;
* SegResNet, representing residual convolutional architectures;
* Swin UNETR, representing Transformer-based architectures;
* SegMamba, representing State Space Models;
* SegMambaV2, representing a newer Mamba-based State Space Model architecture.

The models are evaluated under homogeneous experimental conditions, using the same preprocessing, data augmentation, optimization strategy, validation protocol, and inference procedure. The benchmark considers two clinically distinct BraTS datasets: **BraTS 2023**, for intracranial meningioma segmentation, and **BraTS 2024**, for post-treatment glioma segmentation.

The evaluation considers both segmentation performance and computational requirements, including the Dice coefficient, Intersection over Union (IoU), 95th-percentile Hausdorff Distance (HD95), inference time, and GPU memory consumption. As an additional analysis, model complexity was subsequently assessed based on the number of trainable parameters, reported in millions (M).

---

## Project overview

The experimental pipeline includes:

* exploratory analysis of the MRI volumes and segmentation masks;
* loading and preparation of the BraTS datasets;
* standardized preprocessing of multimodal MRI volumes;
* spatial resampling and foreground cropping;
* intensity normalization;
* data augmentation during training;
* five-fold cross-validation;
* training of five 3D segmentation architectures;
* validation and model checkpointing;
* sliding-window inference;
* calculation of segmentation and spatial metrics;
* measurement of computational cost;
* aggregation and statistical analysis of cross-validation results;
* qualitative and quantitative visualization of the predictions.

The main objective is to provide a reproducible benchmark in which differences between architectures can be analyzed under controlled experimental conditions.

---

## Datasets

The datasets are not included in this repository due to their size and licensing restrictions.

The experiments use two publicly available datasets from the **Brain Tumor Segmentation (BraTS) Challenge**:

* **BraTS 2023**: intracranial meningioma segmentation;
* **BraTS 2024**: post-treatment glioma segmentation.

The datasets can be obtained from their respective official Synapse repositories:

* **BraTS 2023**: [Synapse - BraTS 2023](https://www.synapse.org/Synapse:syn51156910/wiki/627000)
* **BraTS 2024**: [Synapse - BraTS 2024](https://www.synapse.org/Synapse:syn53708249/wiki/627759)

After downloading the datasets, they should be placed inside the `data/` directory.

A structure similar to the following is expected:

```text
data/
├── brats_train_val_2023/
│   ├── BraTS-MEN-Train/
│   └── BraTS-MEN-Validation/
├── training_data1_v2/
├── training_data_additional/
├── validation_data/
├── BraTS-PTG supplementary demographic information and metadata.xlsx
├── CITATIONS.bib
└── manifest.csv
```


The structure shown above is only an example. The dataset locations and paths are not required to follow this exact structure. All dataset paths, as well as the experiment configuration, can be modified at any time in:

```text
src/config.py
```

Before running the project, make sure that the paths defined in `src/config.py` match the location and structure of the datasets on the local machine.

---

## Experimental setup

The benchmark uses the same general experimental protocol for all architectures to enable a fair comparison.

All models receive four MRI modalities as input and perform voxel-wise multi-class segmentation.

The datasets are divided using a five-fold cross-validation strategy. For each fold, the data are divided into training, validation, and test subsets.

The preprocessing pipeline includes:

1. Loading the four MRI modalities and the corresponding segmentation mask.
2. Reorienting volumes to the RAS anatomical convention.
3. Resampling to an isotropic voxel spacing of `1.0 × 1.0 × 1.0 mm³`.
4. Cropping the foreground region.
5. Padding volumes to a minimum spatial size of `128 × 128 × 128` voxels.
6. Normalizing MRI intensities independently for each modality.
7. Applying random spatial crops, flips, intensity scaling, and intensity shifting during training.
8. Using deterministic preprocessing during validation and testing.
9. Applying sliding-window inference during evaluation.

The same preprocessing and augmentation strategy is used across architectures to minimize the influence of external experimental factors.

---

## Model architectures

The repository contains five three-dimensional segmentation architectures:

```text
src/models/
├── get_model.py
├── segresnet.py
├── segmamba.py
├── segmamba_v2.py
├── swin_unetr.py
└── unet3d.py
```

### 3D U-Net

A three-dimensional extension of the original U-Net architecture based on a symmetric encoder-decoder structure with skip connections.

### SegResNet

A residual convolutional architecture based on an asymmetric encoder-decoder structure with residual blocks.

### Swin UNETR

A Transformer-based architecture combining a hierarchical Swin Transformer encoder with a convolutional decoder.

### SegMamba

A State Space Model architecture based on Mamba blocks designed to capture long-range dependencies with a lower computational complexity than conventional self-attention mechanisms.

### SegMambaV2

An updated Mamba-based architecture combining convolutional and State Space components with hierarchical downsampling and orthogonal-oriented Mamba blocks.

---

## Evaluation metrics

The project evaluates segmentation performance using the following metrics:

### Dice coefficient

Measures the overlap between the predicted segmentation and the reference annotation.

Higher values indicate better segmentation agreement.

### Intersection over Union

Measures the intersection between prediction and ground truth relative to their union.

Higher values indicate better segmentation performance.

### 95th-percentile Hausdorff Distance

HD95 evaluates the spatial agreement between the predicted and reference segmentation boundaries.

The metric is reported in millimeters, and lower values indicate better boundary agreement.

The metrics are evaluated for the relevant tumor sub-regions and aggregated according to the experimental protocol.

---

## Computational evaluation

In addition to segmentation accuracy, the benchmark evaluates computational requirements.

The following indicators are considered:

- inference time per test volume;
- peak GPU memory consumption.

As an additional analysis, the complexity of each architecture was assessed based on the number of trainable parameters, reported in millions (M).

These measurements are used to analyze the trade-off between segmentation performance and computational efficiency.

This allows the architectures to be compared not only according to their predictive performance but also according to their practical computational requirements.

---

## Statistical analysis

The repository includes scripts for aggregating cross-validation results and performing statistical analyses.

The statistical analysis is used to determine whether differences observed between architectures are meaningful rather than simply caused by variability between experimental folds.

Cross-validation results can be aggregated and analyzed using the scripts available in:

```text
scripts/
```

---

## Repository structure

```text
unified-brats-benchmark/
├── src/                         # Main project source code
│   ├── config.py                # Global configuration and experiment settings
│   ├── main.py                  # Main experimental pipeline
│   ├── train.py                 # Model training
│   ├── validate.py              # Validation during training
│   ├── evaluate.py              # Evaluation of trained models
│   ├── data/                    # Dataset loading and preprocessing
│   │   ├── dataset.py
│   │   ├── splits.py
│   │   └── transforms.py
│   ├── eda/                     # Exploratory data analysis
│   │   ├── main_eda.py
│   │   ├── metadata_eda.py
│   │   └── utils.py
│   ├── models/                  # 3D segmentation architectures
│   │   ├── get_model.py
│   │   ├── segresnet.py
│   │   ├── segmamba.py
│   │   ├── segmamba_v2.py
│   │   ├── swin_unetr.py
│   │   └── unet3d.py
│   └── utils/                   # Shared project utilities
│       ├── brats_lesionwise.py
│       ├── brats_regions.py
│       ├── checkpoints.py
│       ├── meters.py
│       └── seed.py
├── scripts/                     # Result analysis and visualization scripts
│   ├── analyze_crossval_statistics.py
│   ├── compare_background_results.py
│   ├── create_3d_models_region_grid.py
│   ├── generate_crossval_results.py
│   ├── plot_crossval_results.py
│   └── utils_results.py
├── docs/                        # Additional documentation
│   └── segmamba_notes.md        # SegMamba installation and compatibility notes
├── requirements.txt             # Python dependencies
├── LICENSE                      # Project license
├── NOTICE                       # Third-party code attributions
└── README.md                    # Main project documentation
```

---

## Execution environment

The experiments were developed and executed using the following environment:

```text
OS: Windows 11 Home 25H2
Environment: WSL Ubuntu
Python: 3.12
GPU: NVIDIA GeForce RTX 5090
GPU memory: 24 GB
CPU: Intel Core Ultra 9 275HX
RAM: 64 GB
Storage: 1 TB NVMe SSD
CUDA Toolkit: 12.8
PyTorch: 2.11.0+cu128
```

The project uses **PyTorch** as the main deep learning framework and **MONAI** for medical image processing, data loading, transformations, model components, loss functions, metrics, and sliding-window inference.

Additional libraries include:

* NumPy;
* SciPy;
* Pandas;
* NiBabel;
* Matplotlib;
* MONAI;
* `mamba_ssm`;
* `causal_conv1d`;
* `einops`;
* `transformers`.

---

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Upgrade the basic packaging and compilation tools:

```bash
python -m pip install --upgrade pip wheel ninja
python -m pip install "setuptools==70.2.0"
```

Install the general project dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file uses the PyTorch build with CUDA 12.8 support.

> **Note:** Running SegMamba and SegMambaV2 requires additional dependencies and local compilation steps. These are described in `docs/segmamba_notes.md`.

In particular, `causal-conv1d` and `mamba-ssm` were compiled locally to ensure compatibility with the NVIDIA RTX 5090 and the selected CUDA/PyTorch environment.

---

## Configuration

The main configuration file is:

```text
src/config.py
```

It contains the project paths, dataset configuration, training parameters, model configuration, cross-validation settings, and evaluation options.

Before running an experiment, check the following settings:

* dataset paths;
* output directories;
* selected architecture;
* training hyperparameters;
* cross-validation configuration;
* evaluation configuration.

---

## Usage

Run the complete experimental pipeline from the project root:

```bash
python -m src.main
```

By default, the execution mode is `all`, which performs both training and evaluation.

### Train and evaluate a specific fold

```bash
python -m src.main --fold 1
```

### Train only

```bash
python -m src.main --mode train
```

### Train a specific fold

```bash
python -m src.main --mode train --fold 1
```

### Evaluate trained models

```bash
python -m src.main --mode test
```

### Evaluate a specific fold

```bash
python -m src.main --mode test --fold 1
```

### Include or exclude the background class

Experiments can be executed with or without including the background class in the loss:

```bash
python -m src.main --include-background
python -m src.main --no-include-background
```

These options can also be combined with other execution parameters:

```bash
python -m src.main --mode train --fold 1 --include-background
python -m src.main --mode test --fold 1 --no-include-background
```

---

## Result analysis scripts

The `scripts/` directory contains utilities for aggregating, analyzing, and visualizing experimental results.

Examples:

```bash
python scripts/generate_crossval_results.py
python scripts/analyze_crossval_statistics.py
python scripts/compare_background_results.py
python scripts/plot_crossval_results.py
python scripts/create_3d_models_region_grid.py
```

These scripts can be used to process the results generated during cross-validation and to produce quantitative and qualitative analyses.

---

## Third-party code and attribution

This project includes or adapts code from external open-source projects, including components related to SegMamba and BraTS evaluation utilities.

See the `NOTICE` file for detailed third-party attributions and licensing information.

---

## License

This project is licensed under the Apache License 2.0.

See the `LICENSE` file for more information.
