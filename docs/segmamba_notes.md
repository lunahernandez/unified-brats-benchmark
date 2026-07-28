# SegMamba Installation Notes

This document describes the additional configuration required to run the SegMamba-based experiments included in this project.

SegMamba was installed locally from the original repository. Due to compatibility issues with the NVIDIA RTX 5090, the `causal-conv1d` and `mamba-ssm` dependencies had to be compiled manually with support for the `sm_120` compute capability.

## Environment

- Operating system: WSL Ubuntu
- GPU: NVIDIA RTX 5090
- CUDA Toolkit: 12.8
- PyTorch: 2.11.0+cu128
- MONAI: 1.5.2
- transformers: 4.36.2

## Important Note

`causal-conv1d` and `mamba-ssm` are **not** included in `requirements.txt`, as they must be installed locally from the SegMamba repository after applying the compatibility modifications described in this document.

In the commands below, replace `/path/to/unified-brats-benchmark` with the local path where this repository has been cloned.

```bash
export PROJECT_ROOT=/path/to/unified-brats-benchmark
```

## 1. Install CUDA Toolkit 12.8 on WSL

SegMamba requires local compilation of CUDA-dependent packages. For this reason, CUDA Toolkit 12.8 was used under WSL.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo mkdir -p /usr/share/keyrings
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu-keyring.gpg
sudo mv cuda-wsl-ubuntu-keyring.gpg /usr/share/keyrings/

echo "deb [signed-by=/usr/share/keyrings/cuda-wsl-ubuntu-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-wsl-ubuntu-x86_64.list

sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

These commands:

- add the NVIDIA CUDA package repository for WSL;
- register the repository signing key;
- update the local package index;
- install CUDA Toolkit 12.8, including `nvcc`, CUDA libraries, and header files.

## 2. Configure CUDA 12.8 as the Active Version

```bash
sudo ln -sfn /usr/local/cuda-12.8 /usr/local/cuda

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc
hash -r
nvcc -V
```

These commands:

- create `/usr/local/cuda` as a symbolic link to CUDA 12.8;
- define `CUDA_HOME`, used by build tools to locate CUDA;
- add CUDA executables, such as `nvcc`, to the system `PATH`;
- add CUDA libraries to `LD_LIBRARY_PATH`;
- reload the terminal configuration;
- verify that `nvcc` is available and points to the expected CUDA version.

## 3. Create the Python Virtual Environment

From the project root:

```bash
cd "$PROJECT_ROOT"

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip wheel ninja
python -m pip install "setuptools==70.2.0"
```

These commands:

- create a clean virtual environment;
- activate the virtual environment;
- upgrade the basic packaging tools;
- install `ninja`, which is used during local compilation;
- install the version of `setuptools` used in the working environment.

## 4. Install the Project Dependencies

The project's general dependencies are installed from the main requirements file:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes the CUDA 12.8 build of PyTorch:

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
```

After installation, PyTorch can be verified with:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())"
nvcc -V
```

These commands verify:

- the installed PyTorch version;
- the CUDA version used by PyTorch;
- the GPU architectures supported by the installed PyTorch build;
- the CUDA Toolkit version available through `nvcc`.

## 5. Clone SegMamba Locally

From the project root:

```bash
cd "$PROJECT_ROOT"
git clone https://github.com/ge-xing/SegMamba.git
```

The external SegMamba repository is cloned locally but is **not** included in this repository.

## 6. Adapt the SegMamba Implementation

The SegMamba implementation used in this project was adapted from the following file in the public repository:

```text
SegMamba/model_segmamba/segmamba.py
```

This file was copied and adapted as:

```text
src/models/segmamba.py
```

This adaptation allows SegMamba to be integrated into the same experimental pipeline as the other evaluated models.

### Update the Mamba Import

In `src/models/segmamba.py`, the original import:

```python
from mamba_ssm import Mamba
```

was replaced with:

```python
from mamba_ssm.modules.mamba_simple import Mamba
```

This change was required because the local installation of `mamba-ssm` exposes the `Mamba` module through `mamba_ssm.modules.mamba_simple`.

### Add Support for `sm_120`

The following files in the local SegMamba repository also had to be modified:

```text
SegMamba/causal-conv1d/setup.py
SegMamba/mamba/setup.py
```

The following block was added to both files to enable support for the `sm_120` compute capability:

```python
if bare_metal_version >= Version("12.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_120,code=sm_120")
```

This modification allows the CUDA extensions to be compiled for the NVIDIA RTX 5090.

## 7. Build `causal-conv1d` Locally

Before compiling, ensure that CUDA 12.8 is available in the current terminal:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Then build and install `causal-conv1d` from the local SegMamba repository:

```bash
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

cd "$PROJECT_ROOT/SegMamba/causal-conv1d"
pip install --no-build-isolation --no-cache-dir --no-deps --force-reinstall .
```

These commands:

- force a local build instead of using a precompiled wheel;
- compile the CUDA extension using the local CUDA Toolkit;
- install the package into the active virtual environment.

## 8. Build `mamba-ssm` Locally

```bash
export MAMBA_FORCE_BUILD=TRUE

cd "$PROJECT_ROOT/SegMamba/mamba"
pip install --no-build-isolation --no-cache-dir --no-deps --force-reinstall .
```

These commands:

- force a local build of `mamba-ssm`;
- compile the package using the modified `setup.py` with `sm_120` support;
- install the package into the active virtual environment.

## 9. Final Verification

From the project root:

```bash
cd "$PROJECT_ROOT"

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())"
python -c "import nibabel; print(nibabel.__version__)"
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('mamba-ssm OK')"
nvcc -V
```

These commands verify that:

- PyTorch is correctly installed;
- CUDA is available;
- the CUDA Toolkit points to the expected version;
- medical imaging dependencies such as `nibabel` are installed;
- the local `mamba-ssm` installation can be imported successfully.

Once this setup is complete, the training and evaluation commands are described in the main `README.md`.