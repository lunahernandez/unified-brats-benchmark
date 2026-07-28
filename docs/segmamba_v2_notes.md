# SegMamba-V2 Installation Notes

This document describes the additional steps required to run the SegMamba-V2 experiments included in this project.

The general environment setup (CUDA Toolkit installation, Python virtual environment, and project dependencies) is identical to that described in the SegMamba installation guide. Please complete those steps first.

Unlike the original SegMamba implementation, SegMamba-V2 does **not** require locally compiling `causal-conv1d` or `mamba-ssm`, nor modifying their source code to support `sm_120`.

## Install SegMamba-V2 Dependencies

After installing the project dependencies, install the packages required by SegMamba-V2:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

pip install ninja packaging

pip install "causal-conv1d>=1.2.0" --no-build-isolation

pip install mamba-ssm

pip uninstall cuda-python cuda-bindings -y
pip install --upgrade cuda-bindings==12.9.7 cuda-python==12.9.7

pip install -U transformers
```

These commands:

- install the CUDA 12.8 build of PyTorch;
- install the build utilities required by `mamba-ssm`;
- install `causal-conv1d`;
- install `mamba-ssm`;
- replace the existing CUDA Python bindings with the versions that were compatible with our environment;
- upgrade the `transformers` package.

## Final Verification

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
python -c "from mamba_ssm import Mamba; print('mamba-ssm OK')"
python -c "import causal_conv1d; print('causal-conv1d OK')"
nvcc -V
```

These commands verify that:

- PyTorch is correctly installed;
- CUDA is available;
- `mamba-ssm` can be imported successfully;
- `causal-conv1d` is correctly installed;
- the CUDA Toolkit is correctly configured.

Once this setup is complete, the training and evaluation commands are described in the main `README.md`.