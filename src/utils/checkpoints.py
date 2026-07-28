from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer


def save_checkpoint(
    model: Module,
    optimizer: Optimizer | None,
    epoch: int,
    best_metric: float,
    save_path: str | Path,
) -> None:
    """Saves a checkpoint of the model and, optionally, of the optimizer.

    Note:
        Implementation adapted from an official MONAI tutorial:
        https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb

    Args:
        model: Model to save.
        optimizer: Training optimizer.
        epoch: Epoch number corresponding to the checkpoint.
        best_metric: Best registered metric value.
        save_path: Path where the checkpoint will be saved.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at: {save_path}")


def load_checkpoint(
    model: Module,
    optimizer: Optimizer | None,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[Module, Optimizer | None, int, float]:
    """Loads a checkpoint into the model and, optionally, into the optimizer.

    Args:
        model: Model where the saved state will be loaded.
        optimizer: Optimizer where the saved state will be loaded.
        checkpoint_path: Path to the checkpoint file.
        device: Device where the checkpoint will be loaded.

    Returns:
        A tuple containing the model, the optimizer, the checkpoint epoch, and
        the best registered metric value.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0.0)
    return model, optimizer, epoch, best_metric