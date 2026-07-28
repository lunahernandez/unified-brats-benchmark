import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
from monai.losses import DiceCELoss

from src.utils.meters import AverageMeter
from src.validate import validate
from src.utils.checkpoints import save_checkpoint


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: Optimizer,
    max_epochs: int,
    val_every: int,
    experiment_dir: Path,
    roi_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 1,
    clip_grad: bool = False,
    grad_clip_max_norm: float = 1.0,
    include_background: bool = True,
) -> dict[str, Any]:
    """Trains a segmentation model with mixed precision and periodic validation.

    This function handles the training loop, applying Automatic Mixed Precision 
    (AMP) if a CUDA device is used. It evaluates the model on the validation 
    set every `val_every` epochs, tracks the mean Dice and HD95 metrics, and 
    saves the best model checkpoint based on the mean Dice score.

    Args:
        model: The PyTorch neural network module to train.
        train_loader: DataLoader providing the training set batches.
        val_loader: DataLoader providing the validation set batches.
        device: The computation device (e.g., 'cuda' or 'cpu').
        optimizer: The PyTorch optimizer used to update the weights.
        max_epochs: Total number of training epochs to run.
        val_every: Frequency (in epochs) at which validation is performed.
        experiment_dir: Directory where the best model checkpoint will be saved.
        roi_size: Spatial dimensions of the region of interest for sliding 
            window inference during validation.
        sw_batch_size: Number of sliding windows to process in a single batch 
            during validation inference.
        clip_grad: Whether to apply gradient clipping to prevent exploding gradients.
        grad_clip_max_norm: Maximum allowed norm for the gradients if clipping 
            is enabled.
        include_background: Whether the background class is included in the loss computation.

    Returns:
        A dictionary containing the overall training summary:
            - "best_val_dice" (float): The highest mean Dice score achieved.
            - "best_epoch" (int): The epoch at which the best score was achieved.
            - "total_train_time_sec" (float): Total training time in seconds.
            - "history" (list[dict]): List of dictionaries containing the metric 
              history per epoch.
    """
    loss_function = DiceCELoss(include_background=include_background, to_onehot_y=True, softmax=True)
    best_metric = -1.0
    best_metric_epoch = -1
    history = []

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_start = time.perf_counter()

    for epoch in range(max_epochs):
        print("-" * 50)
        print(f"epoch {epoch + 1}/{max_epochs}")

        epoch_start = time.perf_counter()

        model.train()
        epoch_loss_meter = AverageMeter()

        progress_bar = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{max_epochs}",
            leave=True,
        )

        for batch_data in progress_bar:
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()

            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_max_norm,
                )

            scaler.step(optimizer)
            scaler.update()

            epoch_loss_meter.update(loss.item(), n=inputs.shape[0])

            progress_bar.set_postfix({
                "loss": f"{epoch_loss_meter.avg:.4f}"
            })

        epoch_time = time.perf_counter() - epoch_start

        print(f"epoch {epoch + 1} loss: {epoch_loss_meter.avg:.4f}")
        print(f"epoch {epoch + 1} time: {epoch_time:.2f} s")

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": float(epoch_loss_meter.avg),
            "epoch_time_sec": float(epoch_time),
        }

        if (epoch + 1) % val_every == 0:
            metrics = validate(
                model=model,
                data_loader=val_loader,
                device=device,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
            )

            val_mean_dice = metrics["mean_dice"]
            val_mean_hd95 = metrics["mean_hd95"]

            print(f"val mean Dice: {val_mean_dice:.4f}" if val_mean_dice is not None else "val mean Dice: None")
            print(f"val mean HD95: {val_mean_hd95:.4f}" if val_mean_hd95 is not None else "val mean HD95: None")

            epoch_info["val_mean_dice"] = val_mean_dice
            epoch_info["val_mean_hd95"] = val_mean_hd95
            epoch_info["val_regions"] = metrics["regions"]

            current_metric = val_mean_dice if val_mean_dice is not None else -1.0

            if current_metric > best_metric:
                best_metric = current_metric
                best_metric_epoch = epoch + 1

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    best_metric=best_metric,
                    save_path=experiment_dir / "best_model.pt",
                )

                print("new best model saved")

        history.append(epoch_info)

    total_time = time.perf_counter() - total_start

    print(f"Best validation mean Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    print(f"Total training time: {total_time:.2f} s")

    return {
        "best_val_dice": float(best_metric),
        "best_epoch": int(best_metric_epoch),
        "total_train_time_sec": float(total_time),
        "history": history,
    }
