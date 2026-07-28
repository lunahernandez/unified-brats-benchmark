import torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference

from src.config import DATASET
from src.utils.brats_regions import (
    initialize_region_metrics,
    update_region_metrics,
    summarize_region_metrics,
)


def validate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    roi_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 1,
) -> dict:
    """Evaluate the model on the validation set.

    The model generates predictions using sliding-window inference and
    calculates metrics for each BraTS region according to the selected dataset.

    Args:
        model: Segmentation model to evaluate.
        data_loader: Data loader for the validation set.
        device: Device on which the evaluation is performed.
        roi_size: Size of the region of interest.
        sw_batch_size: Number of windows processed simultaneously in
            sliding-window inference.

    Returns:
        Dictionary with the validation metrics.
    """
    model.eval()

    use_amp = device.type == "cuda"

    region_stats = initialize_region_metrics(
        dataset=DATASET,
    )

    progress_bar = tqdm(
        data_loader,
        desc="Validation",
        leave=False,
    )

    with torch.no_grad():
        for batch_data in progress_bar:
            images = batch_data["image"].to(
                device,
                non_blocking=True,
            )

            labels = batch_data["label"].to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                )

            pred_labels = torch.argmax(outputs, dim=1)

            update_region_metrics(
                stats=region_stats,
                pred_labels=pred_labels,
                gt_labels=labels,
                dataset=DATASET,
            )

    return summarize_region_metrics(
        stats=region_stats,
        dataset=DATASET,
    )


