from torch.nn import Module

from src.models.unet3d import build_unet3d
from src.models.resunet3d import build_resunet3d
from src.models.segresnet import build_segresnet
from src.models.swin_unetr import build_swin_unetr
from src.models.segmamba import SegMamba
from src.models.segmambav2 import build_segmambav2


def get_model(
    model_name: str,
    in_channels: int,
    out_channels: int,
    use_checkpoint: bool = True,
) -> Module:
    """Build a segmentation model.

    Args:
        model_name: Name of the model to build.
        in_channels: Number of input MRI modalities.
        out_channels: Number of output classes, including background.
        use_checkpoint: Whether to enable gradient checkpointing for
            compatible models.

    Returns:
        An initialized segmentation model.

    Raises:
        ValueError: If `model_name` is not supported.
    """
    model_name = model_name.lower()

    if model_name == "unet3d":
        return build_unet3d(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    if model_name == "resunet3d":
        return build_resunet3d(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    if model_name == "segresnet":
        return build_segresnet(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    if model_name == "swin_unetr":
        return build_swin_unetr(
            in_channels=in_channels,
            out_channels=out_channels,
            use_checkpoint=use_checkpoint,
        )

    if model_name == "segmamba":
        return SegMamba(
            in_chans=in_channels,
            out_chans=out_channels,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
        )

    if model_name == "segmambav2":
        return build_segmambav2(
            in_channels=in_channels,
            out_channels=out_channels,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0.3
        )

    supported_models = [
        "unet3d",
        "resunet3d",
        "segresnet",
        "swin_unetr",
        "segmamba",
        "segmambav2",
    ]

    raise ValueError(
        f"Unsupported model: {model_name}. "
        f"Supported models: {', '.join(supported_models)}."
    )


