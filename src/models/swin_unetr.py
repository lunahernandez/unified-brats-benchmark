from typing import Tuple

from monai.networks.nets import SwinUNETR


def build_swin_unetr(
    in_channels: int = 4,
    out_channels: int = 5,
    feature_size: int = 48,
    use_checkpoint: bool = True,
) -> SwinUNETR:
    """Builds a SwinUNETR model for volumetric segmentation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        feature_size: Dimension of the embedding features.
        use_checkpoint: Indicates whether to use gradient checkpointing to save memory.

    Returns:
        An instance of the MONAI SwinUNETR model for 3D segmentation.
    """
    return SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        spatial_dims=3,
    )