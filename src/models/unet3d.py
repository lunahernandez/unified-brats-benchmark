from typing import Tuple

from monai.networks.nets import UNet


def build_unet3d(
    in_channels: int = 4,
    out_channels: int = 5,
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
) -> UNet:
    """Builds a 3D U-Net model for volumetric segmentation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        channels: Sequence with the number of feature channels at each level.
        strides: Sequence with the spatial reduction factors between levels.

    Returns:
        An instance of the MONAI U-Net model for 3D segmentation.
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
    )