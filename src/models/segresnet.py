from monai.networks.nets import SegResNet


def build_segresnet(
    in_channels: int,
    out_channels: int,
) -> SegResNet:
    """Build and return a SegResNet model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels/classes.

    Returns:
        An initialized SegResNet model.
    """
    return SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=0.2,
    )


