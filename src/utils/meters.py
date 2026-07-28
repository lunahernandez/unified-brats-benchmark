class AverageMeter:
    """Computes the average of a metric.

    Note:
        Implementation adapted from an official MONAI tutorial:
        https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb

    Attributes:
        val: Last recorded value.
        avg: Accumulated average.
        sum: Accumulated sum of the values.
        count: Total number of accumulated elements.
    """
    def __init__(self) -> None:
        """Initializes the class and resets its attributes."""
        self.reset()

    def reset(self) -> None:
        """Resets the attribute values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Updates the values from a new record.

        Args:
            val: New metric value.
            n: Number of elements associated with that value.
        """
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0