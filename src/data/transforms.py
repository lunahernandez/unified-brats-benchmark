from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
)


def get_train_transforms(
    roi_size: tuple[int, int, int] = (128, 128, 128),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Builds the transformations for the training set.

    Includes data loading, format preparation, reorientation,
    resampling, normalization, cropping, spatial padding, and
    data augmentation.

    Args:
        roi_size: Size of the region of interest.
        spacing: Spacing applied to resampling.

    Returns:
        A composition of transformations for the training set.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=roi_size,
            random_size=False,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ])


def get_val_transforms(
    roi_size: tuple[int, int, int] = (128, 128, 128),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Builds the transformations for the validation set.

    Includes data loading, format preparation, reorientation,
    resampling, normalization, cropping, and spatial padding.

    Args:
        roi_size: Size of the region of interest.
        spacing: Spacing applied to resampling.

    Returns:
        A composition of transformations for the validation set.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ])


def get_test_transforms(
    roi_size: tuple[int, int, int] = (128, 128, 128),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Builds the transformations for the testing set.

    Includes data loading, format preparation, reorientation,
    resampling, normalization, cropping, and spatial padding.

    Args:
        roi_size: Size of the region of interest.
        spacing: Spacing applied to resampling.

    Returns:
        A composition of transformations for the testing set.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ])
