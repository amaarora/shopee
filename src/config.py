import albumentations
from pathlib import Path

IMG_SIZE = 512

Config = dict(
    DATA_DIR="../data/",
    TRAIN_DATA_DIR="../data/train_images",
    TEST_DATA_DIR="../data/test_images",
    TRAIN_CSV_PATH="../data/train.csv",
    DEVICE="cuda",
    MODEL="tf_efficientnet_b5",
    EPOCHS=10,
    SCHEDULER_PARAMS={
        "lr_start": 1e-5,
        "lr_max": 1e-5,
        "lr_min": 1e-6,
        "lr_ramp_ep": 2,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    },
    BS=8,
    TRAIN_AUG=albumentations.Compose(
        [
            albumentations.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            albumentations.Normalize(),
        ]
    ),
    TEST_AUG=albumentations.Compose(
        [
            albumentations.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
            albumentations.Normalize(),
        ]
    ),
)
