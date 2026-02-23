import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any


def get_train_transforms(config: dict, policy: str = None) -> A.Compose:
    aug_config = config.get("augmentation", {})
    if policy is None:
        policy = aug_config.get("policy", "medium")

    p_cfg = aug_config.get(policy, {})
    ds_cfg = config.get("dataset", {})
    prep_cfg = config.get("preprocessing", {})

    input_size = prep_cfg.get("resize", 224)
    mean = ds_cfg.get("mean", [0.485, 0.456, 0.406])
    std = ds_cfg.get("std", [0.229, 0.224, 0.225])

    if policy == "light":
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=p_cfg.get("horizontal_flip_p", 0.5)),
            A.Rotate(limit=p_cfg.get("rotation_limit", 15), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=p_cfg.get("brightness_limit", 0.1),
                contrast_limit=p_cfg.get("contrast_limit", 0.1),
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    elif policy == "medium":
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=p_cfg.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=p_cfg.get("vertical_flip_p", 0.1)),
            A.Rotate(limit=p_cfg.get("rotation_limit", 30), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=p_cfg.get("brightness_limit", 0.2),
                contrast_limit=p_cfg.get("contrast_limit", 0.2),
                p=0.4,
            ),
            A.GaussianBlur(blur_limit=p_cfg.get("blur_limit", 3), p=0.2),
            A.GaussNoise(var_limit=p_cfg.get("noise_var_limit", (10.0, 30.0)), p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=p_cfg.get("hue_shift_limit", 10),
                sat_shift_limit=p_cfg.get("sat_shift_limit", 20),
                val_shift_limit=10,
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    elif policy == "strong":
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=p_cfg.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=p_cfg.get("vertical_flip_p", 0.2)),
            A.ShiftScaleRotate(
                shift_limit=p_cfg.get("shift_limit", 0.1),
                scale_limit=p_cfg.get("scale_limit", 0.2),
                rotate_limit=p_cfg.get("rotation_limit", 45),
                border_mode=0,
                p=0.5,
            ),
            A.ElasticTransform(
                alpha=p_cfg.get("elastic_alpha", 120),
                sigma=p_cfg.get("elastic_sigma", 6),
                p=0.2,
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=p_cfg.get("brightness_limit", 0.3),
                    contrast_limit=p_cfg.get("contrast_limit", 0.3),
                ),
                A.CLAHE(clip_limit=4.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ], p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=p_cfg.get("hue_shift_limit", 15),
                sat_shift_limit=p_cfg.get("sat_shift_limit", 30),
                val_shift_limit=15,
                p=0.4,
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=p_cfg.get("blur_limit", 5)),
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
            ], p=0.2),
            A.GaussNoise(var_limit=p_cfg.get("noise_var_limit", (10.0, 50.0)), p=0.2),
            A.CoarseDropout(
                max_holes=p_cfg.get("coarse_dropout_max_holes", 8),
                max_height=p_cfg.get("coarse_dropout_max_height", 24),
                max_width=p_cfg.get("coarse_dropout_max_width", 24),
                min_holes=1,
                fill_value=0,
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    else:
        raise ValueError(f"Unknown policy: {policy}. Use: light, medium, strong")


def get_valid_transforms(config: dict) -> A.Compose:
    """
    Ne radimo augmentaciju za valid skup, samo resize i normalize
    """
    ds_cfg = config.get("dataset", {})
    prep_cfg = config.get("preprocessing", {})

    input_size = prep_cfg.get("resize", 224)
    mean = ds_cfg.get("mean", [0.485, 0.456, 0.406])
    std = ds_cfg.get("std", [0.229, 0.224, 0.225])

    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])