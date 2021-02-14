import albumentations as A
import cv2
import numpy as np
import streamlit as st

from PIL import Image


def select_augmentation():
    st.sidebar.subheader("Augmentations")
    selected = st.sidebar.multiselect("select augmentations", options=[
        "CircleCrop", "Resize",
        "ShiftScaleRotate", "CenterCrop", "RandomCrop", "RandomResizedCrop",
        "RandomBrightness", "RandomBrightnessContrast",
        "CoarseDropout", "Cutout", "Blur", "CLAHE",
        "ColorJitter", "Downscale",
        "Emboss", "Equalize", "FancyPCA", "Flip", "GaussianBlur",
        "GaussNoise", "GlassBlur", "GridDistortion", "GridDropout",
        "HorizontalFlip", "HueSaturationValue", "MaskDropout",
        "MedianBlur", "MotionBlur", "MultiplicativeNoise",
        "OpticalDistortion", "Posterize", "RandomContrast",
        "RandomGamma", "VerticalFlip"
    ])
    return selected


def collect_augmentation(augmentations: list):
    compose = []
    if "ShiftScaleRotate" in augmentations:
        st.sidebar.markdown("ShiftScaleRotate")
        shift_limit = st.sidebar.slider(
            "shift_limit",
            min_value=0.0,
            max_value=1.0,
            value=0.0625,
            step=0.0001,
            key="shift_scale_rotate_shift_limit")
        scale_limit = st.sidebar.slider(
            "scale_limit",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            key="shift_scale_rotate_scale_limit")
        rotate_limit = st.sidebar.slider(
            "rotate_limit",
            min_value=0,
            max_value=180,
            value=45,
            step=1,
            key="shift_scale_rotate_rotate_limit")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="shift_scale_rotate_p")
        compose.append(A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=p))
    if "CenterCrop" in augmentations:
        st.sidebar.markdown("CenterCrop")
        height = st.sidebar.number_input(
            "height",
            min_value=1,
            max_value=8192,
            value=512,
            step=1,
            key="center_crop_height")
        width = st.sidebar.number_input(
            "width",
            min_value=1,
            max_value=8192,
            value=512,
            step=1,
            key="center_crop_width")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="center_crop_p")
        compose.append(A.CenterCrop(
            height=height,
            width=width,
            p=p))
    if "RandomCrop" in augmentations:
        st.sidebar.markdown("RandomCrop")
        height = st.sidebar.number_input(
            "height",
            min_value=1,
            max_value=8192,
            value=512,
            step=1,
            key="random_crop_height")
        width = st.sidebar.number_input(
            "width",
            min_value=1,
            max_value=8192,
            value=512,
            step=1,
            key="random_crop_width")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="random_crop_p")
        compose.append(A.RandomCrop(
            height=height,
            width=width,
            p=p))
    if "RandomResizedCrop" in augmentations:
        st.sidebar.markdown("RandomResizedCrop")
        height = st.sidebar.number_input(
            "height",
            min_value=1,
            max_value=8192,
            value=512,
            step=1,
            key="random_resized_crop_height")
        width = st.sidebar.number_input(
            "width",
            min_value=1,
            max_value=8192,
            value=512,
            step=1,
            key="random_resized_crop_width")
        scale_max = st.sidebar.slider(
            "scale_max",
            min_value=0.8,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key="random_resized_crop_scale_max")
        scale_min = st.sidebar.slider(
            "scale_min",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.01,
            key="random_resized_crop_scale_min")
        ratio_max = st.sidebar.slider(
            "ratio_max",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.01,
            key="random_resized_crop_ratio_max")
        ratio_min = st.sidebar.slider(
            "ratio_min",
            min_value=1.0,
            max_value=1.6,
            value=1.1,
            step=0.01,
            key="random_resized_crop_ratio_min")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="random_resized_crop_p")
        compose.append(A.RandomResizedCrop(
            height=height,
            width=width,
            scale=(scale_min, scale_max),
            ratio=(ratio_min, ratio_max),
            p=p))
    if "Resize" in augmentations:
        st.sidebar.markdown("Resize")
        height = st.sidebar.number_input(
            "height",
            min_value=1,
            max_value=8192,
            value=512,
            step=1,
            key="resize_height")
        width = st.sidebar.number_input(
            "width",
            min_value=1,
            max_value=8192,
            value=512,
            step=1,
            key="resize_width")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="resize_p")
        compose.append(A.Resize(
            height=height,
            width=width,
            p=p
        ))
    if "RandomBrightness" in augmentations:
        st.sidebar.markdown("RandomBrightness")
        limit = st.sidebar.slider(
            "limit",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            key="random_brightness_limit")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="random_brightness_p")
        compose.append(A.RandomBrightness(
            limit=limit,
            p=p))
    if "RandomBrightnessContrast" in augmentations:
        st.sidebar.markdown("RandomBrightnessContrast")
        brightness_limit = st.sidebar.slider(
            "blightness_limit",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            key="random_brighness_contrast_brightness_limit")
        contrast_limit = st.sidebar.slider(
            "contrast_limit",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            key="random_brightness_contrast_contrast_limit")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="random_brightness_contrast_p")
        compose.append(A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p))
    if "CoarseDropout" in augmentations:
        st.sidebar.markdown("CoarseDropout")
        max_holes = st.sidebar.number_input(
            "max_holes",
            min_value=1,
            max_value=20,
            value=8,
            step=1,
            key="coarse_dropout_max_holes")
        max_height = st.sidebar.number_input(
            "max_height",
            min_value=1,
            max_value=100,
            value=8,
            step=1,
            key="coarse_dropout_max_height")
        max_width = st.sidebar.number_input(
            "max_width",
            min_value=1,
            max_value=100,
            value=8,
            step=1,
            key="coarse_dropout_max_width")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="coarse_dropout_p")
        compose.append(A.CoarseDropout(
            max_holes=max_holes,
            max_height=max_height,
            max_width=max_width,
            p=p))
    if "Downscale" in augmentations:
        st.sidebar.markdown("Downscale")
        scale_min = st.sidebar.slider(
            "scale_min",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            key="downscale_scale_min")
        scale_max = st.sidebar.slider(
            "scale_max",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            key="downscale_scale_max")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="downscale_p")
        compose.append(A.Downscale(
            scale_min=scale_min,
            scale_max=scale_max,
            p=p))
    if "FancyPCA" in augmentations:
        st.sidebar.markdown("FancyPCA")
        alpha = st.sidebar.slider(
            "alpha",
            min_value=0.0,
            max_value=5.0,
            value=0.1,
            step=0.1,
            key="fancy_pca_alpha")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="fancy_pca_p")
        compose.append(A.FancyPCA(
            alpha=alpha,
            p=p))
    if "HueSaturationValue" in augmentations:
        st.sidebar.markdown("HueSaturationValue")
        hue_shift_limit = st.sidebar.slider(
            "hue_shift_limit",
            min_value=0,
            max_value=255,
            value=20,
            step=1,
            key="hue_saturation_value_hue_shift_limit")
        sat_shift_limit = st.sidebar.slider(
            "sat_shift_limit",
            min_value=0,
            max_value=255,
            value=30,
            step=1,
            key="hue_saturation_value_sat_shift_limit")
        val_shift_limit = st.sidebar.slider(
            "val_shift_limit",
            min_value=0,
            max_value=255,
            value=20,
            step=1,
            key="hue_saturation_value_val_shift_limit")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="hue_saturation_value_p")
        compose.append(A.HueSaturationValue(
            hue_shift_limit=hue_shift_limit,
            sat_shift_limit=sat_shift_limit,
            val_shift_limit=val_shift_limit,
            p=p))
    return A.Compose(compose)


def crop_image_from_gray(image: np.ndarray, threshold: int = 7):
    if image.ndim == 2:
        mask = image > threshold
        return image[np.ix_(mask.any(1), mask.any(0))]
    elif image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray_image > threshold

    check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if (check_shape == 0):
        return image
    else:
        image1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        image2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        image3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]

        image = np.stack([image1, image2, image3], axis=-1)
        return image


def apply_augmentation(image: Image, augmentation, circle_crop=False):
    image_np = np.array(image)
    if circle_crop:
        image_np = crop_image_from_gray(image_np)
    augmented = augmentation(image=image_np)
    return Image.fromarray(augmented["image"])
