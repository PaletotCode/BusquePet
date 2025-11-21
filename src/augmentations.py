"""Realistic augmentation utilities tailored for BusquePet training."""

from __future__ import annotations

import os
import random
from typing import Optional

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image


class RealisticAugmentations:
    """Advanced augmentation suite that mimics adverse capture conditions."""

    def __init__(self, image_size: int = 224, p: float = 0.5) -> None:
        self.image_size = image_size
        self.p = p

    def low_light_augmentation(self, image: np.ndarray, **_: dict) -> np.ndarray:
        """Simulate low-light captures by shrinking the V channel in HSV."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        factor = random.uniform(0.3, 0.7)
        v = (v * factor).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def add_shadows(self, image: np.ndarray, **_: dict) -> np.ndarray:
        """Add irregular soft shadows to emulate obstacles blocking light."""
        h, w = image.shape[:2]
        top_y = random.randint(0, h // 2)
        bottom_y = random.randint(h // 2, h)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.rectangle(mask, (0, top_y), (w, bottom_y), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 30)
        shadow_intensity = random.uniform(0.3, 0.7)
        mask = 1.0 - (mask * (1.0 - shadow_intensity))
        for channel in range(3):
            image[:, :, channel] = (image[:, :, channel] * mask).astype(np.uint8)
        return image

    def jpeg_compression(self, image: np.ndarray, quality: Optional[int] = None, **_: dict) -> np.ndarray:
        """Apply a heavy JPEG compression artefact."""
        quality = random.randint(30, 80) if quality is None else quality
        pil_img = Image.fromarray(image)
        import io

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return np.array(Image.open(buffer))

    def add_iso_noise(self, image: np.ndarray, **_: dict) -> np.ndarray:
        """Inject ISO-like sensor noise."""
        noise_level = random.uniform(10, 40)
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy_image.astype(np.uint8)

    def add_fog(self, image: np.ndarray, **_: dict) -> np.ndarray:
        """Overlay a fog layer with adjustable transparency."""
        h, w = image.shape[:2]
        fog_intensity = random.uniform(0.3, 0.7)
        fog_color = np.array([200, 200, 200], dtype=np.uint8)
        fog_layer = np.ones((h, w, 3), dtype=np.float32) * fog_color
        blended = cv2.addWeighted(image.astype(np.float32), 1 - fog_intensity, fog_layer, fog_intensity, 0)
        return blended.astype(np.uint8)

    def motion_blur(self, image: np.ndarray, **_: dict) -> np.ndarray:
        """Simulate motion blur by rotating a 1D kernel."""
        kernel_size = random.choice([7, 9, 11, 13])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        angle = random.uniform(0, 360)
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        return cv2.filter2D(image, -1, kernel)

    def perspective_transform(self, image: np.ndarray, **_: dict) -> np.ndarray:
        """Apply a random perspective warp."""
        h, w = image.shape[:2]
        offset = random.uniform(0.1, 0.3)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32(
            [
                [w * offset * random.uniform(-1, 1), h * offset * random.uniform(-1, 1)],
                [w - w * offset * random.uniform(-1, 1), h * offset * random.uniform(-1, 1)],
                [w - w * offset * random.uniform(-1, 1), h - h * offset * random.uniform(-1, 1)],
                [w * offset * random.uniform(-1, 1), h - h * offset * random.uniform(-1, 1)],
            ]
        )
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def get_training_transform(self):
        """Return the full training-time augmentation pipeline."""
        return A.Compose(
            [
                A.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.7, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.Lambda(image=self.low_light_augmentation, p=1.0),
                        A.Lambda(image=self.add_shadows, p=1.0),
                        A.Lambda(image=self.add_fog, p=1.0),
                    ],
                    p=self.p,
                ),
                A.OneOf(
                    [
                        A.Lambda(image=self.jpeg_compression, p=1.0),
                        A.Lambda(image=self.add_iso_noise, p=1.0),
                        A.Lambda(image=self.motion_blur, p=1.0),
                    ],
                    p=self.p,
                ),
                A.Lambda(image=self.perspective_transform, p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def get_validation_transform(self):
        """Return the deterministic validation transform."""
        return A.Compose(
            [
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


def test_augmentations() -> None:
    """Utility for visually testing the augmentation blocks."""
    import matplotlib.pyplot as plt

    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    aug = RealisticAugmentations(image_size=224, p=1.0)

    augmented_images = {
        "Original": test_image,
        "Low Light": aug.low_light_augmentation(test_image.copy()),
        "Shadows": aug.add_shadows(test_image.copy()),
        "JPEG Compression": aug.jpeg_compression(test_image.copy(), quality=40),
        "ISO Noise": aug.add_iso_noise(test_image.copy()),
        "Fog": aug.add_fog(test_image.copy()),
        "Motion Blur": aug.motion_blur(test_image.copy()),
        "Perspective": aug.perspective_transform(test_image.copy()),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for idx, (title, img) in enumerate(augmented_images.items()):
        axes[idx].imshow(img)
        axes[idx].set_title(title)
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/augmentations_test.png", dpi=150)
    print("Augmentation test saved to outputs/augmentations_test.png")


if __name__ == "__main__":
    test_augmentations()
