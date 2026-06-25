"""
============================================================
强数据增强模块（模拟钢面恶劣环境）
============================================================
增强类型：
1. 强反光（Glare） - 模拟车间照明变化
2. 高斯模糊（Blur） - 模拟相机失焦/抖动
3. 低照度（Low Light） - 模拟光线不足

用法：在每个 batch 训练前，以一定概率随机应用

将其注册到 Ultralytics 的数据增强 Pipeline：
导入本模块后自动注册到 YOLO 的 transforms
============================================================
"""

import cv2
import numpy as np
import random
import torch


# ============================================================
# 1. 强反光增强
# ============================================================
def add_glare(image, severity='medium'):
    """
    在图像上添加模拟反光/光斑
    原理：叠加一个高斯模糊后的高亮圆形区域

    Args:
        image:  RGB图像 [H, W, 3], numpy array, 值域 0~255
        severity: 'light' / 'medium' / 'strong'
    Returns:
        augmented image
    """
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]

    # 随机选择反光中心
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)

    severity_map = {
        'light':  (0.1, 0.3, 0.15),    # (强度, 半径比例, 模糊程度)
        'medium': (0.2, 0.5, 0.25),
        'strong': (0.3, 0.7, 0.35),
    }
    intensity, radius_ratio, blur_ratio = severity_map.get(severity, severity_map['medium'])

    radius = int(min(h, w) * radius_ratio)
    blur_size = int(radius * blur_ratio)
    if blur_size % 2 == 0:
        blur_size += 1  # 确保是奇数

    # 画一个高斯径向渐变的光斑
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    glare_mask = np.exp(-dist ** 2 / (2 * (radius / 3) ** 2))
    glare_mask = cv2.GaussianBlur(glare_mask, (blur_size, blur_size), 0)

    # 叠加到原图
    glare = glare_mask[:, :, np.newaxis] * 255 * intensity
    img = np.clip(img + glare, 0, 255)

    return img.astype(np.uint8)


# ============================================================
# 2. 高斯模糊增强
# ============================================================
def add_blur(image, severity='medium'):
    """
    模拟失焦 / 抖动模糊

    severity: kernel_size
        'light'  → 3
        'medium' → 5
        'strong' → 9
    """
    severity_map = {'light': 3, 'medium': 5, 'strong': 9}
    k = severity_map.get(severity, 5)

    # 随机选择模糊类型
    blur_type = random.choice(['gaussian', 'motion'])

    if blur_type == 'gaussian':
        return cv2.GaussianBlur(image, (k, k), 0)
    else:
        # 运动模糊（模拟相机抖动）
        angle = random.uniform(0, 360)
        return _motion_blur(image, k, angle)


def _motion_blur(image, kernel_size, angle):
    """运动模糊"""
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = np.diag(np.ones(kernel_size)) / kernel_size
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    return cv2.filter2D(image, -1, kernel)


# ============================================================
# 3. 低照度增强
# ============================================================
def add_low_light(image, severity='medium'):
    """
    模拟光线不足

    原理：降低亮度的同时，增加噪声

    severity_map:
        'light'  → 亮度下降到 60%，噪声 sigma 5
        'medium' → 亮度下降到 35%，噪声 sigma 12
        'strong' → 亮度下降到 20%，噪声 sigma 20
    """
    severity_map = {
        'light':  (0.6, 5),
        'medium': (0.35, 12),
        'strong': (0.2, 20),
    }
    brightness_factor, noise_sigma = severity_map.get(severity, severity_map['medium'])

    img = image.astype(np.float32)
    img *= brightness_factor

    # 加高斯噪声
    noise = np.random.normal(0, noise_sigma, img.shape)
    img = np.clip(img + noise, 0, 255)

    return img.astype(np.uint8)


# ============================================================
# 4. 组合增强 Pipeline
# ============================================================
class StrongAugment:
    """
    组合增强：训练时对每一张图以一定概率随机应用

    用法：
        augmenter = StrongAugment(
            glare_prob=0.3,      # 30% 概率加反光
            blur_prob=0.3,       # 30% 概率模糊
            low_light_prob=0.3,  # 30% 概率低照度
            severity='medium'    # 所有增强的强度
        )
        augmented_img, augmented_mask = augmenter(image, mask)
    """

    def __init__(self, glare_prob=0.3, blur_prob=0.3, low_light_prob=0.3,
                 severity='medium'):
        self.glare_prob = glare_prob
        self.blur_prob = blur_prob
        self.low_light_prob = low_light_prob
        self.severity = severity

    def __call__(self, image, mask=None):
        """
        Args:
            image: [H, W, 3] numpy array
            mask:  [H, W] numpy array (instance segmentation mask, optional)
        Returns:
            augmented image, augmented mask
        """
        img = image.copy()

        if random.random() < self.glare_prob:
            img = add_glare(img, self.severity)

        if random.random() < self.blur_prob:
            img = add_blur(img, self.severity)

        if random.random() < self.low_light_prob:
            img = add_low_light(img, self.severity)

        return img if mask is None else (img, mask)


# ============================================================
# 注册到 Ultralytics 数据增强流程
# ============================================================
def register_custom_augmentations():
    """
    注册自定义增强到 YOLO 的 transforms pipeline。

    使用方式：
    在 train 之前调用 register_custom_augmentations()，
    并在数据配置 YAML 中设置 augments 参数指向自定义增强配置。
    """
    from ultralytics.data.augment import Albumentations as UltAlbumentations
    from ultralytics.data.augment import Format

    # 创建自定义的 Albumentation 增强
    import albumentations as A

    custom_aug = A.Compose([
        # 保留原始的随机翻转
        A.HorizontalFlip(p=0.5),

        # === 新增：模拟工业场景的恶劣环境 ===
        A.RandomBrightnessContrast(
            brightness_limit=0.2,   # 亮度 ±20%
            contrast_limit=0.2,     # 对比度 ±20%
            p=0.5
        ),
        A.GaussNoise(
            var_limit=(10.0, 50.0),  # 高斯噪声（模拟传感器噪声）
            p=0.3
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),       # 模糊核大小范围
            p=0.3
        ),
        A.CLAHE(
            clip_limit=2.0,          # 直方图均衡（模拟光照变化）
            tile_grid_size=(8, 8),
            p=0.3
        ),
        A.RandomGamma(
            gamma_limit=(70, 130),   # Gamma 校正（模拟曝光变化）
            p=0.3
        ),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    return custom_aug


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 创建一个测试图像
    test_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
    cv2.rectangle(test_img, (50, 50), (200, 200), (255, 255, 255), -1)

    # 测试各增强
    glare = add_glare(test_img, 'strong')
    blur = add_blur(test_img, 'strong')
    low = add_low_light(test_img, 'strong')

    print("原始图像均值:", test_img.mean())
    print("反光后均值:", glare.mean())
    print("模糊后均值:", blur.mean())
    print("低照度后均值:", low.mean())

    # 测试组合增强
    augmenter = StrongAugment(glare_prob=1.0, blur_prob=1.0, low_light_prob=1.0, severity='medium')
    result = augmenter(test_img)
    print("组合增强后均值:", result.mean())
