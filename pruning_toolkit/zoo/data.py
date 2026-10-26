"""离线合成分类数据：不需要任何下载，光靠 ``torch`` 就能造出可训练的玩具数据集。

三个数据集
----------
===============  ==============================================================
``blobs``        各高斯簇中心均匀分布在半径 2.5 的圆上，整体再随机旋转一个角度；
                 簇之间刻意留一点**重叠**，剪枝后的精度变化才看得出来。
``two_moons``    经典双月牙，非线性可分，用来检验小模型容量是否够用。
``tiny_images``  每个类别一张由 seed 决定的固定随机「模板图」，样本 = 模板 + 高斯噪声；
                 给 ``small_vgg`` / ``resnet_small`` 之类的卷积网络当输入。
===============  ==============================================================

统一入口是 :func:`get_loaders`：按 8:2 划分训练/测试，返回两个 ``DataLoader``
和一个 ``meta`` 字典（含 ``name`` / ``in_features`` 或 ``in_channels`` /
``image_size`` / ``num_classes`` / ``n_train`` / ``n_test`` 等字段）。

复现性
------
每个函数都接受显式 ``seed``，内部一律使用 ``torch.Generator().manual_seed(seed)``；
``DataLoader`` 的 ``shuffle`` 也用固定 ``generator``，因此同 seed 的批次顺序完全一致。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "DATASET_NAMES",
    "make_blobs",
    "make_two_moons",
    "make_tiny_images",
    "get_loaders",
]

#: 全部可用数据集名。
DATASET_NAMES: Tuple[str, ...] = ("blobs", "two_moons", "tiny_images")

#: 训练集占比。
TRAIN_RATIO = 0.8

#: :func:`make_blobs` 中高斯簇中心所在的圆半径。
BLOB_RADIUS = 2.5


def _generator(seed: int) -> torch.Generator:
    """构造一个由 ``seed`` 确定的 CPU ``torch.Generator``。"""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


# ---------------------------------------------------------------------- blobs


def make_blobs(
    n_samples: int = 1200,
    n_features: int = 2,
    n_classes: int = 3,
    spread: float = 0.9,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成「高斯簇」分类数据（scikit-learn ``make_blobs`` 的轻量替代品）。

    各类簇中心均匀分布在半径 ``2.5`` 的圆上，整体再乘一个由 ``seed`` 决定的随机旋转矩阵；
    簇内样本服从以该中心为均值、``spread`` 为标准差的高斯分布。
    当 ``spread`` 相对簇间距不太小时，类别之间会**少量重叠** —— 这是刻意为之：
    完全线性可分的数据上剪枝几乎不掉精度，看不出差异。

    参数:
        n_samples: 总样本数；按类尽量均分，余数分给靠前的类别。
        n_features: 特征维度；大于 2 时前两维按圆排布，其余维度为标准高斯噪声。
        n_classes: 类别数。
        spread: 簇内高斯标准差（越大越重叠）。
        seed: 随机种子，决定中心、旋转与簇内样本。

    返回:
        ``(features, labels)``：``features`` 为 ``float32``、形状 ``(n_samples, n_features)``；
        ``labels`` 为 ``int64``、形状 ``(n_samples,)``。
    """
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正整数")
    if n_features < 1:
        raise ValueError("n_features 必须为正整数")
    if n_classes < 1:
        raise ValueError("n_classes 必须为正整数")

    generator = _generator(seed)

    angles = torch.arange(n_classes, dtype=torch.float32) * (2.0 * math.pi / float(n_classes))
    angles = angles + torch.rand(1, generator=generator) * (2.0 * math.pi / float(n_classes))
    centers = torch.stack(
        [BLOB_RADIUS * torch.cos(angles), BLOB_RADIUS * torch.sin(angles)], dim=1
    )  # (n_classes, 2)

    # 每个样本的标签：尽量均分，余数给靠前的类别。
    counts = [n_samples // n_classes] * n_classes
    for index in range(n_samples % n_classes):
        counts[index] += 1
    labels = torch.cat(
        [torch.full((count,), index, dtype=torch.int64) for index, count in enumerate(counts)]
    )

    # 每个样本对应的簇中心。
    center_index = torch.cat(
        [torch.full((count,), index, dtype=torch.int64) for index, count in enumerate(counts)]
    )
    features = centers[center_index]  # (n_samples, 2)

    # 额外特征维：纯高斯噪声，不携带类别信息。
    if n_features > 2:
        extra = torch.randn(n_samples, n_features - 2, generator=generator) * spread
        features = torch.cat([features, extra], dim=1)

    features = features + torch.randn(n_samples, n_features, generator=generator) * float(spread)
    # 右乘旋转矩阵：对最后一维（特征维）做整体旋转，不改变簇间距离分布。
    features = features @ _rotation(n_features, generator)
    return features.to(torch.float32), labels.to(torch.int64)


def _rotation(dim: int, generator: torch.Generator) -> torch.Tensor:
    """返回一个 ``dim x dim`` 的随机正交矩阵，用于整体旋转特征空间。

    由若干对坐标轴上的随机角度旋转复合而成，保证正交（不改变簇间距离分布）。
    """
    matrix = torch.eye(dim, dtype=torch.float32)
    for i in range(0, dim - 1, 2):
        theta = torch.rand(1, generator=generator).item() * 2.0 * math.pi
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        pair = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float32)
        matrix[i : i + 2, i : i + 2] = pair
    return matrix


# ------------------------------------------------------------------- two moons


def make_two_moons(
    n_samples: int = 1200,
    noise: float = 0.15,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成经典「双月牙」二分类数据。

    上半月牙来自 ``(cos t, sin t)``，下半月牙来自 ``(1 - cos t, -sin t + 0.5)``，
    ``t`` 在 ``[0, pi]`` 上均匀取值；随后加标准差为 ``noise`` 的高斯噪声。
    两团数据互相咬合、线性不可分，适合检验小模型的非线性容量。

    参数:
        n_samples: 总样本数，两半各占一半（奇数时上半多一个）。
        noise: 高斯噪声标准差。
        seed: 随机种子。

    返回:
        ``(features, labels)``：``features`` 为 ``float32``、形状 ``(n_samples, 2)``；
        ``labels`` 为 ``int64``（取值 0/1）。
    """
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正整数")

    generator = _generator(seed)
    n_outer = (n_samples + 1) // 2
    n_inner = n_samples - n_outer

    t_outer = torch.rand(n_outer, generator=generator) * math.pi
    outer = torch.stack([torch.cos(t_outer), torch.sin(t_outer)], dim=1)

    t_inner = torch.rand(n_inner, generator=generator) * math.pi
    inner = torch.stack([1.0 - torch.cos(t_inner), 0.5 - torch.sin(t_inner)], dim=1)

    features = torch.cat([outer, inner], dim=0)
    labels = torch.cat(
        [torch.zeros(n_outer, dtype=torch.int64), torch.ones(n_inner, dtype=torch.int64)]
    )
    features = features + torch.randn(n_samples, 2, generator=generator) * float(noise)
    return features.to(torch.float32), labels.to(torch.int64)


# ----------------------------------------------------------------- tiny images


def make_tiny_images(
    n_samples: int = 1000,
    n_classes: int = 10,
    image_size: int = 32,
    in_channels: int = 3,
    seed: int = 0,
    template_scale: float = 1.0,
    noise: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成合成小图像分类数据：每类一张固定随机模板 + 高斯噪声。

    每个类别的模板图由 ``seed`` 决定（``template_scale`` 倍标准高斯），
    同类样本共享模板、只加不同噪声；因此任务本身是「去噪后做模板匹配」，
    卷积网络在几轮之内就能学到较高精度，非常适合当剪枝实验的快速基线。

    参数:
        n_samples: 总样本数，按类尽量均分。
        n_classes: 类别数（= 模板数量）。
        image_size: 图像边长 ``H = W``。
        in_channels: 图像通道数。
        seed: 随机种子，决定全部模板图与噪声。
        template_scale: 模板图的标准差缩放（越大类间差异越明显）。
        noise: 样本噪声的标准差。

    返回:
        ``(images, labels)``：``images`` 为 ``float32``、形状 ``(n_samples, in_channels, H, W)``；
        ``labels`` 为 ``int64``、形状 ``(n_samples,)``。
    """
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正整数")
    if n_classes < 1:
        raise ValueError("n_classes 必须为正整数")
    if image_size < 1:
        raise ValueError("image_size 必须为正整数")
    if in_channels < 1:
        raise ValueError("in_channels 必须为正整数")

    generator = _generator(seed)
    shape = (n_classes, in_channels, image_size, image_size)
    templates = torch.randn(*shape, generator=generator) * float(template_scale)  # (K, C, H, W)

    counts = [n_samples // n_classes] * n_classes
    for index in range(n_samples % n_classes):
        counts[index] += 1
    labels = torch.cat(
        [torch.full((count,), index, dtype=torch.int64) for index, count in enumerate(counts)]
    )
    images = templates[labels] + torch.randn(n_samples, in_channels, image_size, image_size, generator=generator) * float(noise)
    return images.to(torch.float32), labels.to(torch.int64)


# --------------------------------------------------------------------- loaders


class _IndexedTensorDataset(TensorDataset):
    """``TensorDataset`` 的薄封装，附带 ``num_features`` / ``num_channels`` 之类的元信息。"""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor, extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(features, labels)
        self.extra: Dict[str, Any] = dict(extra or {})


def get_loaders(
    name: str = "tiny_images",
    batch_size: int = 64,
    seed: int = 0,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """按名字构造数据集并切成训练/测试两个 ``DataLoader``。

    数据集按 8:2 划分：先用固定 ``generator`` 打乱索引，再切分，
    因此同 ``seed`` 下划分完全一致。训练 ``DataLoader`` 打开 ``shuffle=True``
    并使用固定 ``generator``（批次顺序可复现）；测试 ``DataLoader`` 不打乱。

    参数:
        name: ``"blobs"`` / ``"two_moons"`` / ``"tiny_images"`` 之一。
        batch_size: 批大小。
        seed: 随机种子，同时决定数据生成、划分与训练集 shuffle 顺序。
        **kwargs: 透传给对应 ``make_*`` 函数的额外参数
            （例如 ``n_samples``、``n_classes``、``spread``、``image_size``、
            ``in_channels``、``noise``）。注意 ``seed`` 不从这里传，
            统一使用本函数的 ``seed`` 参数。

    返回:
        ``(train_loader, test_loader, meta)``。``meta`` 至少含：
        ``name``、``num_classes``、``n_train``、``n_test``、``batch_size``、``seed``，
        以及按数据集类型补充的 ``in_features``（blobs / two_moons）
        或 ``in_channels`` + ``image_size``（tiny_images）。

    异常:
        KeyError: 数据集名未知，异常信息里会列出全部可选名字。
    """
    if name not in DATASET_NAMES:
        raise KeyError("未知数据集名 %r；可选名字：%s" % (name, ", ".join(DATASET_NAMES)))
    if batch_size < 1:
        raise ValueError("batch_size 必须为正整数")

    options = dict(kwargs)
    options.pop("seed", None)  # seed 统一由本函数给出，避免出现两个来源。

    if name == "blobs":
        n_classes = int(options.pop("n_classes", 3))
        options.pop("n_features", None)  # 特征维度与类别数绑定，见下。
        features, labels = make_blobs(n_classes=n_classes, n_features=n_classes, seed=seed, **options)
        feature_shape: Tuple[int, ...] = (features.shape[1],)
    elif name == "two_moons":
        features, labels = make_two_moons(seed=seed, **options)
        feature_shape = (features.shape[1],)
    else:  # tiny_images
        n_classes = int(options.get("n_classes", 10))
        options.pop("n_classes", None)
        features, labels = make_tiny_images(n_classes=n_classes, seed=seed, **options)
        feature_shape = tuple(features.shape[1:])

    n_total = int(features.shape[0])
    n_train = int(round(n_total * TRAIN_RATIO))
    n_train = max(1, min(n_total - 1, n_train)) if n_total > 1 else n_total

    split_generator = _generator(seed + 10_000)
    permutation = torch.randperm(n_total, generator=split_generator)
    train_index = permutation[:n_train]
    test_index = permutation[n_train:]

    train_set = _IndexedTensorDataset(features[train_index], labels[train_index])
    test_set = _IndexedTensorDataset(features[test_index], labels[test_index])

    shuffle_generator = _generator(seed + 20_000)
    train_loader = DataLoader(
        train_set,
        batch_size=int(batch_size),
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    num_classes = int(torch.unique(labels).numel())
    meta: Dict[str, Any] = {
        "name": name,
        "num_classes": num_classes,
        "n_total": n_total,
        "n_train": int(train_index.numel()),
        "n_test": int(test_index.numel()),
        "batch_size": int(batch_size),
        "seed": int(seed),
    }
    if name == "tiny_images":
        meta["in_channels"] = int(feature_shape[0])
        meta["image_size"] = int(feature_shape[1])
        meta["in_features"] = int(feature_shape[0] * feature_shape[1] * feature_shape[2])
    else:
        meta["in_features"] = int(feature_shape[0])
        meta["in_channels"] = 0
        meta["image_size"] = 0
    return train_loader, test_loader, meta
