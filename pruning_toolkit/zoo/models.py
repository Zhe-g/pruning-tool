"""示例模型工厂：一批刻意覆盖「剪枝难点」的小型网络。

每个模型都对应下游剪枝工具的一个关注点：

===================  ==========================================================
工厂函数             演示的剪枝结构
===================  ==========================================================
``mlp``              纯 ``nn.Linear`` 堆叠，无通道耦合，最简单的基线。
``small_vgg``        链式 Conv-BN-ReLU + MaxPool，通道逐层 16/32/64，
                     剪枝后参数量下降非常直观（利于演示与画图）。
``resnet_small``     真正的残差网络，短接是显式的 ``out + identity`` 张量加法；
                     剪枝工具靠 fx 追踪这个 ``add`` 推导「主干通道 ↔ 短接通道」
                     的耦合关系，因此**不要**用 ``nn.Sequential`` 把恒等映射藏起来。
``depthwise_cnn``    深度可分离卷积 ``Conv2d(c, c, groups=c)``，
                     输入输出通道严格 1:1 耦合，剪枝时必须同步处理。
``inception_like``   含 ``torch.cat([a, b], dim=1)`` 的分支拼接，
                     剪枝工具在拼接处只能「保守跳过」，否则通道索引会错位。
===================  ==========================================================

设计约束
--------
* 只依赖 ``torch``；所有模型都接受 ``num_classes``，输出形状恒为 ``(N, num_classes)``；
* 全部可被 ``torch.fx.symbolic_trace`` 成功追踪（下游剪枝工具的硬前提）；
* 通道数刻意取小（16/32/64），输入 32x32 时 CPU 前向足够快。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "mlp",
    "small_vgg",
    "resnet_small",
    "depthwise_cnn",
    "inception_like",
    "all_models",
    "build_model",
    "example_input_for",
    "image_models",
    "MODEL_NAMES",
    "IMAGE_MODELS",
    "DEFAULT_IMAGE_SIZE",
    "BasicBlock",
    "SeparableBlock",
    "SmallVGG",
    "ResNetSmall",
    "DepthwiseCNN",
    "InceptionLike",
]

#: 需要 ``(N, C, H, W)`` 图像输入的模型名集合。
IMAGE_MODELS: Tuple[str, ...] = (
    "small_vgg",
    "resnet_small",
    "depthwise_cnn",
    "inception_like",
)

#: :func:`example_input_for` 默认使用的图像边长。
DEFAULT_IMAGE_SIZE = 32

#: 全部可用模型名（与 :func:`all_models` 的键一致）。
MODEL_NAMES: Tuple[str, ...] = ("mlp",) + IMAGE_MODELS


# ---------------------------------------------------------------------- MLP


def mlp(
    in_features: int = 2,
    hidden: Sequence[int] = (64, 64),
    num_classes: int = 10,
) -> nn.Module:
    """构造一个全连接 + ReLU 的分类器。

    参数:
        in_features: 输入特征维度。
        hidden: 各隐藏层宽度，例如 ``(64, 64)``；传空序列即退化成单个线性层。
        num_classes: 分类数，也是输出维度。

    返回:
        ``nn.Module``，前向输入 ``(N, in_features)``，输出 ``(N, num_classes)``。
    """
    layers = []
    prev = int(in_features)
    for width in hidden:
        layers.append(nn.Linear(prev, int(width)))
        layers.append(nn.ReLU(inplace=True))
        prev = int(width)
    layers.append(nn.Linear(prev, int(num_classes)))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ VGG 风格


class SmallVGG(nn.Module):
    """链式卷积网络：4 组 Conv-BN-ReLU，通道 16/32/64/64，末端全局池化 + 全连接。

    结构和 ``torchvision.models.vgg`` 同族，但通道数刻意压小，方便演示结构化通道剪枝
    带来的参数量下降（大部分参数集中在最后一层 ``Linear``，剪掉卷积通道后会显著缩小）。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        stem_channels: Sequence[int] = (16, 32, 64, 64),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        widths = [int(c) for c in stem_channels]
        if len(widths) < 4:
            raise ValueError("small_vgg 至少需要 4 个卷积层，当前只有 %d 个" % len(widths))

        blocks = []
        prev = self.in_channels
        for index, width in enumerate(widths):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                )
            )
            # 每两层下采样一次，保证 4 层卷积后空间尺寸仍有 8x8（32x32 输入）。
            if index % 2 == 1:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(prev, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：``(N, in_channels, H, W)`` -> ``(N, num_classes)``。"""
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


def small_vgg(in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """返回 :class:`SmallVGG` 实例（链式卷积，便于演示通道剪枝）。

    参数:
        in_channels: 输入图像通道数。
        num_classes: 分类数。

    返回:
        ``nn.Module``，前向输入 ``(N, in_channels, H, W)``，输出 ``(N, num_classes)``。
    """
    return SmallVGG(in_channels=in_channels, num_classes=num_classes)


# ---------------------------------------------------------------------- ResNet


class BasicBlock(nn.Module):
    """残差基本块：conv-bn-relu-conv-bn，再与恒等短接做**显式张量相加**。

    短接特意写成 ``out + identity`` 而不是用 ``nn.Sequential`` 包一层恒等映射：
    下游剪枝工具需要 ``torch.fx`` 在计算图里看到一个真实的 ``add`` 节点，
    才能据此推出「主干输出通道」与「短接通道」共享同一个通道索引空间。
    通道数变化（``stride != 1`` 或 ``in_channels != out_channels``）时，
    短接退化为 1x1 卷积 + BN，同样以显式相加的方式接入。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        out_channels = int(out_channels if out_channels is not None else in_channels)
        self.in_channels = int(in_channels)
        self.out_channels = out_channels
        self.stride = int(stride)

        self.conv1 = nn.Conv2d(
            self.in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.downsample: Optional[nn.Module] = None
        if self.stride != 1 or self.in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                norm_layer(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：``add(主干输出, 恒等短接)`` 后再 ReLU。"""
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 显式张量相加：fx 图上会出现 add 节点，剪枝工具据此推导通道耦合。
        out = out + identity
        out = self.relu(out)
        return out


class ResNetSmall(nn.Module):
    """小型残差网络：卷积 stem + 4 个 :class:`BasicBlock`（两阶段）+ 全局池化 + 全连接。"""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        widths: Sequence[int] = (32, 64),
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        widths = [int(w) for w in widths]
        if not widths:
            raise ValueError("widths 不能为空")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage 至少为 1")

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        prev = widths[0]
        for stage_index, width in enumerate(widths):
            # 第一阶段保持分辨率（stride=1），后续阶段各下采样一次。
            stride = 1 if stage_index == 0 else 2
            for block_index in range(int(blocks_per_stage)):
                stages.append(
                    BasicBlock(prev, width, stride=stride if block_index == 0 else 1)
                )
                prev = width
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(prev, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：``(N, in_channels, H, W)`` -> ``(N, num_classes)``。"""
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


def resnet_small(in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """返回 :class:`ResNetSmall` 实例（真残差连接，短接为显式张量相加）。

    参数:
        in_channels: 输入图像通道数。
        num_classes: 分类数。

    返回:
        ``nn.Module``，前向输入 ``(N, in_channels, H, W)``，输出 ``(N, num_classes)``。
    """
    return ResNetSmall(in_channels=in_channels, num_classes=num_classes)


# ---------------------------------------------------------------- 深度可分离卷积


class SeparableBlock(nn.Module):
    """深度可分离卷积块：深度卷积 ``groups=in`` -> 逐点卷积 -> BN-ReLU。

    深度卷积的输入、输出通道被 ``groups`` 强制 1:1 绑定，
    剪枝工具必须把这两个通道当作同一个通道处理（同步裁剪）。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：空间尺寸不变，通道数由 ``in_channels`` 变为 ``out_channels``。"""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class DepthwiseCNN(nn.Module):
    """以深度可分离卷积为主体的小型 CNN（4 个 :class:`SeparableBlock`）。"""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        widths: Sequence[int] = (16, 32, 64, 64),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        widths = [int(w) for w in widths]
        if len(widths) < 4:
            raise ValueError("depthwise_cnn 至少需要 4 个卷积层，当前只有 %d 个" % len(widths))

        blocks = []
        prev = self.in_channels
        for index, width in enumerate(widths):
            blocks.append(SeparableBlock(prev, width))
            if index % 2 == 1:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(prev, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：``(N, in_channels, H, W)`` -> ``(N, num_classes)``。"""
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


def depthwise_cnn(in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """返回 :class:`DepthwiseCNN` 实例（演示深度卷积的输入输出通道 1:1 耦合）。

    参数:
        in_channels: 输入图像通道数。
        num_classes: 分类数。

    返回:
        ``nn.Module``，前向输入 ``(N, in_channels, H, W)``，输出 ``(N, num_classes)``。
    """
    return DepthwiseCNN(in_channels=in_channels, num_classes=num_classes)


# ------------------------------------------------------------------ 多分支拼接


class InceptionLike(nn.Module):
    """含通道拼接的多分支网络：两条并行卷积分支 -> ``torch.cat`` -> 逐点卷积融合。

    ``torch.cat`` 会把两条分支的通道拼成一段连续索引，单剪某一支会破坏
    通道编号与下游权重的对应关系。剪枝工具遇到拼接节点只能**保守跳过**，
    这个模型就是用来触发 / 验证那条保守路径的。
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, branch_channels: int = 8) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.branch_channels = int(branch_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(16, self.branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(16, self.branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.branch_channels),
            nn.ReLU(inplace=True),
        )
        fused = self.branch_channels * 2
        self.fuse = nn.Sequential(
            nn.Conv2d(fused, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(32, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：两条分支各自卷积后按通道维拼接，再融合、池化、分类。"""
        x = self.stem(x)
        a = self.branch1x1(x)
        b = self.branch3x3(x)
        # 通道维拼接：fx 图上会出现 cat 节点，剪枝工具必须在此保守跳过。
        merged = torch.cat([a, b], dim=1)
        merged = self.fuse(merged)
        merged = self.pool(merged)
        merged = self.flatten(merged)
        return self.classifier(merged)


def inception_like(in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """返回 :class:`InceptionLike` 实例（演示 ``torch.cat`` 拼接处的保守剪枝）。

    参数:
        in_channels: 输入图像通道数。
        num_classes: 分类数。

    返回:
        ``nn.Module``，前向输入 ``(N, in_channels, H, W)``，输出 ``(N, num_classes)``。
    """
    return InceptionLike(in_channels=in_channels, num_classes=num_classes)


# ---------------------------------------------------------------------- 注册表


def all_models() -> Dict[str, Callable[..., nn.Module]]:
    """返回「模型名 -> 工厂函数」的字典。

    返回:
        ``{"mlp": mlp, "small_vgg": small_vgg, "resnet_small": resnet_small,
        "depthwise_cnn": depthwise_cnn, "inception_like": inception_like}``。
        所有工厂函数都接受 ``num_classes`` 关键字参数并返回 ``nn.Module``。
    """
    return {
        "mlp": mlp,
        "small_vgg": small_vgg,
        "resnet_small": resnet_small,
        "depthwise_cnn": depthwise_cnn,
        "inception_like": inception_like,
    }


def image_models() -> Tuple[str, ...]:
    """返回需要 ``(N, C, H, W)`` 图像输入的模型名（元组）。"""
    return IMAGE_MODELS


def build_model(name: str, num_classes: int = 10, **kwargs: Any) -> nn.Module:
    """按名字构建模型。

    参数:
        name: 模型名，必须是 :func:`all_models` 的键之一。
        num_classes: 分类数，透传给工厂函数。
        **kwargs: 其余参数原样透传给工厂函数（如 ``in_features`` / ``in_channels``）。

    返回:
        构建好的 ``nn.Module``。

    异常:
        KeyError: 名字未知，异常信息里会列出全部可选名字。
    """
    registry = all_models()
    if name not in registry:
        raise KeyError(
            "未知模型名 %r；可选名字：%s" % (name, ", ".join(sorted(registry)))
        )
    return registry[name](num_classes=num_classes, **kwargs)


def example_input_for(
    name: str,
    batch_size: int = 4,
    num_classes: int = 10,
    in_channels: int = 3,
    in_features: int = 2,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """返回适合指定模型的示例输入张量。

    确定性说明：
        本函数**不使用** ``torch.randn``，而是返回 ``torch.zeros`` 构造的全零张量
        （形状随机采样无关，因此结果与随机种子无关、每次调用完全一致）。
        零输入对 ``fx`` 追踪、形状推断与通道剪枝的健全性检查已经足够；
        若需要非零数值来观察激活，请改用 ``torch.Generator().manual_seed(seed)``
        自行生成 ``randn``。

    参数:
        name: 模型名；``"mlp"`` 返回 ``(N, in_features)``，图像模型返回 ``(N, in_channels, S, S)``。
        batch_size: 批大小。
        num_classes: 分类数（仅用于校验与文档，不影响形状）。
        in_channels: 图像模型的输入通道数。
        in_features: ``"mlp"`` 的输入特征维度。
        image_size: 图像模型的空间边长。

    返回:
        ``float32`` 张量，形状与目标模型前向输入一致。

    异常:
        KeyError: 名字未知，异常信息里会列出全部可选名字。
    """
    registry = all_models()
    if name not in registry:
        raise KeyError(
            "未知模型名 %r；可选名字：%s" % (name, ", ".join(sorted(registry)))
        )
    if name == "mlp":
        return torch.zeros(int(batch_size), int(in_features), dtype=torch.float32)
    return torch.zeros(
        int(batch_size),
        int(in_channels),
        int(image_size),
        int(image_size),
        dtype=torch.float32,
    )
