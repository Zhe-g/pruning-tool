"""示例模型动物园 + 合成数据 + 最小训练器。

这一层存在的唯一目的，是让下游的剪枝工具（依赖图、准则、调度器）**不必自带模型**：
只要 ``import pruning_toolkit.zoo``，就能拿到一批故意「结构各具特点」的小模型、
完全离线可复现的合成分类数据，以及一个够用就好的训练循环。

三个子模块
----------
* :mod:`.models` —— 示例模型工厂：``mlp`` / ``small_vgg`` / ``resnet_small`` /
  ``depthwise_cnn`` / ``inception_like``，外加 :func:`~.models.build_model` 与
  :func:`~.models.example_input_for`。
* :mod:`.data` —— 离线合成分类数据：高斯簇 ``blobs``、双月牙 ``two_moons``、
  模板图 ``tiny_images``，统一由 :func:`~.data.get_loaders` 返回 DataLoader。
* :mod:`.train` —— 最小训练器：:func:`~.train.evaluate` / :func:`~.train.fit`，
  其中 ``fit`` 支持 ``post_step_hook``，供剪枝工具做「掩码感知微调」。

设计约定
--------
* 只依赖 ``torch``（标准库可用），不引入 ``torchvision`` / ``pandas`` / ``matplotlib``；
* 完全离线，绝不下载数据；
* 所有随机性都由显式 ``seed`` 参数驱动，同 seed 结果可复现；
* 命名标识符用英文，面向用户的说明与输出用中文。
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import data, models, train
from .data import get_loaders, make_blobs, make_tiny_images, make_two_moons
from .models import (
    all_models,
    build_model,
    depthwise_cnn,
    example_input_for,
    inception_like,
    mlp,
    resnet_small,
    small_vgg,
)
from .train import evaluate, fit, format_history, resolve_device, trainable_parameters

__all__: List[str] = [
    # 子模块
    "models",
    "data",
    "train",
    # models
    "mlp",
    "small_vgg",
    "resnet_small",
    "depthwise_cnn",
    "inception_like",
    "all_models",
    "build_model",
    "example_input_for",
    # data
    "make_blobs",
    "make_two_moons",
    "make_tiny_images",
    "get_loaders",
    # train
    "evaluate",
    "fit",
    "trainable_parameters",
    "format_history",
    "resolve_device",
]


def describe() -> Dict[str, Any]:
    """返回本层的自描述信息，便于 CLI / 报告打印。

    返回:
        字典，含 ``"models"`` （模型名列表）、``"datasets"`` （数据集名列表）
        与 ``"note"`` （一句中文说明）。
    """
    return {
        "models": sorted(all_models().keys()),
        "datasets": ["blobs", "two_moons", "tiny_images"],
        "note": "全部为离线合成数据与小型示例网络，可用 seed 保证复现。",
    }
