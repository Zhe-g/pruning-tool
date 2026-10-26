"""重要性准则：给「权重元素」「通道」「通道组」打分，分数越小越先被剪掉。

三种粒度
--------
* :func:`elementwise_importance` —— 与权重同形状的逐元素分数，用于**非结构化剪枝**；
* :func:`channel_importance` —— 形状 ``(out,)`` 的逐通道分数，用于**结构化剪枝**；
* :func:`group_importance` —— 形状 ``(count,)`` 的通道组分数。一个通道组常常同时包含
  卷积和它后面的 BatchNorm，此时把两者的分数归一化后相加，即经典 Network Slimming
  的做法（BN 的缩放因子 ``gamma`` 本身就是一种重要性度量）。

准则一览
--------
============  ==========================================================
``l1``        权重绝对值之和 / 逐元素 ``|w|``
``l2``        权重 L2 范数 / 逐元素 ``w^2``
``random``    均匀随机分数，作为随机基线（需要固定 seed 才能复现）
``taylor``    一阶泰勒 ``|w * grad|``，用梯度估计删除该权重对损失的影响
``bn_scale``  组内归一化层的 ``|gamma|``（Network Slimming），无 BN 时退化为 ``l2``
``fpgm``      Filter Pruning via Geometric Median：到几何中位数距离越小的卷积核越可剪
``none``      全 1 分数（等价于按参数顺序裁剪）
============  ==========================================================

梯度从哪来
----------
``taylor`` 与其它需要梯度的准则要传入 ``grads``：一个 ``{参数名: 梯度张量}`` 字典，
可由 :func:`collect_grads` 在一小批校准数据上反传得到。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .modules import (
    CONV_TRANSPOSE_TYPES,
    CONV_TYPES,
    LINEAR_TYPES,
    NORM_TYPES,
    channel_semantics,
)

__all__ = [
    "CRITERIA",
    "CRITERIA_REQUIRING_GRAD",
    "elementwise_importance",
    "channel_importance",
    "group_importance",
    "weight_out_dim",
    "criterion_requires_grad",
    "collect_grads",
]

CRITERIA: Dict[str, str] = {
    "l1": "权重绝对值（逐元素）或绝对值之和（逐通道）",
    "l2": "权重平方（逐元素）或 L2 范数（逐通道）",
    "random": "均匀随机分数，随机剪枝基线",
    "taylor": "一阶泰勒 |w * grad|，依赖校准梯度",
    "bn_scale": "组内 BatchNorm 缩放因子 |gamma|，无 BN 时退化为 l2",
    "fpgm": "到卷积核几何中位数的 L2 距离",
    "none": "恒为 1，等价于按顺序裁剪",
}

#: 需要提供梯度的准则
CRITERIA_REQUIRING_GRAD = ("taylor",)


def criterion_requires_grad(criterion: str) -> bool:
    return criterion in CRITERIA_REQUIRING_GRAD


def _check_criterion(criterion: str) -> None:
    if criterion not in CRITERIA:
        raise KeyError("未知的重要性准则 %r，可选：%s" % (criterion, sorted(CRITERIA)))


def _require_grad(criterion: str, grad: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if grad is None:
        raise ValueError(
            "准则 %r 需要梯度信息（参数 %s 缺少 grad）。"
            "可先用 collect_grads() 在少量校准数据上反传得到梯度和。" % (criterion, name)
        )
    return grad


def weight_out_dim(module: nn.Module) -> int:
    """返回模块权重中「输出通道」所在的维度下标。

    ``ConvTranspose`` 的权重布局是 ``(in, out/groups, *kernel)``，输出通道在第 1 维，
    这是最容易写错的地方，统一在这里处理。
    """
    if isinstance(module, CONV_TRANSPOSE_TYPES):
        return 1
    return 0


# ---------------------------------------------------------------------- 逐元素


def elementwise_importance(
    weight: torch.Tensor,
    criterion: str = "l1",
    grad: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    name: str = "weight",
) -> torch.Tensor:
    """与 ``weight`` 同形状的逐元素重要性分数。"""
    _check_criterion(criterion)
    if criterion == "l1":
        return weight.detach().abs()
    if criterion == "l2":
        detached = weight.detach()
        return detached * detached
    if criterion == "random":
        return torch.rand(
            weight.shape, generator=generator, device=weight.device, dtype=torch.float32
        ).to(weight.dtype)
    if criterion == "none":
        return torch.ones_like(weight.detach())
    if criterion == "taylor":
        grad = _require_grad(criterion, grad, name)
        if grad.shape != weight.shape:
            raise ValueError(
                "参数 %s 的梯度形状 %s 与权重形状 %s 不一致"
                % (name, tuple(grad.shape), tuple(weight.shape))
            )
        return (weight.detach() * grad.detach()).abs()
    if criterion in ("bn_scale", "fpgm"):
        # 这两个准则定义在「通道」粒度上，逐元素使用时退化为 l2 / l1
        detached = weight.detach()
        return detached * detached if criterion == "bn_scale" else detached.abs()
    raise KeyError("未知的重要性准则 %r" % criterion)


# ---------------------------------------------------------------------- 逐通道


def _filters_matrix(weight: torch.Tensor, out_dim: int) -> torch.Tensor:
    """把权重整理成 ``(out, 其余维度拉平)`` 的矩阵。"""
    if out_dim != 0:
        order = [out_dim] + [dim for dim in range(weight.ndim) if dim != out_dim]
        weight = weight.permute(order)
    return weight.detach().reshape(weight.shape[0], -1).float()


def fpgm_importance(module: nn.Module, max_filters: int = 4096) -> torch.Tensor:
    """Filter Pruning via Geometric Median。

    计算每个卷积核到所有卷积核几何中位数的 L2 距离；距离越小说明该卷积核越「冗余」，
    越应该被剪掉。卷积核数量超过 ``max_filters`` 时改用随机子集估计中位数，避免 O(n^2)
    距离矩阵爆内存（并在结果上属于近似计算）。
    """
    weight = module.weight
    out_dim = weight_out_dim(module)
    filters = _filters_matrix(weight, out_dim)
    count = filters.shape[0]
    if count <= 1:
        return torch.ones(count, device=weight.device)

    sample = filters
    if count > max_filters:
        index = torch.randperm(count, device=filters.device)[:max_filters]
        sample = filters[index]

    # 成对距离：dist[i, j] = ||f_i - f_j||
    distances = torch.cdist(sample, sample)
    median_index = int(torch.argmin(distances.sum(dim=1)).item())
    median = sample[median_index]
    return torch.linalg.vector_norm(filters - median, dim=1)


def channel_importance(
    module: nn.Module,
    criterion: str = "l1",
    grad: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """返回模块**输出通道**的重要性分数，形状 ``(out,)``。"""
    _check_criterion(criterion)

    if criterion == "fpgm":
        if module.weight.ndim < 2:
            return elementwise_importance(module.weight, "l1", name=module.__class__.__name__)
        return fpgm_importance(module)

    if criterion == "bn_scale":
        # 模块自身就是归一化层时，|gamma| 就是分数
        if isinstance(module, NORM_TYPES) and getattr(module, "weight", None) is not None:
            return module.weight.detach().abs().reshape(-1).float()
        # 否则退化为 l2（组级别会优先使用组内 BN 的 gamma）
        criterion = "l2"

    weight = module.weight
    elementwise = elementwise_importance(
        weight,
        criterion=criterion,
        grad=grad,
        generator=generator,
        name=module.__class__.__name__,
    )
    if elementwise.ndim == 1:
        return elementwise.float()

    out_dim = weight_out_dim(module)
    # 把输出维挪到最前面，再对其余维度做归约
    if out_dim != 0:
        order = [out_dim] + [dim for dim in range(elementwise.ndim) if dim != out_dim]
        elementwise = elementwise.permute(order)
    reduced_dims = tuple(range(1, elementwise.ndim))
    flat = elementwise.reshape(elementwise.shape[0], -1)
    if criterion == "l2":
        # w^2 求和后再开方 = L2 范数
        return torch.sqrt(flat.sum(dim=1).clamp_min(0.0)).float()
    if reduce == "sum":
        return flat.sum(dim=1).float()
    if reduce == "mean":
        return flat.mean(dim=1).float()
    if reduce == "max":
        return flat.max(dim=1).values.float()
    raise KeyError("未知的归约方式 %r，可选：sum / mean / max" % reduce)


# ---------------------------------------------------------------------- 通道组


def _normalize(scores: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """按均值归一化，让不同层（尺度可能差几个数量级）的分数可比。"""
    return scores / (scores.mean() + eps)


def group_importance(
    group,
    model: nn.Module,
    criterion: str = "l1",
    grads: Optional[Mapping[str, torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    normalize: bool = True,
    aggregate: str = "sum",
    reduce: str = "sum",
) -> torch.Tensor:
    """计算一个通道组内所有输出侧模块的综合重要性，形状 ``(count,)``。

    参数
    ----
    group:     :class:`pruning_toolkit.graph.ChannelGroup`
    model:     被剪枝的模型（用于按名字取子模块）
    criterion: 重要性准则；``bn_scale`` 会优先采用组内归一化层的 ``|gamma|``
    grads:     ``{参数名: 梯度}``，``taylor`` 准则需要
    normalize: 是否对每个模块的分数按均值归一化后再相加（推荐开启）——
               同一个组里卷积权重和 BN 缩放因子的量纲差异极大
    aggregate: 多模块分数的聚合方式（``sum`` / ``mean`` / ``max``）
    """
    _check_grads = grads or {}

    def score_of(module_name: str) -> torch.Tensor:
        module = model.get_submodule(module_name)
        grad = None
        weight = getattr(module, "weight", None)
        if weight is not None:
            for key in ("%s.weight" % module_name, "%s" % module_name):
                if key in _check_grads:
                    grad = _check_grads[key]
                    break
        return channel_importance(
            module, criterion=criterion, grad=grad, generator=generator, reduce=reduce
        )

    # bn_scale：优先使用组内归一化层的 gamma
    if criterion == "bn_scale":
        norm_scores: List[torch.Tensor] = []
        for module_name in group.output_modules:
            module = model.get_submodule(module_name)
            if isinstance(module, NORM_TYPES) and getattr(module, "weight", None) is not None:
                norm_scores.append(module.weight.detach().abs().reshape(-1).float())
        if norm_scores:
            return torch.stack(norm_scores, dim=0).sum(dim=0)

    scores: List[torch.Tensor] = []
    for edit in group.output_edits:
        value = score_of(edit.module_name).reshape(-1).float()
        if value.numel() != group.count:
            # 深度卷积等情况下归一化层与卷积的通道数一致，这里只做防御性检查
            continue
        scores.append(_normalize(value) if normalize else value)
    if not scores:
        return torch.ones(group.count)

    stacked = torch.stack(scores, dim=0)
    if aggregate == "sum":
        return stacked.sum(dim=0)
    if aggregate == "mean":
        return stacked.mean(dim=0)
    if aggregate == "max":
        return stacked.max(dim=0).values
    raise KeyError("未知的聚合方式 %r，可选：sum / mean / max" % aggregate)


# ---------------------------------------------------------------------- 校准梯度


def collect_grads(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, Any], torch.Tensor],
    batches: Iterable[Any],
    max_batches: int = 8,
    device: Optional[torch.device] = None,
    accumulate: bool = True,
) -> Dict[str, torch.Tensor]:
    """在一小批校准数据上反传，收集参数梯度，供 ``taylor`` 准则使用。

    参数
    ----
    model:      待分析模型
    loss_fn:    ``loss_fn(model, batch) -> loss``，batch 由调用方定义（如 ``(x, y)``）
    batches:    可迭代的数据批次
    max_batches: 最多使用多少个批次（越多越稳，越慢）
    device:     执行设备；``None`` 表示沿用模型参数所在设备
    accumulate: True 时把多个批次的梯度相加，False 时只取最后一个批次

    返回
    ----
    ``{参数名: 梯度张量}``，张量已 ``detach`` 并复制到 CPU 之外的原设备上。
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - 无参数模型
            device = torch.device("cpu")

    mode = model.training
    model.train()
    grads: Dict[str, torch.Tensor] = {}
    try:
        model.zero_grad(set_to_none=True)
        for index, batch in enumerate(batches):
            if index >= max_batches:
                break
            if not accumulate:
                model.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grads[name] = param.grad.detach().clone()
    finally:
        model.zero_grad(set_to_none=True)
        model.train(mode)
    return grads
