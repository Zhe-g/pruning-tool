"""评估指标：参数量、稀疏度、FLOPs、推理延迟、精度。

设计原则是**说清楚每个数字的含义**，避免常见的误导：

* ``count_parameters`` 统计全部参数；非结构化剪枝后的「有效参数量」要用
  ``count_nonzero_parameters``（掩码置零不会减少稠密算子的计算量）；
* ``count_flops`` 统计的是乘加密集型算子（卷积/线性）的 FLOPs，1 次 MAC 记 2 FLOPs，
  这是业界通用口径，但**不等于**实际墙钟时间；
* 延迟必须实测（``benchmark_latency``），因为它同时受内存带宽、算子实现、
  稀疏支持等影响——剪枝后参数量下降但延迟不降是很常见的事。
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import report as report_mod
from .graph import DependencyGraph, UnsupportedGraphError
from .masks import MaskBook
from .modules import CONV_TYPES, LINEAR_TYPES

__all__ = [
    "count_parameters",
    "count_nonzero_parameters",
    "model_sparsity",
    "count_flops",
    "benchmark_latency",
    "evaluate",
    "module_table",
    "summarize",
]


# ---------------------------------------------------------------------- 参数量


def count_parameters(
    model: nn.Module, only_conv_linear: bool = False, trainable_only: bool = False
) -> int:
    """统计参数量。``only_conv_linear`` 只统计卷积与线性层（不含 BN 等）。"""
    total = 0
    for module in model.modules():
        if only_conv_linear and not isinstance(module, CONV_TYPES + LINEAR_TYPES):
            continue
        for param in module.parameters(recurse=False):
            if trainable_only and not param.requires_grad:
                continue
            total += param.numel()
    return int(total)


def count_nonzero_parameters(model: nn.Module, masks: Optional[MaskBook] = None) -> int:
    """统计**非零**参数个数。

    非结构化剪枝后这就是「有效参数量」；``masks`` 用于把掩码置零但尚未写回权重的
    情况也算进去。
    """
    parameters = dict(model.named_parameters())
    total = 0
    with torch.no_grad():
        for name, param in parameters.items():
            mask = masks.get(name) if masks is not None and name in masks else None
            values = param
            if mask is not None and mask.shape == param.shape:
                values = param * mask.to(device=param.device, dtype=param.dtype)
            total += int(torch.count_nonzero(values).item())
    return total


def model_sparsity(model: nn.Module, masks: Optional[MaskBook] = None) -> float:
    """整体稀疏度 = 1 - 非零参数 / 全部参数。"""
    total = count_parameters(model)
    if total == 0:
        return 0.0
    return 1.0 - count_nonzero_parameters(model, masks=masks) / float(total)


# ---------------------------------------------------------------------- FLOPs


def count_flops(
    model: nn.Module, example_inputs: Sequence[torch.Tensor]
) -> Optional[int]:
    """统计乘加密集型算子的 FLOPs（卷积/线性，1 MAC = 2 FLOPs）。

    依赖 ``torch.fx`` 追踪；无法追踪的模型返回 ``None``（而不是给一个错的数字）。
    """
    try:
        graph = DependencyGraph(model, example_inputs)
    except UnsupportedGraphError:
        return None
    return graph.flops()


def flops_breakdown(
    model: nn.Module, example_inputs: Sequence[torch.Tensor]
) -> Dict[str, int]:
    """逐模块 FLOPs 明细。无法追踪时返回空字典。"""
    try:
        graph = DependencyGraph(model, example_inputs)
    except UnsupportedGraphError:
        return {}
    _, breakdown = graph.flops(per_module=True)
    return dict(breakdown)


# ---------------------------------------------------------------------- 延迟


def benchmark_latency(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    warmup: int = 20,
    iters: int = 100,
    device: Optional[torch.device] = None,
    sync_cuda: bool = True,
) -> Dict[str, float]:
    """实测推理延迟（毫秒）。

    参数
    ----
    warmup:  预热次数（排除首次运行的初始化、线程池启动等开销）
    iters:   计时次数
    sync_cuda: GPU 上必须同步才能得到真实耗时

    返回
    ----
    ``{"mean_ms", "std_ms", "min_ms", "p50_ms", "p90_ms", "throughput"}``。

    关于读数：小模型在 CPU 上单次前向只有几十微秒，**均值容易被偶发抖动拉高**，
    因此汇报时建议用 ``p50_ms``（中位数）。剪枝后延迟不降（甚至略升）是完全正常的：
    稠密算子并不会因为权重里有零而变快，真正的加速要么来自结构化剪枝带来的形状变小，
    要么来自稀疏算子/稀疏库的支持。
    """
    target_device = device or next(model.parameters()).device
    inputs = tuple(tensor.to(target_device) for tensor in example_inputs)
    batch_size = int(inputs[0].shape[0]) if inputs and inputs[0].dim() > 0 else 1

    mode = model.training
    model.eval()

    def sync() -> None:
        if sync_cuda and target_device.type == "cuda":
            torch.cuda.synchronize()

    durations: List[float] = []
    try:
        with torch.no_grad():
            for _ in range(max(0, warmup)):
                model(*inputs)
            sync()
            for _ in range(max(1, iters)):
                started = time.perf_counter()
                model(*inputs)
                sync()
                durations.append((time.perf_counter() - started) * 1000.0)
    finally:
        model.train(mode)

    ordered = sorted(durations)
    mean_ms = statistics.fmean(durations)
    return {
        "mean_ms": mean_ms,
        "std_ms": statistics.pstdev(durations) if len(durations) > 1 else 0.0,
        "min_ms": ordered[0],
        "p50_ms": ordered[len(ordered) // 2],
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "throughput": batch_size / (mean_ms / 1000.0) if mean_ms > 0 else float("inf"),
    }


# ---------------------------------------------------------------------- 精度


def _split_batch(batch: Any) -> Tuple[Any, Any]:
    """把 ``(输入, 标签)`` 拆开，兼容输入是元组/列表的情况。"""
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(
        "批次必须是 (输入, 标签) 形式；若数据格式特殊，请自行实现评估函数并传给 eval_fn"
    )


def evaluate(
    model: nn.Module,
    batches: Iterable[Any],
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    """在给定批次上评估损失与准确率。

    不改变模型原有的 ``train/eval`` 状态，不累计梯度。
    """
    target_device = device or next(model.parameters()).device
    criterion = loss_fn or (lambda logits, target: torch.nn.functional.cross_entropy(logits, target))
    mode = model.training
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    batches_used = 0
    try:
        with torch.no_grad():
            for index, batch in enumerate(batches):
                if max_batches is not None and index >= max_batches:
                    break
                inputs, targets = _split_batch(batch)
                if isinstance(inputs, (list, tuple)):
                    inputs = [item.to(target_device) for item in inputs]
                    logits = model(*inputs)
                else:
                    logits = model(inputs.to(target_device))
                targets = targets.to(target_device)
                loss = criterion(logits, targets)
                total_loss += float(loss.item()) * int(targets.shape[0])
                total_samples += int(targets.shape[0])
                total_correct += int((logits.argmax(dim=1) == targets).sum().item())
                batches_used += 1
    finally:
        model.train(mode)

    if total_samples == 0:
        return {"loss": float("nan"), "acc": float("nan"), "samples": 0, "batches": 0}
    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / float(total_samples),
        "samples": total_samples,
        "batches": batches_used,
    }


# ---------------------------------------------------------------------- 汇总


def module_table(model: nn.Module, masks: Optional[MaskBook] = None) -> List[Dict[str, Any]]:
    """逐模块的参数与稀疏度明细。"""
    rows: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        own = list(module.named_parameters(recurse=False))
        if not own:
            continue
        total = sum(param.numel() for _, param in own)
        nonzero = 0
        with torch.no_grad():
            for leaf, param in own:
                full_name = "%s.%s" % (name, leaf) if name else leaf
                mask = masks.get(full_name) if masks is not None and full_name in masks else None
                values = param
                if mask is not None and mask.shape == param.shape:
                    values = param * mask.to(device=param.device, dtype=param.dtype)
                nonzero += int(torch.count_nonzero(values).item())
        weight = getattr(module, "weight", None)
        rows.append(
            {
                "name": name or "(root)",
                "type": type(module).__name__,
                "shape": tuple(weight.shape) if isinstance(weight, torch.Tensor) else "-",
                "params": total,
                "nonzero": nonzero,
                "sparsity": 1.0 - nonzero / total if total else 0.0,
            }
        )
    return rows


def summarize(
    model: nn.Module,
    example_inputs: Optional[Sequence[torch.Tensor]] = None,
    masks: Optional[MaskBook] = None,
    latency: Optional[Mapping[str, float]] = None,
) -> report_mod.Report:
    """生成模型体检报告（参数量、稀疏度、FLOPs、延迟、逐模块明细）。"""
    report = report_mod.Report("模型概览", model_name=type(model).__name__)
    total = count_parameters(model)
    nonzero = count_nonzero_parameters(model, masks=masks)
    overview: Dict[str, Any] = {
        "参数量": total,
        "非零参数": nonzero,
        "稀疏度": model_sparsity(model, masks=masks),
        "卷积/线性参数量": count_parameters(model, only_conv_linear=True),
        "可训练参数量": count_parameters(model, trainable_only=True),
        "模块数": sum(1 for _ in model.modules()),
    }
    if example_inputs is not None:
        flops = count_flops(model, example_inputs)
        overview["FLOPs（乘加，示例输入）"] = flops if flops is not None else "无法追踪"
    if latency:
        overview["延迟均值(ms)"] = latency.get("mean_ms")
        overview["延迟 P90(ms)"] = latency.get("p90_ms")
    report.add_kv("总览", overview)

    rows = [
        [row["name"], row["type"], row["shape"], row["params"], row["nonzero"], row["sparsity"]]
        for row in module_table(model, masks=masks)
    ]
    report.add_table("逐模块明细", ["模块", "类型", "权重形状", "参数量", "非零", "稀疏度"], rows)
    if example_inputs is not None:
        breakdown = flops_breakdown(model, example_inputs)
        if breakdown:
            top = sorted(breakdown.items(), key=lambda item: item[1], reverse=True)[:15]
            report.add_table(
                "FLOPs 占比最高的模块",
                ["模块", "FLOPs"],
                [[name, value] for name, value in top],
            )
    return report
