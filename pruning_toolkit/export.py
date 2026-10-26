"""导出：把剪枝结果变成可以落盘、可以复现、可以部署的文件。

导出对象分三类，别混用：

* **检查点**（:func:`save_checkpoint`）—— ``state_dict`` + 剪枝元信息 + 掩码。
  适合继续训练/继续实验。注意：结构化剪枝改变了模型结构，重新加载时必须先构造
  **同样结构**的模型（本工具会写下每层的形状信息，便于核对）。
* **TorchScript**（:func:`export_torchscript`）—— 通过 ``torch.jit.trace`` 固化，
  部署友好，不需要原始模型类定义。
* **ONNX**（:func:`export_onnx`）—— 需要额外安装 ``onnx``；未安装时给出明确提示。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from . import report as report_mod
from .masks import MaskBook
from .metrics import count_parameters, count_nonzero_parameters, model_sparsity

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "export_torchscript",
    "export_onnx",
    "export_fx_graph_module",
    "save_reports",
    "model_meta",
]

FORMAT_NAME = "pruning-toolkit.checkpoint"
FORMAT_VERSION = 1


def model_meta(model: nn.Module, masks: Optional[MaskBook] = None) -> Dict[str, Any]:
    """收集模型的形状与规模元信息，用于后续核对结构是否一致。"""
    layers = []
    for name, module in model.named_modules():
        own = list(module.named_parameters(recurse=False))
        if not own:
            continue
        layers.append(
            {
                "name": name,
                "type": type(module).__name__,
                "parameters": {
                    leaf: list(param.shape) for leaf, param in own
                },
            }
        )
    return {
        "class": type(model).__name__,
        "params": count_parameters(model),
        "nonzero_params": count_nonzero_parameters(model, masks=masks),
        "sparsity": model_sparsity(model, masks=masks),
        "layers": layers,
    }


def _ensure_directory(path: str) -> str:
    directory = os.path.dirname(os.path.abspath(os.fspath(path)))
    if directory:
        os.makedirs(directory, exist_ok=True)
    return os.fspath(path)


def save_checkpoint(
    model: nn.Module,
    path: str,
    meta: Optional[Mapping[str, Any]] = None,
    masks: Optional[MaskBook] = None,
    history: Optional[Mapping[str, Any]] = None,
) -> str:
    """保存检查点：``state_dict`` + 剪枝元信息 + 掩码 + 训练历史。"""
    target = _ensure_directory(path)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "meta": dict(meta or {}),
        "model_meta": model_meta(model, masks=masks),
        "masks": masks.state_dict() if masks is not None else None,
        "history": dict(history) if history else None,
    }
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    strict: bool = True,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """读取检查点；给了 ``model`` 就顺便把权重加载进去。

    返回原始载荷字典，其中 ``masks`` 需要时可传给 :class:`MaskBook`。
    """
    payload = torch.load(os.fspath(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("不是有效的剪枝检查点：%s" % path)
    if model is not None:
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=strict)
        payload["load_result"] = {"missing": list(missing), "unexpected": list(unexpected)}
    return payload


def export_torchscript(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    path: str,
    optimize: bool = False,
) -> str:
    """用 ``torch.jit.trace`` 导出 TorchScript，便于脱离源码部署。"""
    target = _ensure_directory(path)
    mode = model.training
    device = next(model.parameters()).device
    inputs = tuple(tensor.to(device) for tensor in example_inputs)
    model.eval()
    try:
        with torch.no_grad():
            traced = torch.jit.trace(model, inputs, strict=False)
            if optimize:
                traced = torch.jit.optimize_for_inference(traced)
            traced.save(target)
    finally:
        model.train(mode)
    return target


def export_onnx(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    path: str,
    opset_version: int = 13,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_batch: bool = False,
) -> str:
    """导出 ONNX。需要先 ``pip install onnx``。"""
    try:
        import onnx  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "导出 ONNX 需要 onnx 包：pip install onnx。"
            "若只想部署，也可以改用 export_torchscript()。"
        ) from exc

    target = _ensure_directory(path)
    mode = model.training
    device = next(model.parameters()).device
    inputs = tuple(tensor.to(device) for tensor in example_inputs)
    dynamic_axes = None
    if dynamic_batch:
        names = list(input_names or ["input"])
        names += list(output_names or ["output"])
        dynamic_axes = {name: {0: "batch"} for name in names}
    model.eval()
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs,
                target,
                opset_version=opset_version,
                input_names=list(input_names or ["input"]),
                output_names=list(output_names or ["output"]),
                dynamic_axes=dynamic_axes,
            )
    finally:
        model.train(mode)
    return target


def export_fx_graph_module(
    model: nn.Module, example_inputs: Sequence[torch.Tensor], path: str
) -> str:
    """把模型固化成 ``torch.fx.GraphModule`` 并保存（自带权重，加载即可用）。

    这是结构化剪枝后**最省事**的部署方式：不需要原始模型类，也不需要重新构造
    缩小后的结构，加载出来的对象本身就是剪枝后的计算图。
    """
    import torch.fx as fx

    target = _ensure_directory(path)
    mode = model.training
    model.eval()
    try:
        graph_module = fx.symbolic_trace(model)
    finally:
        model.train(mode)
    torch.save(graph_module, target)
    return target


def save_reports(reports: Mapping[str, report_mod.Report], directory: str) -> Dict[str, str]:
    """把多个报告同时落成 JSON 与 Markdown。

    参数
    ----
    reports:   ``{文件名前缀: Report}``
    directory: 输出目录

    返回
    ----
    ``{文件名前缀: {"json": 路径, "markdown": 路径}}``
    """
    os.makedirs(os.fspath(directory), exist_ok=True)
    saved: Dict[str, str] = {}
    for name, report in reports.items():
        json_path = os.path.join(directory, "%s.json" % name)
        md_path = os.path.join(directory, "%s.md" % name)
        report.save(json_path)
        report.save(md_path)
        saved[name] = {"json": json_path, "markdown": md_path}
    return saved  # type: ignore[return-value]


def write_json(payload: Mapping[str, Any], path: str) -> str:
    """把任意可序列化字典写成 JSON（用于保存配置、指标等）。"""
    target = _ensure_directory(path)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return target
