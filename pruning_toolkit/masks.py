"""非结构化剪枝的掩码管理。

非结构化剪枝把「不重要的单个权重」置零，权重张量形状不变，因此必须额外保存一张
**掩码**。掩码有两个用途：

1. **微调阶段**：把掩码压回权重（:meth:`MaskBook.apply`），并屏蔽被剪权重的梯度
   （:meth:`MaskBook.install_hooks`），这样才能做到「只训练没被剪的部分」；
2. **导出阶段**：把掩码一并保存，部署时才能知道哪些权重是真正有效的，
   也才能统计真实的有效参数量。

⚠️ 注意：掩码置零**不会**减少稠密算子的计算量（FLOPs 不变），它的收益在于有效参数量、
显存占用与后续稀疏核优化；想要真正降低计算量请用结构化剪枝。
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from . import report as report_mod

__all__ = ["MaskBook", "apply_masks"]


class MaskBook:
    """以参数名为键的掩码集合。"""

    def __init__(self, masks: Optional[Mapping[str, torch.Tensor]] = None) -> None:
        self._masks: Dict[str, torch.Tensor] = {}
        self._float_cache: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        for name, mask in (masks or {}).items():
            self.add(name, mask)

    # ------------------------------------------------------------------ 容器
    def add(self, name: str, mask: torch.Tensor) -> torch.Tensor:
        """登记一个掩码。统一转成 CPU 上的 ``bool`` 张量保存，使用时再按需搬运。"""
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        self._masks[name] = mask.detach().to("cpu").to(torch.bool).clone()
        self._float_cache.clear()
        return self._masks[name]

    def __contains__(self, name: object) -> bool:
        return name in self._masks

    def __len__(self) -> int:
        return len(self._masks)

    def __iter__(self) -> Iterator[str]:
        return iter(self._masks)

    def keys(self) -> List[str]:
        return list(self._masks)

    def items(self):
        return list(self._masks.items())

    def values(self) -> List[torch.Tensor]:
        return list(self._masks.values())

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self._masks.get(name)

    # ------------------------------------------------------------------ 统计
    def total_numel(self) -> int:
        return int(sum(mask.numel() for mask in self._masks.values()))

    def kept_numel(self) -> int:
        return int(sum(int(mask.sum().item()) for mask in self._masks.values()))

    def density(self) -> float:
        """掩码保留比例（1 表示未剪枝）。"""
        total = self.total_numel()
        if total == 0:
            return 1.0
        return self.kept_numel() / float(total)

    def sparsity(self) -> float:
        return 1.0 - self.density()

    # ------------------------------------------------------------------ 应用
    def _float_mask(self, name: str, param: torch.Tensor) -> torch.Tensor:
        key = (name, str(param.device), str(param.dtype))
        cached = self._float_cache.get(key)
        if cached is None:
            cached = self._masks[name].to(device=param.device, dtype=param.dtype)
            self._float_cache[key] = cached
        return cached

    def apply(self, model: nn.Module, strict: bool = False) -> int:
        """把掩码乘回权重（原地）。每个优化器 step 之后都应调用一次。"""
        parameters = dict(model.named_parameters())
        applied = 0
        for name, mask in self._masks.items():
            param = parameters.get(name)
            if param is None:
                if strict:
                    raise KeyError("模型中没有名为 %r 的参数" % name)
                continue
            if param.shape != mask.shape:
                if strict:
                    raise ValueError(
                        "参数 %r 形状 %s 与掩码形状 %s 不一致"
                        % (name, tuple(param.shape), tuple(mask.shape))
                    )
                continue
            with torch.no_grad():
                param.data.mul_(self._float_mask(name, param))
            applied += 1
        return applied

    # ------------------------------------------------------------------ 梯度屏蔽
    def install_hooks(self, model: nn.Module, strict: bool = False) -> int:
        """给被剪参数挂上梯度钩子，让被剪位置的梯度恒为 0。

        这是「掩码感知微调」的性能优化；正确性仍依赖每个 step 后调用 :meth:`apply`
        （因为 Adam 等优化器的解耦权重衰减不受梯度钩子约束）。
        """
        self.remove_hooks()
        parameters = dict(model.named_parameters())
        for name, mask in self._masks.items():
            param = parameters.get(name)
            if param is None or param.shape != mask.shape:
                if strict:
                    raise KeyError("无法为参数 %r 安装掩码钩子" % name)
                continue
            handle = param.register_hook(self._make_hook(name))
            self._handles.append(handle)
        return len(self._handles)

    def _make_hook(self, name: str):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * self._float_mask(name, grad)

        return hook

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------ 序列化
    def to(self, device) -> "MaskBook":
        """把掩码移动到指定设备（返回自身，便于链式调用）。"""
        self._masks = {name: mask.to(device) for name, mask in self._masks.items()}
        self._float_cache.clear()
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: mask.clone() for name, mask in self._masks.items()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> "MaskBook":
        return cls(state)

    def clone(self) -> "MaskBook":
        return MaskBook(self._masks)

    def save(self, path: str) -> str:
        target = os.fspath(path)
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {"format": "pruning-toolkit.masks", "version": 1, "masks": self.state_dict()}, target
        )
        return target

    @classmethod
    def load(cls, path: str) -> "MaskBook":
        payload = torch.load(os.fspath(path), map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "masks" in payload:
            return cls(payload["masks"])
        if isinstance(payload, dict):  # 兼容直接保存的 {名字: 掩码}
            return cls(payload)
        raise ValueError("无法识别的掩码文件格式：%s" % path)

    # ------------------------------------------------------------------ 展示
    def summary(self) -> Dict[str, object]:
        return {
            "被剪参数个数": len(self._masks),
            "掩码元素总数": self.total_numel(),
            "保留元素数": self.kept_numel(),
            "稀疏度": self.sparsity(),
        }

    def to_report(self, title: str = "非结构化剪枝掩码", top_k: int = 20) -> report_mod.Report:
        report = report_mod.Report(title)
        report.add_kv("总览", self.summary())
        rows = []
        for name, mask in sorted(self._masks.items()):
            kept = int(mask.sum().item())
            rows.append(
                [name, tuple(mask.shape), mask.numel(), kept, 1.0 - kept / max(mask.numel(), 1)]
            )
        if len(rows) > top_k:
            rows = rows[:top_k]
        report.add_table("掩码明细", ["参数", "形状", "元素数", "保留数", "稀疏度"], rows)
        return report

    def __repr__(self) -> str:  # pragma: no cover
        return "MaskBook(params=%d, sparsity=%.4f)" % (len(self._masks), self.sparsity())


def apply_masks(model: nn.Module, masks: Optional[MaskBook], strict: bool = False) -> int:
    """便捷函数：把掩码压回权重；``masks`` 为 ``None`` 时不做任何事。"""
    if masks is None:
        return 0
    return masks.apply(model, strict=strict)
