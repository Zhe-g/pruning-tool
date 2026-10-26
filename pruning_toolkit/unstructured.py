"""非结构化剪枝：把不重要的单个权重置零，并生成可复用的掩码。

提供两种范围：

* ``scope="global"``（默认）——把所有待剪参数摊平后统一排序，**全局**取前 k 个保留。
  各层实际稀疏度不同，通常比逐层等比例稀疏度效果更好，是幅值剪枝的推荐做法；
* ``scope="local"`` —— 每个参数各自保留相同比例，行为更可预测；
* 也可以用 ``per_layer={"conv1": 0.5, ...}`` 指定逐层目标，配合敏感度分析使用。

保留个数是**精确**的（用 ``topk`` 取索引，而不是比较阈值），因此不存在并列值导致的
稀疏度偏差，结果可复现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from . import report as report_mod
from .criteria import (
    CRITERIA,
    criterion_requires_grad,
    elementwise_importance,
)
from .masks import MaskBook
from .modules import iter_weight_parameters

__all__ = [
    "LayerPruneStat",
    "UnstructuredReport",
    "UnstructuredPruner",
    "prune_unstructured",
]


def _module_of(param_name: str) -> str:
    """从参数名反推模块名：``features.0.weight`` -> ``features.0``。"""
    for suffix in (".weight", ".bias"):
        if param_name.endswith(suffix):
            return param_name[: -len(suffix)]
    return param_name


def _exact_topk_mask(importance: torch.Tensor, keep: int) -> torch.Tensor:
    """精确保留分数最大的 ``keep`` 个元素，返回与 ``importance`` 同形状的布尔掩码。"""
    flat = importance.reshape(-1)
    numel = flat.numel()
    keep = int(max(0, min(keep, numel)))
    if keep == 0:
        return torch.zeros_like(flat, dtype=torch.bool).reshape(importance.shape)
    if keep >= numel:
        return torch.ones_like(flat, dtype=torch.bool).reshape(importance.shape)
    indices = torch.topk(flat, keep, largest=True, sorted=False).indices
    mask = torch.zeros(numel, dtype=torch.bool, device=flat.device)
    mask[indices] = True
    return mask.reshape(importance.shape)


@dataclass
class LayerPruneStat:
    """单个参数的剪枝统计。"""

    param_name: str
    module_name: str
    shape: Tuple[int, ...]
    numel: int
    kept: int
    removed: int
    sparsity_before: float
    sparsity_after: float
    score_min_kept: float
    score_max_removed: float
    target_sparsity: float

    @property
    def threshold(self) -> float:
        """保留与丢弃之间的分界分数（保留侧的最小值）。"""
        return self.score_min_kept

    def to_row(self) -> List[object]:
        return [
            self.param_name,
            tuple(self.shape),
            self.numel,
            self.kept,
            self.sparsity_before,
            self.sparsity_after,
            self.threshold,
        ]

    def to_dict(self) -> Dict[str, object]:
        return {
            "param_name": self.param_name,
            "module_name": self.module_name,
            "shape": list(self.shape),
            "numel": self.numel,
            "kept": self.kept,
            "removed": self.removed,
            "sparsity_before": self.sparsity_before,
            "sparsity_after": self.sparsity_after,
            "score_min_kept": self.score_min_kept,
            "score_max_removed": self.score_max_removed,
            "target_sparsity": self.target_sparsity,
        }


@dataclass
class UnstructuredReport:
    """非结构化剪枝结果。"""

    criterion: str
    scope: str
    target_sparsity: float
    achieved_sparsity: float
    total_numel: int
    total_kept: int
    layers: List[LayerPruneStat] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)
    duration_s: float = 0.0
    notes: List[str] = field(default_factory=list)
    masks: Optional[MaskBook] = None

    @property
    def kept_ratio(self) -> float:
        return self.total_kept / float(self.total_numel) if self.total_numel else 1.0

    def to_report(self, title: str = "非结构化剪枝报告") -> report_mod.Report:
        report = report_mod.Report(title)
        report.add_kv(
            "配置",
            {
                "重要性准则": self.criterion,
                "剪枝范围": "全局（所有参数统一排序）" if self.scope == "global" else "逐层等比例",
                "目标稀疏度": self.target_sparsity,
                "实际稀疏度": self.achieved_sparsity,
                "耗时(秒)": round(self.duration_s, 3),
            },
        )
        report.add_kv(
            "参数量",
            {
                "剪枝前元素总数": self.total_numel,
                "保留元素数": self.total_kept,
                "置零元素数": self.total_numel - self.total_kept,
            },
            note="非结构化剪枝不改变稠密算子的 FLOPs，收益体现在有效参数量与稀疏存储上。",
        )
        report.add_table(
            "逐参数统计",
            ["参数", "形状", "元素数", "保留", "剪前稀疏度", "剪后稀疏度", "分界分数"],
            [layer.to_row() for layer in self.layers],
        )
        if self.excluded:
            report.add_text("已排除", ", ".join(self.excluded))
        for note in self.notes:
            report.add_text("说明", note)
        return report

    def to_dict(self) -> Dict[str, object]:
        return {
            "criterion": self.criterion,
            "scope": self.scope,
            "target_sparsity": self.target_sparsity,
            "achieved_sparsity": self.achieved_sparsity,
            "total_numel": self.total_numel,
            "total_kept": self.total_kept,
            "duration_s": self.duration_s,
            "excluded": list(self.excluded),
            "layers": [layer.to_dict() for layer in self.layers],
            "notes": list(self.notes),
        }

    def to_text(self) -> str:
        return self.to_report().to_text()

    def to_markdown(self) -> str:
        return self.to_report().to_markdown()


class UnstructuredPruner:
    """幅值/梯度类准则的非结构化剪枝器。

    参数
    ----
    model:        待剪模型（会被原地修改）
    criterion:    重要性准则，见 :data:`pruning_toolkit.criteria.CRITERIA`
    scope:        ``global`` 或 ``local``
    exclude:      需要排除的模块名（或参数名前缀）
    include:      仅处理这些模块（``None`` 表示全部）
    include_bias: 是否连偏置一起剪（默认不剪，偏置参数量通常很小）
    include_norm: 是否剪归一化层的仿射权重
    protect_first_last: 自动保护首个与最后一个可剪模块（避免破坏输入/分类头）
    grads:        ``{参数名: 梯度}``，``taylor`` 准则需要
    seed:         随机准则的随机种子
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: str = "l1",
        scope: str = "global",
        exclude: Iterable[str] = (),
        include: Optional[Iterable[str]] = None,
        include_bias: bool = False,
        include_norm: bool = True,
        protect_first_last: bool = False,
        grads: Optional[Mapping[str, torch.Tensor]] = None,
        seed: int = 0,
    ) -> None:
        if criterion not in CRITERIA:
            raise KeyError("未知的重要性准则 %r，可选：%s" % (criterion, sorted(CRITERIA)))
        if scope not in ("global", "local"):
            raise ValueError("scope 只能是 'global' 或 'local'，实际 %r" % scope)
        self.model = model
        self.criterion = criterion
        self.scope = scope
        self.grads = dict(grads or {})
        self.seed = int(seed)
        self.exclude = set(exclude)
        self.include = set(include) if include is not None else None
        self.protect_first_last = bool(protect_first_last)

        candidates = list(
            iter_weight_parameters(model, include_bias=include_bias, include_norm=include_norm)
        )
        if not candidates:
            raise ValueError("模型中没有找到可剪的权重参数")

        self._excluded: List[str] = []
        self.params: List[Tuple[str, torch.nn.Parameter]] = []
        for name, param in candidates:
            module_name = _module_of(name)
            if name in self.exclude or module_name in self.exclude:
                self._excluded.append(name)
                continue
            if any(name.startswith(prefix) for prefix in self.exclude):
                self._excluded.append(name)
                continue
            if self.include is not None and module_name not in self.include and name not in self.include:
                self._excluded.append(name)
                continue
            self.params.append((name, param))

        if self.protect_first_last and len(self.params) > 2:
            protected = {self.params[0][0], self.params[-1][0]}
            self._excluded.extend(sorted(protected))
            self.params = [(name, param) for name, param in self.params if name not in protected]

        if not self.params:
            raise ValueError("排除/筛选之后没有剩余的可剪参数")

        self._masks: Optional[MaskBook] = None
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------ 属性
    @property
    def masks(self) -> Optional[MaskBook]:
        return self._masks

    @property
    def param_names(self) -> List[str]:
        return [name for name, _ in self.params]

    def total_numel(self) -> int:
        return int(sum(param.numel() for _, param in self.params))

    def sparsity(self) -> float:
        """当前权重的实际稀疏度（含历史剪枝结果）。"""
        total = kept = 0
        with torch.no_grad():
            for _, param in self.params:
                total += param.numel()
                kept += int(torch.count_nonzero(param).item())
        return 1.0 - kept / float(total) if total else 0.0

    # ------------------------------------------------------------------ 内部
    def _ensure_grads(self) -> None:
        if criterion_requires_grad(self.criterion) and not self.grads:
            raise ValueError(
                "准则 %r 需要梯度：请用 pruning_toolkit.collect_grads() 在少量校准数据上"
                "反传得到梯度字典，再传给 UnstructuredPruner(grads=...)" % self.criterion
            )

    def _generator(self, param: torch.Tensor) -> Optional[torch.Generator]:
        if self.criterion != "random":
            return None
        generator = torch.Generator(device=param.device)
        generator.manual_seed(self.seed)
        return generator

    def importances(self) -> Dict[str, torch.Tensor]:
        """返回每个参数的逐元素重要性分数（不修改模型）。"""
        self._ensure_grads()
        result: Dict[str, torch.Tensor] = {}
        for name, param in self.params:
            result[name] = elementwise_importance(
                param.data,
                criterion=self.criterion,
                grad=self.grads.get(name),
                generator=self._generator(param),
                name=name,
            )
        return result

    def _module_targets(
        self, sparsity: float, per_layer: Optional[Mapping[str, float]]
    ) -> Dict[str, float]:
        thresholds = dict(per_layer or {})
        unknown = set(thresholds) - {_module_of(name) for name, _ in self.params}
        if unknown:
            raise KeyError(
                "per_layer 中指定的模块不在可剪参数里：%s" % ", ".join(sorted(unknown))
            )
        targets: Dict[str, float] = {}
        for name, _ in self.params:
            module_name = _module_of(name)
            value = thresholds.get(module_name, sparsity)
            targets[name] = float(min(max(value, 0.0), 1.0))
        return targets

    # ------------------------------------------------------------------ 核心
    def plan(
        self, sparsity: float, per_layer: Optional[Mapping[str, float]] = None
    ) -> MaskBook:
        """计算掩码但**不修改**模型。"""
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError("sparsity 需在 [0, 1] 内，实际 %r" % sparsity)
        importances = self.importances()
        targets = self._module_targets(sparsity, per_layer)
        masks = MaskBook()

        if self.scope == "global" and per_layer is None:
            # 全局：把所有参数的分数摊平后统一取前 k 个
            flat = torch.cat([importances[name].reshape(-1) for name, _ in self.params])
            numel = flat.numel()
            keep = int(round(numel * (1.0 - sparsity)))
            keep = int(max(0, min(keep, numel)))
            if keep == 0:
                global_mask = torch.zeros(numel, dtype=torch.bool, device=flat.device)
            elif keep >= numel:
                global_mask = torch.ones(numel, dtype=torch.bool, device=flat.device)
            else:
                indices = torch.topk(flat, keep, largest=True, sorted=False).indices
                global_mask = torch.zeros(numel, dtype=torch.bool, device=flat.device)
                global_mask[indices] = True
            offset = 0
            for name, param in self.params:
                size = param.numel()
                piece = global_mask[offset : offset + size].reshape(param.shape)
                masks.add(name, piece)
                offset += size
        else:
            for name, param in self.params:
                target = targets[name]
                keep = int(round(param.numel() * (1.0 - target)))
                masks.add(name, _exact_topk_mask(importances[name], keep))
        return masks

    def prune(
        self,
        sparsity: float,
        per_layer: Optional[Mapping[str, float]] = None,
        materialize: bool = True,
    ) -> UnstructuredReport:
        """计算掩码并（默认）立即把权重置零。

        可重复调用以实现迭代式剪枝：每次传入**累积**目标稀疏度即可，
        因为已被置零的权重得分为 0，不会被重新选中。
        """
        started = time.time()
        self.ensure_backup()
        importances = self.importances()
        targets = self._module_targets(sparsity, per_layer)
        masks = self.plan(sparsity, per_layer=per_layer)

        layers: List[LayerPruneStat] = []
        for name, param in self.params:
            mask = masks.get(name)
            assert mask is not None
            before = param.detach()
            sparsity_before = 1.0 - float(torch.count_nonzero(before).item()) / max(before.numel(), 1)
            kept = int(mask.sum().item())
            score = importances[name]
            kept_scores = score[mask]
            removed_scores = score[~mask]
            layers.append(
                LayerPruneStat(
                    param_name=name,
                    module_name=_module_of(name),
                    shape=tuple(param.shape),
                    numel=param.numel(),
                    kept=kept,
                    removed=param.numel() - kept,
                    sparsity_before=sparsity_before,
                    sparsity_after=1.0 - kept / float(max(param.numel(), 1)),
                    score_min_kept=float(kept_scores.min().item()) if kept_scores.numel() else 0.0,
                    score_max_removed=(
                        float(removed_scores.max().item()) if removed_scores.numel() else 0.0
                    ),
                    target_sparsity=targets[name],
                )
            )

        if materialize:
            masks.apply(self.model)
        self._masks = masks

        total_numel = self.total_numel()
        total_kept = masks.kept_numel()
        notes = [
            "非结构化剪枝只把权重置零，张量形状不变；密算子 FLOPs 不变，"
            "有效参数量与稀疏存储才是收益所在。",
            "微调时每个优化器 step 之后都要调用 masks.apply(model)，或安装梯度钩子，"
            "否则被剪权重会被优化器重新更新为非零。",
        ]
        return UnstructuredReport(
            criterion=self.criterion,
            scope=self.scope,
            target_sparsity=float(sparsity),
            achieved_sparsity=1.0 - total_kept / float(total_numel) if total_numel else 0.0,
            total_numel=total_numel,
            total_kept=total_kept,
            layers=layers,
            excluded=sorted(set(self._excluded)),
            duration_s=time.time() - started,
            notes=notes,
            masks=masks,
        )

    def materialize(self) -> int:
        """把当前掩码重新压回权重（微调循环中每个 step 后调用）。"""
        if self._masks is None:
            return 0
        return self._masks.apply(self.model)

    def install_hooks(self) -> int:
        if self._masks is None:
            return 0
        return self._masks.install_hooks(self.model)

    def remove_hooks(self) -> None:
        if self._masks is not None:
            self._masks.remove_hooks()

    # ------------------------------------------------------------------ 备份
    def ensure_backup(self) -> None:
        """保存剪枝前的原始权重，供 :meth:`restore` 回滚。只在首次调用时保存。"""
        if self._backup is None:
            with torch.no_grad():
                self._backup = {name: param.detach().clone() for name, param in self.params}

    def has_backup(self) -> bool:
        return self._backup is not None

    def restore(self) -> int:
        """回滚到首次剪枝前的权重，并清空掩码。"""
        if self._backup is None:
            return 0
        with torch.no_grad():
            for name, param in self.params:
                param.copy_(self._backup[name])
        self._masks = None
        return len(self._backup)

    def drop_backup(self) -> None:
        self._backup = None

    def __repr__(self) -> str:  # pragma: no cover
        return "UnstructuredPruner(criterion=%s, scope=%s, params=%d)" % (
            self.criterion,
            self.scope,
            len(self.params),
        )


def prune_unstructured(
    model: nn.Module, sparsity: float, **kwargs
) -> Tuple[nn.Module, UnstructuredReport, UnstructuredPruner]:
    """一站式接口：剪枝并返回 ``(model, 报告, 剪枝器)``。

    返回剪枝器是为了让调用方能拿到 ``pruner.masks`` 做掩码感知微调。
    """
    pruner = UnstructuredPruner(model, **kwargs)
    report = pruner.prune(sparsity)
    return model, report, pruner
