"""敏感度分析与逐层稀疏度分配。

「每一层都剪掉同样比例」几乎总是次优的：敏感层被剪坏、冗余层剪得不够。
本模块回答两个问题：

1. **哪一层对剪枝更敏感？** —— :func:`unstructured_sensitivity` /
   :func:`structured_sensitivity` 扫描一系列剪枝强度，给出每层的响应曲线。
   默认使用**权重重建相对误差**作为代理指标（快、无需数据、结果稳定）；
   如果提供 ``eval_fn``，也可以扫描真实精度曲线（慢，但更贴近任务）。
2. **给定总稀疏度，各层应该各剪多少？** —— :func:`recommend_sparsity` 按「敏感度
   越低剪得越多」用指数权重分配，并用二分法求解缩放系数，使**参数量加权的平均
   稀疏度精确等于目标值**。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import report as report_mod
from .criteria import CRITERIA, elementwise_importance, group_importance
from .graph import DependencyGraph
from .modules import iter_weight_parameters

__all__ = [
    "SensitivityPoint",
    "ModuleSensitivity",
    "SensitivityReport",
    "SparsityAllocation",
    "unstructured_sensitivity",
    "structured_sensitivity",
    "recommend_sparsity",
    "reconstruction_error_curve",
]

DEFAULT_SPARSITIES: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _module_of(param_name: str) -> str:
    for suffix in (".weight", ".bias"):
        if param_name.endswith(suffix):
            return param_name[: -len(suffix)]
    return param_name


@dataclass
class SensitivityPoint:
    """某个剪枝强度下的一层响应。"""

    target: float
    error: float
    metric: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {"target": self.target, "error": self.error, "metric": self.metric}


@dataclass
class ModuleSensitivity:
    """单个模块（或通道组）的敏感度曲线。"""

    name: str
    kind: str
    params: int
    points: List[SensitivityPoint] = field(default_factory=list)

    def error_at(self, target: float) -> float:
        """取最近一个扫描点的重建误差。"""
        if not self.points:
            return 0.0
        nearest = min(self.points, key=lambda point: abs(point.target - target))
        return nearest.error

    def metric_at(self, target: float) -> Optional[float]:
        if not self.points:
            return None
        nearest = min(self.points, key=lambda point: abs(point.target - target))
        return nearest.metric

    @property
    def knee(self) -> float:
        """曲线「拐点」的粗略估计：误差相对首个扫描点增长最快的位置。"""
        if len(self.points) < 2:
            return self.points[0].target if self.points else 0.0
        best_target = self.points[0].target
        best_slope = -1.0
        previous = self.points[0]
        for point in self.points[1:]:
            span = max(point.target - previous.target, 1e-9)
            slope = (point.error - previous.error) / span
            if slope > best_slope:
                best_slope = slope
                best_target = point.target
            previous = point
        return best_target

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "params": self.params,
            "points": [point.to_dict() for point in self.points],
        }


@dataclass
class SensitivityReport:
    """敏感度扫描结果。"""

    mode: str
    criterion: str
    sparsities: List[float]
    modules: List[ModuleSensitivity] = field(default_factory=list)
    metric_name: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def names(self) -> List[str]:
        return [module.name for module in self.modules]

    def get(self, name: str) -> ModuleSensitivity:
        for module in self.modules:
            if module.name == name:
                return module
        raise KeyError("敏感度结果中没有 %r" % name)

    def importance(self, reference: float = 0.5) -> Dict[str, float]:
        """把敏感度转成「重要性」：同一剪枝强度下重建误差越大越重要。"""
        return {module.name: module.error_at(reference) for module in self.modules}

    def rank(self, reference: float = 0.5) -> List[Tuple[str, float]]:
        """按重要性从高到低排序。"""
        return sorted(self.importance(reference).items(), key=lambda item: item[1], reverse=True)

    def to_report(self, title: str = "敏感度分析报告") -> report_mod.Report:
        report = report_mod.Report(title)
        report.add_kv(
            "配置",
            {
                "剪枝模式": "非结构化" if self.mode == "unstructured" else "结构化（通道组）",
                "重要性准则": self.criterion,
                "扫描强度": list(self.sparsities),
                "真实指标": self.metric_name or "未提供（仅使用权重重建误差代理）",
            },
        )
        columns = ["模块", "类型", "参数量"] + ["误差@%.2f" % s for s in self.sparsities]
        rows = []
        for module in self.modules:
            curve = {round(point.target, 4): point.error for point in module.points}
            rows.append(
                [module.name, module.kind, module.params]
                + [curve.get(round(s, 4), float("nan")) for s in self.sparsities]
            )
        report.add_table("重建误差曲线", columns, rows,
                         note="误差 = 该层被剪掉的权重能量占该层总能量的比例（相对 Frobenius 误差）。")
        if self.metric_name:
            columns = ["模块", "类型"] + ["%s@%.2f" % (self.metric_name, s) for s in self.sparsities]
            rows = []
            for module in self.modules:
                curve = {round(point.target, 4): point.metric for point in module.points}
                rows.append([module.name, module.kind]
                            + [curve.get(round(s, 4)) for s in self.sparsities])
            report.add_table("真实指标曲线", columns, rows)
        report.add_table(
            "排序（敏感度越高越不该剪）",
            ["模块", "重建误差@参考点", "拐点估计"],
            [[module.name, module.error_at(0.5), module.knee] for module in
             sorted(self.modules, key=lambda m: m.error_at(0.5), reverse=True)],
        )
        for note in self.notes:
            report.add_text("说明", note)
        return report

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "criterion": self.criterion,
            "sparsities": list(self.sparsities),
            "metric_name": self.metric_name,
            "modules": [module.to_dict() for module in self.modules],
            "notes": list(self.notes),
        }

    def to_text(self) -> str:
        return self.to_report().to_text()

    def to_markdown(self) -> str:
        return self.to_report().to_markdown()


@dataclass
class SparsityAllocation:
    """逐层稀疏度分配结果。"""

    per_layer: Dict[str, float]
    requested: float
    achieved: float
    feasible: bool = True
    notes: List[str] = field(default_factory=list)

    def to_report(self, title: str = "逐层稀疏度分配") -> report_mod.Report:
        report = report_mod.Report(title)
        report.add_kv(
            "总览",
            {
                "目标稀疏度": self.requested,
                "实际稀疏度（参数量加权）": self.achieved,
                "是否可达": self.feasible,
                "层数": len(self.per_layer),
            },
        )
        report.add_table(
            "逐层目标",
            ["模块", "目标稀疏度"],
            [[name, value] for name, value in
             sorted(self.per_layer.items(), key=lambda item: item[1], reverse=True)],
            note="按目标稀疏度从高到低排列，越靠上的层被认为越冗余。",
        )
        for note in self.notes:
            report.add_text("说明", note)
        return report

    def to_dict(self) -> Dict[str, object]:
        return {
            "per_layer": dict(self.per_layer),
            "requested": self.requested,
            "achieved": self.achieved,
            "feasible": self.feasible,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------- 非结构化


def reconstruction_error_curve(
    weight: torch.Tensor,
    scores: torch.Tensor,
    sparsities: Sequence[float],
) -> List[float]:
    """保留分数最高的若干权重时，被丢弃权重能量占比（相对 Frobenius 误差）。

    实现上只排序一次再取前缀和，因此扫描任意多个强度都很便宜。
    """
    flat_weight = weight.detach().reshape(-1).float()
    flat_scores = scores.detach().reshape(-1).float()
    numel = flat_weight.numel()
    total_energy = float((flat_weight ** 2).sum().item())
    if numel == 0 or total_energy <= 0.0:
        return [0.0 for _ in sparsities]

    order = torch.argsort(flat_scores, descending=True, stable=True)
    energy_sorted = (flat_weight[order] ** 2).cumsum(dim=0)

    errors: List[float] = []
    for sparsity in sparsities:
        keep = int(round(numel * (1.0 - float(sparsity))))
        keep = max(0, min(keep, numel))
        kept_energy = float(energy_sorted[keep - 1].item()) if keep > 0 else 0.0
        removed = max(total_energy - kept_energy, 0.0)
        errors.append((removed / total_energy) ** 0.5)
    return errors


def unstructured_sensitivity(
    model: nn.Module,
    sparsities: Sequence[float] = DEFAULT_SPARSITIES,
    criterion: str = "l1",
    exclude: Iterable[str] = (),
    include_bias: bool = False,
    include_norm: bool = True,
    seed: int = 0,
) -> SensitivityReport:
    """逐参数的非结构化敏感度扫描（不修改模型）。"""
    if criterion not in CRITERIA:
        raise KeyError("未知的重要性准则 %r" % criterion)
    excluded = set(exclude)
    modules: List[ModuleSensitivity] = []
    generator = torch.Generator().manual_seed(seed)

    for name, param in iter_weight_parameters(
        model, include_bias=include_bias, include_norm=include_norm
    ):
        module_name = _module_of(name)
        if module_name in excluded or name in excluded:
            continue
        scores = elementwise_importance(
            param.data,
            criterion=criterion,
            generator=generator if criterion == "random" else None,
            name=name,
        )
        errors = reconstruction_error_curve(param.data, scores, sparsities)
        modules.append(
            ModuleSensitivity(
                name=module_name,
                kind=type(model.get_submodule(module_name)).__name__,
                params=int(param.numel()),
                points=[
                    SensitivityPoint(target=float(s), error=float(e))
                    for s, e in zip(sparsities, errors)
                ],
            )
        )
    notes = [
        "这里是逐参数的敏感度（每个 Linear/Conv 的权重单独扫描）。",
        "误差为相对 Frobenius 误差，衡量「按该强度剪掉最不重要的权重」造成的权重扰动，"
        "它不需要数据，但只反映权重本身，不反映任务损失。",
    ]
    return SensitivityReport(
        mode="unstructured",
        criterion=criterion,
        sparsities=[float(s) for s in sparsities],
        modules=modules,
        notes=notes,
    )


# ---------------------------------------------------------------------- 结构化


def _group_weight_energy(group, model: nn.Module, keep: torch.Tensor) -> Tuple[float, float]:
    """返回 ``(保留能量, 总能量)``：按通道统计通道组输出侧权重的平方和。"""
    kept_energy = 0.0
    total_energy = 0.0
    keep_set = keep.to(torch.long)
    for edit in group.output_edits:
        module = model.get_submodule(edit.module_name)
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        values = weight.detach().float()
        out_dim = 0
        if values.ndim >= 2 and values.shape[0] != group.count and values.shape[1] == group.count:
            out_dim = 1  # ConvTranspose 之类的布局
        if values.shape[out_dim] != group.count:
            values = values.reshape(-1)
            total_energy += float((values ** 2).sum().item())
            continue
        moved = values.movedim(out_dim, 0).reshape(group.count, -1)
        energy_per_channel = (moved ** 2).sum(dim=1)
        total_energy += float(energy_per_channel.sum().item())
        kept_energy += float(energy_per_channel[keep_set].sum().item())
    return kept_energy, total_energy


def structured_sensitivity(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    sparsities: Sequence[float] = DEFAULT_SPARSITIES,
    criterion: str = "l1",
    grads: Optional[Mapping[str, torch.Tensor]] = None,
    proxy: str = "reconstruction",
    eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    metric_name: str = "acc",
    exclude: Iterable[str] = (),
    protect_output: bool = True,
) -> SensitivityReport:
    """逐通道组的结构化敏感度扫描。

    参数
    ----
    proxy:     ``"reconstruction"``（默认，快：直接按权重能量估算，不复制模型）
               或 ``"accuracy"``（慢：每个扫描点深拷贝模型并跑 ``eval_fn``）
    eval_fn:   ``eval_fn(model) -> {"acc": ..., "loss": ...}``，``proxy="accuracy"`` 时必需
    """
    if proxy not in ("reconstruction", "accuracy"):
        raise ValueError("proxy 只能是 'reconstruction' 或 'accuracy'，实际 %r" % proxy)
    if proxy == "accuracy" and eval_fn is None:
        raise ValueError("proxy='accuracy' 需要提供 eval_fn(model) -> 指标字典")

    graph = DependencyGraph(model, example_inputs, protect_output=protect_output,
                            ignore=set(exclude))
    modules: List[ModuleSensitivity] = []
    for group in graph.prunable_groups():
        scores = group_importance(group, model, criterion=criterion, grads=grads)
        order = torch.argsort(scores.float(), descending=True, stable=True)
        params = sum(
            sum(p.numel() for p in model.get_submodule(name).parameters(recurse=False))
            for name in group.module_names
        )
        points: List[SensitivityPoint] = []
        for sparsity in sparsities:
            keep = max(1, int(round(group.count * (1.0 - float(sparsity)))))
            keep_indices = order[:keep].sort().values
            if proxy == "reconstruction":
                kept_energy, total_energy = _group_weight_energy(group, model, keep_indices)
                error = 0.0
                if total_energy > 0:
                    error = max(0.0, 1.0 - kept_energy / total_energy) ** 0.5
                points.append(SensitivityPoint(target=float(sparsity), error=float(error)))
            else:
                candidate = copy.deepcopy(model)
                candidate_graph = DependencyGraph(
                    candidate, example_inputs, protect_output=protect_output, ignore=set(exclude)
                )
                target_group = candidate_graph.group_by_index(group.index)
                target_group.apply(candidate, keep_indices)
                metrics = eval_fn(candidate)  # type: ignore[misc]
                points.append(
                    SensitivityPoint(
                        target=float(sparsity),
                        error=0.0,
                        metric=float(metrics.get(metric_name, float("nan"))),
                    )
                )
        modules.append(
            ModuleSensitivity(
                name=group.display,
                kind="channel_group",
                params=int(params),
                points=points,
            )
        )

    notes = [
        "结构化敏感度以「通道组」为单位，同一个组内的卷积/BN/下游输入维会一起被裁剪。",
        "只有可裁剪的通道组才会出现在曲线里；被屏障保护的组请用 explain() 查看原因。",
    ]
    if proxy == "reconstruction":
        notes.append(
            "当前使用权重能量代理（快）；如需真实精度曲线，请传 proxy='accuracy' 与 eval_fn。"
        )
    return SensitivityReport(
        mode="structured",
        criterion=criterion,
        sparsities=[float(s) for s in sparsities],
        modules=modules,
        metric_name=metric_name if proxy == "accuracy" else None,
        notes=notes,
    )


# ---------------------------------------------------------------------- 分配


def recommend_sparsity(
    model: nn.Module,
    target_sparsity: float,
    criterion: str = "l1",
    alpha: float = 1.0,
    cap: float = 0.99,
    spread: float = 0.5,
    reference: float = 0.5,
    exclude: Iterable[str] = (),
    sensitivity: Optional[SensitivityReport] = None,
    sparsities: Sequence[float] = DEFAULT_SPARSITIES,
    min_sparsity: Optional[float] = None,
    max_sparsity: Optional[float] = None,
) -> SparsityAllocation:
    """按敏感度分配逐层稀疏度，使参数量加权的整体稀疏度命中 ``target_sparsity``。

    分配形式为 ``s_l = clip(λ · exp(-α · c_l), lower, upper)``，其中 ``c_l`` 是该层在
    ``reference`` 强度下的重建误差（经 min-max 归一化），``λ`` 用二分法求解。

    参数
    ----
    alpha:   越大则「按敏感度区分」的力度越强；``alpha=0`` 退化为均匀分配
    cap:     单层稀疏度硬上限（0.99 表示至少要留下 1% 的权重）
    spread:  **单层相对目标的偏离幅度**，默认 0.5 表示每层落在
             ``[target·0.5, min(target·1.5, cap)]`` 区间内。
             这个约束很重要：目标是「参数量加权平均」，若不限制偏离，参数量很小的层
             （例如第一层卷积）会被推到 ``cap`` 才能把平均值凑够，结果把输入层剪废，
             这不是敏感度分析的本意，而是加权口径的副作用。
    min_sparsity/max_sparsity: 直接指定单层下界/上界，优先于 ``spread``
    """
    if not 0.0 < target_sparsity <= cap:
        raise ValueError("target_sparsity 需在 (0, %s] 内，实际 %r" % (cap, target_sparsity))
    if spread < 0.0:
        raise ValueError("spread 不能为负")

    lower = target_sparsity * (1.0 - spread) if min_sparsity is None else float(min_sparsity)
    upper = (
        min(target_sparsity * (1.0 + spread), cap) if max_sparsity is None else float(max_sparsity)
    )
    upper = min(upper, cap)
    lower = max(lower, 0.0)
    if lower > upper:
        raise ValueError(
            "单层下界 %.4f 大于上界 %.4f：目标 %.4f 与 spread=%.2f 不自洽"
            % (lower, upper, target_sparsity, spread)
        )
    if not lower <= target_sparsity <= upper:
        raise ValueError(
            "目标 %.4f 必须落在单层区间 [%.4f, %.4f] 内" % (target_sparsity, lower, upper)
        )

    report = sensitivity or unstructured_sensitivity(
        model, sparsities=sparsities, criterion=criterion, exclude=exclude
    )
    if not report.modules:
        raise ValueError("没有可用于分配的可剪层")

    names = [module.name for module in report.modules]
    sizes = {module.name: max(module.params, 1) for module in report.modules}
    raw = {module.name: max(module.error_at(reference), 0.0) for module in report.modules}
    lowest, highest = min(raw.values()), max(raw.values())
    if highest - lowest < 1e-12:
        capacities = {name: 0.5 for name in names}
    else:
        capacities = {name: (raw[name] - lowest) / (highest - lowest) for name in names}

    weights = {name: pow(2.718281828459045, -alpha * capacities[name]) for name in names}
    total_params = float(sum(sizes.values()))

    def achieved_for(scale: float) -> float:
        removed = 0.0
        for name in names:
            value = min(max(scale * weights[name], lower), upper)
            removed += value * sizes[name]
        return removed / total_params

    # 上界要大到让所有层都顶到 upper，二分区间才一定把目标夹住
    min_weight = max(min(weights.values()), 1e-12)
    low, high = 0.0, max(1.0, upper / min_weight)
    for _ in range(200):
        middle = (low + high) / 2.0
        if achieved_for(middle) < target_sparsity:
            low = middle
        else:
            high = middle
    scale = (low + high) / 2.0

    per_layer = {
        name: float(min(max(scale * weights[name], lower), upper)) for name in names
    }
    achieved = achieved_for(scale)
    feasible = abs(achieved - target_sparsity) < 1e-6
    notes: List[str] = []
    if not feasible:
        notes.append(
            "在单层上限 %.2f 的约束下无法精确命中目标稀疏度，已返回最接近的可行分配。"
            % cap
        )
    notes.append(
        "分配依据是敏感度扫描结果（参考强度 %.2f）：重建误差越大的层分到越低的稀疏度。"
        % reference
    )
    notes.append(
        "单层稀疏度被限制在 [%.4f, %.4f]（spread=%.2f，cap=%.2f）：参数量加权平均会让"
        "小层被推到极端值，这个约束就是用来防止把某一层剪废的。"
        % (lower, upper, spread, cap)
    )
    return SparsityAllocation(
        per_layer=per_layer,
        requested=float(target_sparsity),
        achieved=float(achieved),
        feasible=feasible,
        notes=notes,
    )
