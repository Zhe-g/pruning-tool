"""结构化剪枝：物理删除通道/神经元，真正降低参数量与 FLOPs。

工作方式
--------
1. 用 :class:`~pruning_toolkit.graph.DependencyGraph` 找出所有**可安全裁剪**的通道组；
2. 用重要性准则给组内每个通道打分（:func:`~pruning_toolkit.criteria.group_importance`）；
3. 选出要保留的通道索引 ``keep``，对组内所有模块**用同一组索引**裁剪参数
   （卷积权重、BN 的 ``gamma/beta/running_mean/running_var``、下游层的输入维……）；
4. 立刻跑一次前向，校验输出形状与原始一致且数值有限——这是「没有把模型剪坏」的硬证据。

裁剪范围
--------
* ``scope="local"``：每个通道组按同一比例裁剪，行为可预测（默认）；
* ``scope="global"``：先对每个组的分数做均值归一化，再全局统一排序，
  按总通道数计算保留量，让「不重要的层多剪、重要的层少剪」；
* ``per_group``：按组指定比例，通常来自敏感度分析的结果。

⚠️ 结构化剪枝是**破坏性**的：模型结构被就地改变，无法简单地「还原成原模型」。
需要对比实验时请先 ``copy.deepcopy(model)``；本工具会为每个通道组保留原始通道数，
但重建原始权重需要模型自身的检查点。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from . import report as report_mod
from .criteria import CRITERIA, criterion_requires_grad, group_importance
from .graph import ChannelGroup, DependencyGraph, UnsupportedGraphError

__all__ = [
    "GroupPruneStat",
    "StructuredReport",
    "StructuredPruner",
    "prune_structured",
]


def _count_parameters(model: nn.Module, only_conv_linear: bool = False) -> int:
    if only_conv_linear:
        types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
        return int(
            sum(
                param.numel()
                for module in model.modules()
                if isinstance(module, types)
                for param in module.parameters(recurse=False)
            )
        )
    return int(sum(param.numel() for param in model.parameters()))


@dataclass
class GroupPruneStat:
    """单个通道组的裁剪统计。"""

    group_index: int
    modules: List[str]
    channels_before: int
    channels_after: int
    ratio: float
    kept_indices: List[int]
    criterion: str
    clamped: bool = False

    @property
    def removed(self) -> int:
        return self.channels_before - self.channels_after

    def to_row(self) -> List[object]:
        shown = self.kept_indices[:8]
        suffix = "…" if len(self.kept_indices) > 8 else ""
        return [
            self.group_index,
            "+".join(self.modules),
            self.channels_before,
            self.channels_after,
            self.ratio,
            "%s%s" % (shown, suffix),
        ]

    def to_dict(self) -> Dict[str, object]:
        return {
            "group_index": self.group_index,
            "modules": list(self.modules),
            "channels_before": self.channels_before,
            "channels_after": self.channels_after,
            "removed": self.removed,
            "ratio": self.ratio,
            "kept_indices": list(self.kept_indices),
            "criterion": self.criterion,
            "clamped": self.clamped,
        }


@dataclass
class StructuredReport:
    """结构化剪枝结果。"""

    criterion: str
    scope: str
    target_ratio: float
    groups: List[GroupPruneStat] = field(default_factory=list)
    skipped: List[Tuple[int, str, str]] = field(default_factory=list)
    channels_before: int = 0
    channels_after: int = 0
    params_before: int = 0
    params_after: int = 0
    conv_linear_params_before: int = 0
    conv_linear_params_after: int = 0
    flops_before: int = 0
    flops_after: Optional[int] = None
    duration_s: float = 0.0
    verified: bool = False
    output_shape: Optional[Tuple[int, ...]] = None
    notes: List[str] = field(default_factory=list)

    @property
    def param_reduction(self) -> float:
        if not self.params_before:
            return 0.0
        return 1.0 - self.params_after / float(self.params_before)

    @property
    def flops_reduction(self) -> float:
        if not self.flops_before or not self.flops_after:
            return 0.0
        return 1.0 - self.flops_after / float(self.flops_before)

    @property
    def channel_reduction(self) -> float:
        if not self.channels_before:
            return 0.0
        return 1.0 - self.channels_after / float(self.channels_before)

    def to_report(self, title: str = "结构化剪枝报告") -> report_mod.Report:
        report = report_mod.Report(title)
        report.add_kv(
            "配置",
            {
                "重要性准则": self.criterion,
                "裁剪范围": "全局（组间统一排序）" if self.scope == "global" else "逐组等比例",
                "目标裁剪比例": self.target_ratio,
                "耗时(秒)": round(self.duration_s, 3),
                "前向校验": "通过" if self.verified else "未执行",
            },
        )
        report.add_kv(
            "收益",
            {
                "参数量（裁剪前）": self.params_before,
                "参数量（裁剪后）": self.params_after,
                "参数下降比例": self.param_reduction,
                "卷积/线性参数（前）": self.conv_linear_params_before,
                "卷积/线性参数（后）": self.conv_linear_params_after,
                "FLOPs（裁剪前）": self.flops_before,
                "FLOPs（裁剪后）": self.flops_after,
                "FLOPs 下降比例": self.flops_reduction,
                "通道总量（前）": self.channels_before,
                "通道总量（后）": self.channels_after,
            },
            note="FLOPs 为乘加密集型算子（卷积/线性）的统计值，按示例输入计算。",
        )
        report.add_table(
            "逐通道组结果",
            ["组", "涉及模块", "剪前通道", "剪后通道", "裁剪比例", "保留索引"],
            [group.to_row() for group in self.groups],
        )
        if self.skipped:
            report.add_table(
                "跳过的通道组",
                ["组", "涉及模块", "原因"],
                [[index, modules, reason] for index, modules, reason in self.skipped],
                note="跳过的组保持原样，通常是因为位于拼接等无法对位传播的结构之后。",
            )
        for note in self.notes:
            report.add_text("说明", note)
        return report

    def to_dict(self) -> Dict[str, object]:
        return {
            "criterion": self.criterion,
            "scope": self.scope,
            "target_ratio": self.target_ratio,
            "groups": [group.to_dict() for group in self.groups],
            "skipped": [
                {"group_index": index, "modules": modules, "reason": reason}
                for index, modules, reason in self.skipped
            ],
            "channels_before": self.channels_before,
            "channels_after": self.channels_after,
            "params_before": self.params_before,
            "params_after": self.params_after,
            "conv_linear_params_before": self.conv_linear_params_before,
            "conv_linear_params_after": self.conv_linear_params_after,
            "flops_before": self.flops_before,
            "flops_after": self.flops_after,
            "param_reduction": self.param_reduction,
            "flops_reduction": self.flops_reduction,
            "channel_reduction": self.channel_reduction,
            "verified": self.verified,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "duration_s": self.duration_s,
            "notes": list(self.notes),
        }

    def to_text(self) -> str:
        return self.to_report().to_text()

    def to_markdown(self) -> str:
        return self.to_report().to_markdown()


class StructuredPruner:
    """基于通道依赖图的结构化剪枝器。

    参数
    ----
    model:           待剪模型（会被原地修改）
    example_inputs:  示例输入（与 ``forward`` 位置参数一致），用于追踪与形状校验
    criterion:       重要性准则，见 :data:`pruning_toolkit.criteria.CRITERIA`
    scope:           ``local`` 或 ``global``
    exclude:         排除在剪枝之外的模块名
    protect_output:  保护直接产生模型输出的通道组（分类头）
    grads:           ``{参数名: 梯度}``，``taylor`` 准则需要
    normalize:       组内多模块（如卷积 + BN）分数是否按均值归一化后相加
    seed:            随机准则种子
    """

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Sequence[torch.Tensor],
        criterion: str = "l1",
        scope: str = "local",
        exclude: Iterable[str] = (),
        protect_output: bool = True,
        grads: Optional[Mapping[str, torch.Tensor]] = None,
        normalize: bool = True,
        aggregate: str = "sum",
        seed: int = 0,
    ) -> None:
        if criterion not in CRITERIA:
            raise KeyError("未知的重要性准则 %r，可选：%s" % (criterion, sorted(CRITERIA)))
        if scope not in ("local", "global"):
            raise ValueError("scope 只能是 'local' 或 'global'，实际 %r" % scope)
        if criterion_requires_grad(criterion) and not grads:
            raise ValueError(
                "准则 %r 需要梯度：先用 pruning_toolkit.collect_grads() 得到梯度字典再传入"
                % criterion
            )
        self.model = model
        self.example_inputs = tuple(example_inputs)
        self.criterion = criterion
        self.scope = scope
        self.exclude = set(exclude)
        self.protect_output = protect_output
        self.grads = dict(grads or {})
        self.normalize = normalize
        self.aggregate = aggregate
        self.seed = int(seed)
        self._graph: Optional[DependencyGraph] = None
        self.last_plan: Dict[int, torch.Tensor] = {}
        self.last_clamped: List[int] = []

    # ------------------------------------------------------------------ 依赖图
    def graph(self, rebuild: bool = False) -> DependencyGraph:
        """返回通道依赖图；模型结构变化后需要 ``rebuild=True``。"""
        if self._graph is None or rebuild:
            self._graph = DependencyGraph(
                self.model,
                self.example_inputs,
                protect_output=self.protect_output,
                ignore=self.exclude,
            )
        return self._graph

    def invalidate(self) -> None:
        self._graph = None

    def explain(self) -> report_mod.Report:
        """依赖图分析报告（哪些组能剪、哪些被屏障挡住）。"""
        return self.graph().summary()

    # ------------------------------------------------------------------ 打分
    def group_scores(self, group: ChannelGroup) -> torch.Tensor:
        generator = None
        if self.criterion == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + group.index)
        return group_importance(
            group,
            self.model,
            criterion=self.criterion,
            grads=self.grads,
            generator=generator,
            normalize=self.normalize,
            aggregate=self.aggregate,
        )

    # ------------------------------------------------------------------ 计划
    def plan(
        self, ratio: float, per_group: Optional[Mapping[Union[int, str], float]] = None
    ) -> Dict[int, torch.Tensor]:
        """计算每个通道组要保留的索引，返回 ``{组号: keep 索引张量}``（不修改模型）。"""
        if not 0.0 <= ratio < 1.0:
            raise ValueError("ratio 需在 [0, 1) 内，实际 %r" % ratio)
        graph = self.graph()
        groups = graph.prunable_groups()
        if not groups:
            raise RuntimeError(
                "没有可裁剪的通道组：模型可能全部由不支持的算子构成，或所有组都被屏障保护。"
                "可先调用 explain() 查看原因。"
            )

        overrides = self._normalize_overrides(per_group, groups)
        scores = {group.index: self.group_scores(group) for group in groups}
        keep_counts: Dict[int, int] = {}

        if self.scope == "global" and per_group is None:
            normalized = {}
            for group in groups:
                value = scores[group.index].float()
                normalized[group.index] = value / (value.mean() + 1e-12)
            flat = torch.cat([normalized[group.index] for group in groups])
            total = flat.numel()
            total_keep = int(round(total * (1.0 - ratio)))
            total_keep = max(len(groups), min(total_keep, total))  # 每组至少留 1 个通道
            selected = torch.zeros(total, dtype=torch.bool, device=flat.device)
            if total_keep < total:
                indices = torch.topk(flat, total_keep, largest=True, sorted=False).indices
                selected[indices] = True
            offset = 0
            for group in groups:
                size = group.count
                keep_counts[group.index] = int(selected[offset : offset + size].sum().item())
                offset += size
        else:
            for group in groups:
                group_ratio = overrides.get(group.index, ratio)
                keep = int(round(group.count * (1.0 - group_ratio)))
                keep_counts[group.index] = keep

        plan: Dict[int, torch.Tensor] = {}
        clamped: List[int] = []
        for group in groups:
            requested = keep_counts[group.index]
            keep = max(1, min(requested, group.count))
            if requested < 1:
                clamped.append(group.index)
            order = torch.argsort(scores[group.index].float(), descending=True, stable=True)
            plan[group.index] = order[:keep].sort().values.to(torch.long)
        self.last_plan = plan
        self.last_clamped = clamped
        return plan

    def _normalize_overrides(
        self,
        per_group: Optional[Mapping[Union[int, str], float]],
        groups: Sequence[ChannelGroup],
    ) -> Dict[int, float]:
        if not per_group:
            return {}
        by_name = {}
        for group in groups:
            for module_name in group.module_names:
                by_name[module_name] = group.index
            # 敏感度分析用 "conv+bn" 这样的显示名做键，这里一并接受
            by_name[group.display] = group.index
        resolved: Dict[int, float] = {}
        for key, value in per_group.items():
            if isinstance(key, int):
                resolved[key] = float(value)
                continue
            if key in by_name:
                resolved[by_name[key]] = float(value)
                continue
            raise KeyError("per_group 中的 %r 不是可裁剪通道组的模块名或组号" % (key,))
        for index, value in resolved.items():
            if not 0.0 <= value < 1.0:
                raise ValueError("组 %d 的裁剪比例需在 [0, 1) 内，实际 %r" % (index, value))
        return resolved

    # ------------------------------------------------------------------ 执行
    def prune(
        self,
        ratio: float,
        per_group: Optional[Mapping[Union[int, str], float]] = None,
        verify: bool = True,
    ) -> StructuredReport:
        """按比例裁剪通道组，并（默认）校验前向结果。"""
        started = time.time()
        graph = self.graph()
        expected_shape = self._forward_shape() if verify else None
        params_before = _count_parameters(self.model)
        conv_linear_before = _count_parameters(self.model, only_conv_linear=True)
        flops_before = graph.flops()

        plan = self.plan(ratio, per_group=per_group)
        stats: List[GroupPruneStat] = []
        for group in graph.prunable_groups():
            keep = plan[group.index]
            before = group.count
            after = int(keep.numel())
            if after == before:
                continue
            group.apply(self.model, keep)
            stats.append(
                GroupPruneStat(
                    group_index=group.index,
                    modules=group.module_names,
                    channels_before=before,
                    channels_after=after,
                    ratio=after / float(before),
                    kept_indices=[int(index) for index in keep.tolist()],
                    criterion=self.criterion,
                    clamped=group.index in self.last_clamped,
                )
            )

        skipped = [
            (group.index, group.module_names, group.reason)
            for group in graph.groups
            if not group.prunable
        ]

        prunable_groups = graph.prunable_groups()
        channels_before = int(sum(group.count for group in prunable_groups))
        channels_after = channels_before - int(
            sum(stat.channels_before - stat.channels_after for stat in stats)
        )

        self.invalidate()
        params_after = _count_parameters(self.model)
        conv_linear_after = _count_parameters(self.model, only_conv_linear=True)

        flops_after: Optional[int] = None
        verified = False
        output_shape: Optional[Tuple[int, ...]] = None
        notes: List[str] = [
            "结构化剪枝就地改变了模型结构，请重新构建依赖图（本剪枝器已自动失效缓存）。",
            "需要保留未剪枝基线时，请在剪枝前对模型做深拷贝。",
        ]
        # 先做前向校验：这一步直接暴露「形状被剪坏了」，错误信息最清楚
        if verify:
            output_shape = self._verify_forward(expected_shape)
            verified = True
        else:
            self._forward_shape()
        # 再重建依赖图统计剪枝后的 FLOPs；追踪失败只说明「量不出来」，不掩盖上面的校验结果
        try:
            flops_after = self.graph(rebuild=True).flops()
        except UnsupportedGraphError as exc:
            notes.append("剪枝后无法重新追踪模型，FLOPs 未统计：%s" % exc)

        if self.last_clamped:
            notes.append(
                "组 %s 的目标保留通道数小于 1，已自动收敛为至少保留 1 个通道。"
                % ", ".join(str(index) for index in self.last_clamped)
            )

        return StructuredReport(
            criterion=self.criterion,
            scope=self.scope,
            target_ratio=float(ratio),
            groups=stats,
            skipped=skipped,
            channels_before=channels_before,
            channels_after=channels_after,
            params_before=params_before,
            params_after=params_after,
            conv_linear_params_before=conv_linear_before,
            conv_linear_params_after=conv_linear_after,
            flops_before=flops_before,
            flops_after=flops_after,
            duration_s=time.time() - started,
            verified=verified,
            output_shape=output_shape,
            notes=notes,
        )

    # ------------------------------------------------------------------ 校验
    def _forward_shape(self) -> Tuple[int, ...]:
        mode = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(*self.example_inputs)
        finally:
            self.model.train(mode)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(output, dict):
            output = next(iter(output.values()))
        return tuple(int(dim) for dim in output.shape)

    def _verify_forward(self, expected: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
        """跑一次前向：形状必须与剪枝前一致，且数值有限。"""
        mode = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(*self.example_inputs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "结构化剪枝后前向失败：%s: %s。这说明通道传播有问题，"
                "请用 explain() 检查通道组，或把相关模块加入 exclude。" % (type(exc).__name__, exc)
            ) from exc
        finally:
            self.model.train(mode)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(output, dict):
            output = next(iter(output.values()))
        shape = tuple(int(dim) for dim in output.shape)
        if expected is not None and shape != expected:
            raise RuntimeError(
                "结构化剪枝后输出形状从 %s 变成了 %s，通道传播不一致。"
                "请用 explain() 检查依赖图，或调整 exclude。"
                "如果你**故意**剪掉了分类头的类别维（protect_output=False），"
                "请改用 prune(..., verify=False)。" % (expected, shape)
            )
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("结构化剪枝后前向输出包含 NaN/Inf，剪枝破坏了模型数值。")
        return shape

    def __repr__(self) -> str:  # pragma: no cover
        count = len(self._graph.prunable_groups()) if self._graph is not None else -1
        return "StructuredPruner(criterion=%s, scope=%s, prunable_groups=%s)" % (
            self.criterion,
            self.scope,
            count,
        )


def prune_structured(
    model: nn.Module, ratio: float, example_inputs: Sequence[torch.Tensor], **kwargs
) -> Tuple[nn.Module, StructuredReport, StructuredPruner]:
    """一站式接口：结构化剪枝并返回 ``(model, 报告, 剪枝器)``。"""
    pruner = StructuredPruner(model, example_inputs, **kwargs)
    report = pruner.prune(ratio)
    return model, report, pruner
