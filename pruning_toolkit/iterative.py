"""迭代式剪枝：把「一步剪到位」拆成「多轮剪枝 + 微调」。

为什么需要迭代
--------------
一次性剪掉 90% 的权重，模型几乎必然崩掉，而且**不可恢复**：权重被删掉的信息
无法从剩下的训练中补回来。分多轮、每轮剪一点再微调，能让网络逐轮适应新的稀疏
结构，最终在同样稀疏度下得到明显更好的精度——这就是 Iterative Magnitude
Pruning（IMP）的核心结论，也是本模块实现的对象。

掩码感知微调
------------
非结构化剪枝每一轮结束后必须把掩码重新压回权重（本模块自动完成），否则优化器会把
刚剪掉的权重又更新成非零值。需要更强保证时，可以在 ``finetune_fn`` 里配合
``pruner.masks.install_hooks()``。

结构化剪枝每轮都会改变模型结构，因此每一轮都要**重新构建通道依赖图**（本模块自动
完成），否则第二轮会拿到过期的通道数。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from . import report as report_mod
from .masks import MaskBook
from .metrics import count_parameters, count_nonzero_parameters, count_flops
from .schedules import SCHEDULES, cumulative_targets, incremental_ratios
from .structured import StructuredPruner
from .unstructured import UnstructuredPruner

__all__ = ["RoundContext", "RoundRecord", "IterativeReport", "IterativePruner", "iterative_prune"]

EvalFn = Callable[[nn.Module], Dict[str, float]]
FinetuneFn = Callable[[nn.Module, "RoundContext"], Optional[Dict[str, float]]]


@dataclass
class RoundContext:
    """传给 ``finetune_fn`` 的上下文。

    ``masks`` 是该轮剪枝后的掩码（结构化剪枝时为 ``None``）。微调时应当把它压回权重，
    例如::

        def finetune(model, ctx):
            hook = (lambda m: ctx.masks.apply(m)) if ctx.masks else None
            fit(model, train_loader, epochs=3, post_step_hook=hook)
            return {"acc": ...}
    """

    round_index: int
    target: float
    achieved: float
    masks: Optional[MaskBook] = None


@dataclass
class RoundRecord:
    """单轮迭代的统计。"""

    round_index: int
    target: float
    achieved: float
    params: int
    nonzero_params: int
    params_reduction: float
    flops: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metrics_after_prune: Dict[str, float] = field(default_factory=dict)
    finetune_metrics: Dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    note: str = ""

    def to_row(self) -> List[object]:
        row: List[object] = [
            self.round_index,
            self.target,
            self.achieved,
            self.params,
            self.params_reduction,
            self.flops if self.flops is not None else "-",
        ]
        row.append(self.metrics_after_prune.get("acc", "-"))
        row.append(self.metrics.get("acc", "-"))
        return row

    def to_dict(self) -> Dict[str, object]:
        return {
            "round": self.round_index,
            "target": self.target,
            "achieved": self.achieved,
            "params": self.params,
            "nonzero_params": self.nonzero_params,
            "params_reduction": self.params_reduction,
            "flops": self.flops,
            "metrics": dict(self.metrics),
            "metrics_after_prune": dict(self.metrics_after_prune),
            "finetune_metrics": dict(self.finetune_metrics),
            "duration_s": self.duration_s,
            "note": self.note,
        }


@dataclass
class IterativeReport:
    """迭代式剪枝的完整记录。"""

    mode: str
    criterion: str
    scope: str
    schedule: str
    rounds: int
    final_target: float
    records: List[RoundRecord] = field(default_factory=list)
    initial_metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    params_before: int = 0
    params_after: int = 0
    total_duration_s: float = 0.0
    notes: List[str] = field(default_factory=list)
    masks: Optional[MaskBook] = None

    @property
    def params_reduction(self) -> float:
        if not self.params_before:
            return 0.0
        return 1.0 - self.params_after / float(self.params_before)

    @property
    def accuracy_delta(self) -> Optional[float]:
        """最终指标相对初始指标的变化（只有提供了 ``eval_fn`` 才有值）。"""
        if "acc" in self.initial_metrics and "acc" in self.final_metrics:
            return self.final_metrics["acc"] - self.initial_metrics["acc"]
        return None

    def to_report(self, title: str = "迭代式剪枝报告") -> report_mod.Report:
        report = report_mod.Report(title)
        report.add_kv(
            "配置",
            {
                "剪枝模式": "非结构化" if self.mode == "unstructured" else "结构化",
                "重要性准则": self.criterion,
                "裁剪范围": self.scope,
                "稀疏度调度": self.schedule,
                "轮数": self.rounds,
                "最终目标": self.final_target,
                "总耗时(秒)": round(self.total_duration_s, 3),
            },
        )
        overview: Dict[str, object] = {
            "参数量（前）": self.params_before,
            "参数量（后）": self.params_after,
            "参数下降比例": self.params_reduction,
        }
        if self.initial_metrics:
            overview["初始指标"] = self.initial_metrics
        if self.final_metrics:
            overview["最终指标"] = self.final_metrics
        delta = self.accuracy_delta
        if delta is not None:
            overview["准确率变化"] = delta
        report.add_kv("总览", overview)
        report.add_table(
            "逐轮记录",
            ["轮", "目标", "实际", "参数量", "参数下降", "FLOPs", "剪枝后 acc", "微调后 acc"],
            [record.to_row() for record in self.records],
            note="「目标」与「实际」都是**累积**比例（非结构化=稀疏度，结构化=通道裁剪比例）。"
                 "「剪枝后 acc」是刚剪完、尚未微调时的精度，「微调后 acc」是本轮微调结束后的精度，"
                 "两者之差就是微调的收益。非结构化剪枝的稠密参数量与 FLOPs 不随掩码下降，"
                 "有效参数量请见总览中的非零参数。",
        )
        for note in self.notes:
            report.add_text("说明", note)
        return report

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "criterion": self.criterion,
            "scope": self.scope,
            "schedule": self.schedule,
            "rounds": self.rounds,
            "final_target": self.final_target,
            "records": [record.to_dict() for record in self.records],
            "initial_metrics": dict(self.initial_metrics),
            "final_metrics": dict(self.final_metrics),
            "params_before": self.params_before,
            "params_after": self.params_after,
            "params_reduction": self.params_reduction,
            "accuracy_delta": self.accuracy_delta,
            "total_duration_s": self.total_duration_s,
            "notes": list(self.notes),
        }

    def to_text(self) -> str:
        return self.to_report().to_text()

    def to_markdown(self) -> str:
        return self.to_report().to_markdown()


class IterativePruner:
    """多轮「剪枝 → 微调」循环。

    参数
    ----
    model:          待剪模型（原地修改）
    example_inputs: 示例输入；结构化模式**必需**（用于重建依赖图与统计 FLOPs）
    mode:           ``unstructured`` 或 ``structured``
    criterion:      重要性准则
    scope:          非结构化用 ``global``/``local``；结构化用 ``local``/``global``
    schedule:       稀疏度调度曲线，见 :data:`pruning_toolkit.schedules.SCHEDULES`
    rounds:         迭代轮数（1 表示一步剪到位）
    final_sparsity: 非结构化=最终稀疏度；结构化=最终通道裁剪比例
    per_layer:      逐层最终目标（累积值），非结构化按模块名、结构化按组内模块名
    eval_fn:        ``eval_fn(model) -> {"loss":..., "acc":...}``，用于逐轮记录指标
    finetune_fn:    ``finetune_fn(model, round_index) -> 指标字典``，在原地微调模型
    """

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Optional[Sequence[torch.Tensor]] = None,
        mode: str = "unstructured",
        criterion: str = "l1",
        scope: Optional[str] = None,
        schedule: str = "cubic",
        rounds: int = 4,
        final_sparsity: float = 0.9,
        per_layer: Optional[Mapping[str, float]] = None,
        exclude: Sequence[str] = (),
        seed: int = 0,
        eval_fn: Optional[EvalFn] = None,
        finetune_fn: Optional[FinetuneFn] = None,
        initial_sparsity: float = 0.0,
        protect_first_last: bool = False,
        protect_output: bool = True,
        grads: Optional[Mapping[str, torch.Tensor]] = None,
        verbose: bool = True,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        if mode not in ("unstructured", "structured"):
            raise ValueError("mode 只能是 'unstructured' 或 'structured'，实际 %r" % mode)
        if schedule not in SCHEDULES:
            raise KeyError("未知的调度曲线 %r，可选：%s" % (schedule, list(SCHEDULES)))
        if mode == "structured" and example_inputs is None:
            raise ValueError("结构化迭代剪枝必须提供 example_inputs（每轮都要重建依赖图）")

        self.model = model
        self.example_inputs = tuple(example_inputs) if example_inputs is not None else None
        self.mode = mode
        self.criterion = criterion
        self.scope = scope or ("global" if mode == "unstructured" else "local")
        self.schedule = schedule
        self.rounds = int(rounds)
        self.final_sparsity = float(final_sparsity)
        self.per_layer = dict(per_layer or {})
        self.exclude = list(exclude)
        self.seed = int(seed)
        self.eval_fn = eval_fn
        self.finetune_fn = finetune_fn
        self.initial_sparsity = float(initial_sparsity)
        self.protect_first_last = protect_first_last
        self.protect_output = protect_output
        self.grads = dict(grads or {})
        self.verbose = verbose
        self.log = log or print

        self._targets = cumulative_targets(
            self.final_sparsity, self.rounds, schedule, self.initial_sparsity
        )
        self._incremental = incremental_ratios(self._targets)

    # ------------------------------------------------------------------ 工具
    def _say(self, message: str) -> None:
        if self.verbose and self.log is not None:
            self.log(message)

    def _evaluate(self, model: nn.Module) -> Dict[str, float]:
        if self.eval_fn is None:
            return {}
        return dict(self.eval_fn(model))

    def _flops(self) -> Optional[int]:
        if self.example_inputs is None:
            return None
        return count_flops(self.model, self.example_inputs)

    def _per_layer_targets(self, round_index: int) -> Dict[str, float]:
        """把逐层「最终目标」按本轮的全局进度等比缩放成「累积目标」。

        第 ``round_index`` 轮的全局累积目标为 ``targets[r-1]``，因此每层的累积目标取
        ``final_layer_target * (targets[r-1] / final_sparsity)``；这样既保持单调，
        又保证最后一轮正好落到用户指定的逐层目标上。
        """
        if not self.per_layer:
            return {}
        progress = 1.0
        if self.final_sparsity > 0:
            progress = self._targets[round_index - 1] / self.final_sparsity
        return {
            name: max(0.0, min(1.0, float(value) * progress))
            for name, value in self.per_layer.items()
        }

    @staticmethod
    def _to_incremental(
        cumulative: Mapping[str, float], previous: Mapping[str, float]
    ) -> Dict[str, float]:
        """把累积逐层目标换算成本轮需要剪掉的剩余比例。"""
        result: Dict[str, float] = {}
        for name, target in cumulative.items():
            before = float(previous.get(name, 0.0))
            remaining = max(1.0 - before, 1e-12)
            result[name] = max(0.0, min(1.0, (float(target) - before) / remaining))
        return result

    # ------------------------------------------------------------------ 主循环
    def run(self) -> IterativeReport:
        """执行完整的迭代剪枝流程，返回报告。"""
        started = time.time()
        params_before = count_parameters(self.model)
        initial_metrics = self._evaluate(self.model)
        if initial_metrics:
            self._say("[初始] %s" % _format_metrics(initial_metrics))

        records: List[RoundRecord] = []
        unstructured: Optional[UnstructuredPruner] = None
        initial_channels: Optional[int] = None
        if self.mode == "unstructured":
            unstructured = UnstructuredPruner(
                self.model,
                criterion=self.criterion,
                scope=self.scope,
                exclude=self.exclude,
                protect_first_last=self.protect_first_last,
                grads=self.grads,
                seed=self.seed,
            )

        for round_index in range(1, self.rounds + 1):
            round_started = time.time()
            target = self._targets[round_index - 1]
            note = ""

            if self.mode == "unstructured":
                assert unstructured is not None
                layer_targets = self._per_layer_targets(round_index) or None
                report = unstructured.prune(target, per_layer=layer_targets)
                achieved = report.achieved_sparsity
                note = "累积目标 %.4f" % target
            else:
                assert self.example_inputs is not None
                pruner = StructuredPruner(
                    self.model,
                    self.example_inputs,
                    criterion=self.criterion,
                    scope=self.scope,
                    exclude=self.exclude,
                    protect_output=self.protect_output,
                    grads=self.grads,
                    seed=self.seed,
                )
                ratio = self._incremental[round_index - 1]
                per_group = None
                if self.per_layer:
                    per_group = self._to_incremental(
                        self._per_layer_targets(round_index),
                        self._per_layer_targets(round_index - 1) if round_index > 1 else {},
                    )
                report = pruner.prune(ratio, per_group=per_group)
                if initial_channels is None:
                    initial_channels = report.channels_before
                # 记录**累积**通道裁剪比例，才能与「目标」列直接比较
                achieved = (
                    1.0 - report.channels_after / float(initial_channels)
                    if initial_channels
                    else 0.0
                )
                note = "本轮增量比例 %.4f（目标累积 %.4f）" % (ratio, target)

            # 剪枝刚结束（尚未微调）的指标：这是「剪枝造成多少损伤」的直接证据
            metrics_after_prune = self._evaluate(self.model)

            finetune_metrics: Dict[str, float] = {}
            if self.finetune_fn is not None:
                context = RoundContext(
                    round_index=round_index,
                    target=target,
                    achieved=achieved,
                    masks=unstructured.masks if unstructured else None,
                )
                returned = self.finetune_fn(self.model, context)
                finetune_metrics = dict(returned or {})
                if self.mode == "unstructured" and unstructured is not None:
                    # 微调之后必须把掩码重新压回权重，否则剪掉的权重会被重新激活
                    unstructured.materialize()
                metrics = self._evaluate(self.model)
            else:
                metrics = metrics_after_prune

            params = count_parameters(self.model)
            record = RoundRecord(
                round_index=round_index,
                target=target,
                achieved=achieved,
                params=params,
                nonzero_params=count_nonzero_parameters(
                    self.model, masks=unstructured.masks if unstructured else None
                ),
                params_reduction=1.0 - params / float(params_before) if params_before else 0.0,
                flops=self._flops(),
                metrics=metrics,
                metrics_after_prune=metrics_after_prune,
                finetune_metrics=finetune_metrics,
                duration_s=time.time() - round_started,
                note=note,
            )
            records.append(record)
            message = "[第%d/%d轮] 目标=%.4f 实际=%.4f 参数=%d" % (
                round_index, self.rounds, target, achieved, params
            )
            if metrics_after_prune:
                message += " 剪枝后 " + _format_metrics(metrics_after_prune)
            if self.finetune_fn is not None and metrics:
                message += " 微调后 " + _format_metrics(metrics)
            self._say(message)

        final_metrics = self._evaluate(self.model)
        params_after = count_parameters(self.model)
        notes = [
            "非结构化剪枝的稠密参数量与 FLOPs 不随掩码下降，请关注「非零参数」与实测延迟；"
            "结构化剪枝会真正减少参数量与 FLOPs。",
            "每轮之间的微调由调用方通过 finetune_fn 提供，本工具只负责在微调后重新压回掩码。",
        ]
        if self.mode == "unstructured":
            notes.append(
                "本次为非结构化剪枝：形状与稠密 FLOPs 不变，收益体现在有效参数量与稀疏存储；"
                "微调时必须持续压回掩码，否则被剪权重会被优化器重新激活。"
            )
        else:
            notes.append(
                "本次为结构化剪枝：通道被物理删除，因此每轮都要重建通道依赖图"
                "（本工具已自动完成）；FLOPs 只统计卷积/线性这类乘加密集型算子。"
            )
        if self.finetune_fn is None:
            notes.append("本次未提供 finetune_fn，即没有微调；多轮迭代通常需要配合微调才有效果。")

        return IterativeReport(
            mode=self.mode,
            criterion=self.criterion,
            scope=self.scope,
            schedule=self.schedule,
            rounds=self.rounds,
            final_target=self.final_sparsity,
            records=records,
            initial_metrics=initial_metrics,
            final_metrics=final_metrics,
            params_before=params_before,
            params_after=params_after,
            total_duration_s=time.time() - started,
            notes=notes,
            masks=unstructured.masks if unstructured else None,
        )


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return " ".join("%s=%.4f" % (key, value) for key, value in metrics.items())


def iterative_prune(
    model: nn.Module, **kwargs
) -> "tuple[nn.Module, IterativeReport]":
    """一站式接口：``IterativePruner(model, **kwargs).run()``。"""
    report = IterativePruner(model, **kwargs).run()
    return model, report
