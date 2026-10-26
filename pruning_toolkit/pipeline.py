"""端到端流水线：数据 → 模型 →（可选）训练 → 敏感度分配 → 剪枝 → 微调 → 评估 → 导出。

这是把各层串起来的「一站式入口」。CLI 的 ``prune`` / ``demo`` 子命令直接调用它，
示例脚本也可以用它对照「手工调用各模块」的写法。

流水线的六个阶段
----------------
1. 准备数据（离线合成数据集，见 :mod:`pruning_toolkit.zoo.data`）
2. 构建模型（见 :mod:`pruning_toolkit.zoo.models`）
3. 初始训练（可选；已有权重时关掉）
4. 依赖图分析与（可选）敏感度驱动的逐层稀疏度分配
5. 迭代式剪枝 + 每轮微调
6. 最终评估与产物导出（报告 / 检查点 / 掩码 / TorchScript）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import export as export_mod
from . import report as report_mod
from .analysis import (
    SensitivityReport,
    SparsityAllocation,
    recommend_sparsity,
    structured_sensitivity,
    unstructured_sensitivity,
)
from .config import PruneConfig
from .criteria import collect_grads, criterion_requires_grad
from .graph import DependencyGraph
from .iterative import IterativePruner, IterativeReport
from .metrics import (
    benchmark_latency,
    count_nonzero_parameters,
    model_sparsity,
    summarize,
)
from .zoo import build_model, get_loaders
from .zoo.train import evaluate as zoo_evaluate
from .zoo.train import fit as zoo_fit
from .zoo.train import resolve_device

__all__ = ["PipelineResult", "run", "prepare"]

Logger = Optional[Callable[[str], None]]


@dataclass
class PipelineResult:
    """流水线的全部产出。"""

    config: PruneConfig
    model: nn.Module
    data_meta: Dict[str, Any] = field(default_factory=dict)
    reports: Dict[str, report_mod.Report] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    iteration: Optional[IterativeReport] = None
    sensitivity: Optional[SensitivityReport] = None
    allocation: Optional[SparsityAllocation] = None
    history: Dict[str, List[float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ 摘要
    def summary_text(self) -> str:
        """一段可直接打印给人的中文摘要。"""
        lines = ["=" * 66, "剪枝流水线结果", "=" * 66]
        lines.append("模型：%s    数据集：%s    设备：%s" % (
            self.config.model, self.config.dataset, self.config.device))
        mode = (
            "结构化（物理删除通道）"
            if self.config.mode == "structured"
            else "非结构化（掩码置零）"
        )
        lines.append("模式：%s    准则：%s    范围：%s    轮数：%d" % (
            mode, self.config.criterion, self.config.scope, self.config.rounds))

        metrics = self.metrics
        if "参数量_剪枝前" in metrics:
            lines.append("-" * 66)
            lines.append("参数量      : %d → %d （下降 %.2f%%）" % (
                metrics.get("参数量_剪枝前", 0), metrics.get("参数量_剪枝后", 0),
                100.0 * metrics.get("参数下降比例", 0.0)))
            lines.append("非零参数    : %d → %d" % (
                metrics.get("非零参数_剪枝前", 0), metrics.get("非零参数_剪枝后", 0)))
            lines.append("稀疏度      : %.4f（全模型参数，含未参与剪枝的偏置）" % metrics.get("稀疏度_剪枝后", 0.0))
            if "稀疏度_可剪参数" in metrics:
                lines.append("稀疏度      : %.4f（仅参与剪枝的参数，与目标可比）" % metrics["稀疏度_可剪参数"])
            if metrics.get("FLOPs_剪枝前"):
                flops_after = metrics.get("FLOPs_剪枝后")
                lines.append("FLOPs       : %s → %s （下降 %.2f%%）" % (
                    metrics["FLOPs_剪枝前"],
                    flops_after if flops_after is not None else "未统计",
                    100.0 * metrics.get("FLOPs下降比例", 0.0)))
            if metrics.get("延迟_p50_ms"):
                lines.append("延迟实测 p50(ms) : %.4f → %.4f" % (
                    metrics.get("延迟_剪枝前_p50_ms", float("nan")), metrics["延迟_p50_ms"]))
            if self.config.mode == "unstructured":
                lines.append("提示        : 非结构化剪枝不改变稠密算子，FLOPs 与延迟差异属于测量噪声；")
                lines.append("              它省的是有效参数量与存储，不是稠密计算量。")
        for key in ("准确率_剪枝前", "准确率_剪枝后未微调", "准确率_微调后"):
            if key in metrics:
                lines.append("%-14s: %.4f" % (key.replace("_", " "), metrics[key]))
        if self.artifacts:
            lines.append("-" * 66)
            lines.append("产物：")
            for name, path in sorted(self.artifacts.items()):
                lines.append("  %-16s %s" % (name, path))
        lines.append("=" * 66)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "data_meta": self.data_meta,
            "reports": list(self.reports),
            "iteration": self.iteration.to_dict() if self.iteration else None,
            "allocation": self.allocation.to_dict() if self.allocation else None,
            "sensitivity": self.sensitivity.to_dict() if self.sensitivity else None,
        }


# ---------------------------------------------------------------------- 辅助


def _say(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(message)


def _model_kwargs(config: PruneConfig, data_meta: Mapping[str, Any]) -> Dict[str, Any]:
    """按数据集形状给模型工厂函数准备参数。"""
    if config.model == "mlp":
        return {"in_features": int(data_meta.get("in_features", 2))}
    return {"in_channels": int(data_meta.get("in_channels", 3))}


def _example_inputs(loader: DataLoader, device: torch.device, size: int = 4) -> Tuple[torch.Tensor, ...]:
    """从真实数据里取一小批做示例输入（追踪与基准测试都用它）。"""
    inputs, _ = next(iter(loader))
    if isinstance(inputs, (list, tuple)):
        return tuple(item[:size].to(device) for item in inputs)
    return (inputs[:size].to(device),)


def _loss_fn(device: torch.device) -> Callable[[nn.Module, Any], torch.Tensor]:
    def loss(model: nn.Module, batch: Any) -> torch.Tensor:
        inputs, targets = batch[0], batch[1]
        if isinstance(inputs, (list, tuple)):
            logits = model(*[item.to(device) for item in inputs])
        else:
            logits = model(inputs.to(device))
        return torch.nn.functional.cross_entropy(logits, targets.to(device))

    return loss


def prepare(
    config: PruneConfig, log: Logger = None
) -> Tuple[nn.Module, DataLoader, DataLoader, Dict[str, Any], Tuple[torch.Tensor, ...], torch.device]:
    """构建数据与模型（流水线前两步），供外部复用。"""
    _say(log, "[1/6] 准备数据集：%s" % config.dataset)
    train_loader, test_loader, data_meta = get_loaders(
        config.dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        **config.dataset_kwargs,
    )
    _say(
        log,
        "      训练样本=%s 测试样本=%s 类别=%s"
        % (data_meta.get("n_train"), data_meta.get("n_test"), data_meta.get("num_classes")),
    )
    device = resolve_device(config.device)
    _say(log, "[2/6] 构建模型：%s（设备 %s）" % (config.model, device))
    model = build_model(config.model, num_classes=int(data_meta.get("num_classes", 10)),
                        **_model_kwargs(config, data_meta))
    model.to(device)
    example_inputs = _example_inputs(test_loader, device)
    return model, train_loader, test_loader, data_meta, example_inputs, device


# ---------------------------------------------------------------------- 主流程


def run(
    config: Optional[PruneConfig] = None,
    log: Logger = print,
    **overrides: Any,
) -> PipelineResult:
    """执行一次完整的剪枝流水线。

    参数
    ----
    config:    配置对象；``None`` 表示使用 :meth:`PruneConfig.for_demo`
    log:       日志回调；传 ``None`` 则静默
    **overrides: 直接覆盖配置字段（如 ``sparsity=0.7``、``rounds=5``）

    返回
    ----
    :class:`PipelineResult`
    """
    started = time.time()
    cfg = config or PruneConfig.for_demo()
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    cfg.validate()
    logger = log if cfg.verbose else None

    model, train_loader, test_loader, data_meta, example_inputs, device = prepare(cfg, logger)

    # ---- 初始训练 ----
    history: Dict[str, List[float]] = {}
    if cfg.train.enabled and cfg.train.epochs > 0:
        _say(logger, "[3/6] 初始训练 %d 轮（lr=%.4g）" % (cfg.train.epochs, cfg.train.lr))
        history = zoo_fit(
            model,
            train_loader,
            test_loader,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
            device=device,
            log=logger,
            seed=cfg.seed,
            max_batches=cfg.train.max_batches,
        )
    else:
        _say(logger, "[3/6] 跳过初始训练")

    def eval_fn(current: nn.Module) -> Dict[str, float]:
        result = zoo_evaluate(current, test_loader, device=device)
        return {"loss": float(result.get("loss", float("nan"))),
                "acc": float(result.get("acc", float("nan")))}

    metrics: Dict[str, Any] = {}
    metrics_before = eval_fn(model)
    metrics["准确率_剪枝前"] = metrics_before["acc"]
    _say(logger, "[4/6] 剪枝前测试准确率 = %.4f" % metrics_before["acc"])

    params_before = sum(param.numel() for param in model.parameters())
    nonzero_before = count_nonzero_parameters(model)
    latency_before = None
    if cfg.benchmark:
        latency_before = benchmark_latency(model, example_inputs, device=device)

    # ---- 依赖图与敏感度 ----
    reports: Dict[str, report_mod.Report] = {}
    graph = DependencyGraph(
        model, example_inputs, protect_output=cfg.protect_output, ignore=cfg.exclude
    )
    reports["graph"] = graph.summary()
    _say(
        logger,
        "      通道组 %d 个（可裁剪 %d 个，屏障 %d 个）"
        % (len(graph.groups), len(graph.prunable_groups()), len(graph.barriers)),
    )
    flops_before = graph.flops()

    grads: Optional[Dict[str, torch.Tensor]] = None
    if criterion_requires_grad(cfg.criterion):
        _say(logger, "      准则 %s 需要梯度，正在用测试集做前向-反传校准…" % cfg.criterion)
        grads = collect_grads(model, _loss_fn(device), test_loader, max_batches=4, device=device)

    per_layer = dict(cfg.per_layer)
    sensitivity: Optional[SensitivityReport] = None
    allocation: Optional[SparsityAllocation] = None
    if cfg.allocate:
        _say(logger, "      敏感度扫描（自动分配逐层目标）…")
        if cfg.mode == "structured":
            sensitivity = structured_sensitivity(
                model,
                example_inputs,
                criterion=cfg.criterion,
                grads=grads,
                exclude=cfg.exclude,
                protect_output=cfg.protect_output,
            )
        else:
            sensitivity = unstructured_sensitivity(
                model, criterion=cfg.criterion, exclude=cfg.exclude
            )
        allocation = recommend_sparsity(
            model,
            cfg.sparsity,
            criterion=cfg.criterion,
            alpha=cfg.alpha,
            reference=cfg.reference,
            exclude=cfg.exclude,
            sensitivity=sensitivity,
        )
        per_layer = {**allocation.per_layer, **per_layer}
        reports["sensitivity"] = sensitivity.to_report()
        reports["allocation"] = allocation.to_report()
        _say(
            logger,
            "      分配完成：目标 %.3f，实际 %.3f，可行=%s"
            % (allocation.requested, allocation.achieved, allocation.feasible),
        )

    # ---- 迭代剪枝 + 微调 ----
    _say(
        logger,
        "[5/6] 开始%s剪枝：%d 轮，目标 %.3f，调度 %s"
        % ("结构化" if cfg.mode == "structured" else "非结构化",
           cfg.rounds, cfg.sparsity, cfg.schedule),
    )

    finetune_fn = None
    if cfg.finetune.enabled and cfg.finetune.epochs_per_round > 0:
        def finetune_fn(current: nn.Module, context) -> Dict[str, float]:
            hook = None
            if context.masks is not None:
                masks = context.masks
                hook = lambda target: masks.apply(target)
            result = zoo_fit(
                current,
                train_loader,
                test_loader,
                epochs=cfg.finetune.epochs_per_round,
                lr=cfg.finetune.lr,
                momentum=cfg.finetune.momentum,
                weight_decay=cfg.finetune.weight_decay,
                device=device,
                post_step_hook=hook,
                log=logger,
                seed=cfg.seed + context.round_index,
            )
            accuracy = result.get("test_acc") or []
            return {"acc": float(accuracy[-1])} if accuracy else {}

    iteration = IterativePruner(
        model,
        example_inputs if cfg.mode == "structured" else None,
        mode=cfg.mode,
        criterion=cfg.criterion,
        scope=cfg.scope,
        schedule=cfg.schedule,
        rounds=cfg.rounds,
        final_sparsity=cfg.sparsity,
        per_layer=per_layer or None,
        exclude=cfg.exclude,
        seed=cfg.seed,
        eval_fn=eval_fn,
        finetune_fn=finetune_fn,
        protect_first_last=cfg.protect_first_last,
        protect_output=cfg.protect_output,
        grads=grads,
        verbose=bool(cfg.verbose),
        log=log,
    ).run()

    reports["pruning"] = iteration.to_report("剪枝过程报告")

    # ---- 评估与产物 ----
    metrics_after = eval_fn(model)
    latency_after = benchmark_latency(model, example_inputs, device=device) if cfg.benchmark else None
    reports["model"] = summarize(
        model, example_inputs, masks=iteration.masks, latency=latency_after
    )

    params_after = sum(param.numel() for param in model.parameters())
    nonzero_after = count_nonzero_parameters(model, masks=iteration.masks)
    flops_after = None
    try:
        flops_after = DependencyGraph(
            model, example_inputs, protect_output=cfg.protect_output, ignore=cfg.exclude
        ).flops()
    except Exception:  # noqa: BLE001 - 剪枝后追踪失败不应中断整个流程
        flops_after = None

    metrics.update(
        {
            "模型": cfg.model,
            "数据集": cfg.dataset,
            "剪枝模式": cfg.mode,
            "重要性准则": cfg.criterion,
            "轮数": cfg.rounds,
            "目标": cfg.sparsity,
            "参数量_剪枝前": params_before,
            "参数量_剪枝后": params_after,
            "参数下降比例": 1.0 - params_after / float(params_before) if params_before else 0.0,
            "非零参数_剪枝前": nonzero_before,
            "非零参数_剪枝后": nonzero_after,
            "稀疏度_剪枝后": model_sparsity(model, masks=iteration.masks),
            "FLOPs_剪枝前": flops_before,
            "FLOPs_剪枝后": flops_after,
            "FLOPs下降比例": (
                1.0 - flops_after / float(flops_before)
                if flops_after and flops_before
                else 0.0
            ),
            "准确率_剪枝后": metrics_after["acc"],
            "通道组总数": len(graph.groups),
            "可剪通道组": len(graph.prunable_groups()),
            "屏障节点数": len(graph.barriers),
            "总耗时_秒": time.time() - started,
        }
    )
    if latency_before and latency_after:
        metrics["延迟_剪枝前_ms"] = latency_before["mean_ms"]
        metrics["延迟_剪枝前_p50_ms"] = latency_before["p50_ms"]
        metrics["延迟均值_ms"] = latency_after["mean_ms"]
        metrics["延迟_p50_ms"] = latency_after["p50_ms"]
        metrics["延迟_P90_ms"] = latency_after["p90_ms"]
        metrics["吞吐_样本每秒"] = latency_after["throughput"]
    if iteration.records:
        metrics["准确率_剪枝后未微调"] = iteration.records[-1].metrics_after_prune.get(
            "acc", metrics_after["acc"]
        )
        metrics["准确率_微调后"] = iteration.records[-1].metrics.get("acc", metrics_after["acc"])
        # 未结构化剪枝时，模型级稀疏度会被「不参与剪枝的偏置」稀释，所以两个口径都报
        metrics["稀疏度_可剪参数"] = iteration.records[-1].achieved

    artifacts: Dict[str, str] = {}
    out_dir = cfg.output.run_directory
    if cfg.output.write_reports:
        saved = export_mod.save_reports(reports, out_dir)
        for name, paths in saved.items():
            artifacts["报告:%s(json)" % name] = paths["json"]
            artifacts["报告:%s(md)" % name] = paths["markdown"]
    artifacts["配置"] = cfg.save(os.path.join(out_dir, "config.json"))
    artifacts["指标"] = export_mod.write_json(metrics, os.path.join(out_dir, "metrics.json"))
    if cfg.output.write_checkpoint:
        artifacts["检查点"] = export_mod.save_checkpoint(
            model,
            os.path.join(out_dir, "checkpoint.pt"),
            meta={
                "config": cfg.to_dict(),
                "metrics": metrics,
                "pruning": (
                    "structured" if cfg.mode == "structured" else "unstructured"
                ),
            },
            masks=iteration.masks,
            history=history or None,
        )
    if cfg.output.write_masks and iteration.masks is not None:
        artifacts["掩码"] = iteration.masks.save(os.path.join(out_dir, "masks.pt"))
    if cfg.output.export_torchscript:
        artifacts["TorchScript"] = export_mod.export_torchscript(
            model, example_inputs, os.path.join(out_dir, "model.ts")
        )

    _say(logger, "[6/6] 完成：参数下降 %.2f%%，准确率 %.4f → %.4f" % (
        100 * metrics["参数下降比例"], metrics["准确率_剪枝前"],
        metrics.get("准确率_微调后", metrics["准确率_剪枝后"])))

    return PipelineResult(
        config=cfg,
        model=model,
        data_meta=data_meta,
        reports=reports,
        metrics=metrics,
        artifacts=artifacts,
        iteration=iteration,
        sensitivity=sensitivity,
        allocation=allocation,
        history=history,
    )
