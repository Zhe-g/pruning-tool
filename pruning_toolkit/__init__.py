"""pruning-toolkit：框架无关的模型剪枝工具箱（PyTorch 原生实现）。

按层次组织，逐层可用、也可只用其中一层：

======================  ==========================================================
层                     模块
======================  ==========================================================
依赖分析               :mod:`~pruning_toolkit.graph` —— fx 追踪 + 通道依赖组
重要性准则             :mod:`~pruning_toolkit.criteria` —— l1 / l2 / taylor / bn_scale / fpgm
非结构化剪枝           :mod:`~pruning_toolkit.unstructured` + :mod:`~pruning_toolkit.masks`
结构化剪枝             :mod:`~pruning_toolkit.structured` —— 通道组联动裁剪
敏感度与分配           :mod:`~pruning_toolkit.analysis`
迭代剪枝 + 微调        :mod:`~pruning_toolkit.iterative`
评估指标               :mod:`~pruning_toolkit.metrics`
报告                   :mod:`~pruning_toolkit.report` —— JSON + Markdown 双格式
导出                   :mod:`~pruning_toolkit.export`
配置与流水线           :mod:`~pruning_toolkit.config` / :mod:`~pruning_toolkit.pipeline`
示例模型与数据         :mod:`~pruning_toolkit.zoo`
======================  ==========================================================

最小用法::

    import torch
    from pruning_toolkit import StructuredPruner, benchmark_latency

    model = ...
    example = (torch.randn(4, 3, 32, 32),)
    pruner = StructuredPruner(model, example, criterion="l1", scope="local")
    report = pruner.prune(0.5)          # 裁掉 50% 通道
    print(report.to_text())             # 参数量/FLOPs 下降多少一目了然

一条命令跑完整个流程::

    python -m pruning_toolkit demo
"""

from __future__ import annotations

from .analysis import (
    ModuleSensitivity,
    SensitivityPoint,
    SensitivityReport,
    SparsityAllocation,
    recommend_sparsity,
    reconstruction_error_curve,
    structured_sensitivity,
    unstructured_sensitivity,
)
from .config import FinetuneConfig, OutputConfig, PruneConfig, TrainConfig, load_config
from .criteria import (
    CRITERIA,
    channel_importance,
    collect_grads,
    criterion_requires_grad,
    elementwise_importance,
    group_importance,
)
from .export import (
    export_fx_graph_module,
    export_onnx,
    export_torchscript,
    load_checkpoint,
    model_meta,
    save_checkpoint,
    save_reports,
)
from .graph import ChannelGroup, DependencyGraph, UnsupportedGraphError, build_graph
from .iterative import IterativePruner, IterativeReport, RoundContext, iterative_prune
from .masks import MaskBook, apply_masks
from .metrics import (
    benchmark_latency,
    count_flops,
    count_nonzero_parameters,
    count_parameters,
    evaluate,
    module_table,
    model_sparsity,
    summarize,
)
from .modules import (
    channel_semantics,
    is_depthwise,
    iter_weight_parameters,
    resize_both_,
    resize_coupled_,
    resize_input_,
    resize_output_,
)
from .pipeline import PipelineResult, prepare, run
from .report import Report, format_value, kv, markdown, table, text
from .schedules import (
    SCHEDULES,
    Scheduler,
    cumulative_targets,
    incremental_ratios,
    per_round_ratio,
)
from .structured import GroupPruneStat, StructuredPruner, StructuredReport, prune_structured
from .unstructured import (
    LayerPruneStat,
    UnstructuredPruner,
    UnstructuredReport,
    prune_unstructured,
)
from .zoo import all_models, build_model, example_input_for, get_loaders

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # 依赖分析
    "DependencyGraph",
    "ChannelGroup",
    "build_graph",
    "UnsupportedGraphError",
    "channel_semantics",
    "is_depthwise",
    "iter_weight_parameters",
    "resize_output_",
    "resize_input_",
    "resize_both_",
    "resize_coupled_",
    # 准则
    "CRITERIA",
    "elementwise_importance",
    "channel_importance",
    "group_importance",
    "collect_grads",
    "criterion_requires_grad",
    # 掩码
    "MaskBook",
    "apply_masks",
    # 剪枝
    "UnstructuredPruner",
    "UnstructuredReport",
    "LayerPruneStat",
    "prune_unstructured",
    "StructuredPruner",
    "StructuredReport",
    "GroupPruneStat",
    "prune_structured",
    # 敏感度与迭代
    "unstructured_sensitivity",
    "structured_sensitivity",
    "recommend_sparsity",
    "reconstruction_error_curve",
    "SensitivityReport",
    "SensitivityPoint",
    "ModuleSensitivity",
    "SparsityAllocation",
    "IterativePruner",
    "IterativeReport",
    "RoundContext",
    "iterative_prune",
    # 调度
    "Scheduler",
    "cumulative_targets",
    "incremental_ratios",
    "per_round_ratio",
    "SCHEDULES",
    # 指标
    "count_parameters",
    "count_nonzero_parameters",
    "model_sparsity",
    "count_flops",
    "benchmark_latency",
    "evaluate",
    "module_table",
    "summarize",
    # 报告
    "Report",
    "table",
    "kv",
    "text",
    "markdown",
    "format_value",
    # 导出
    "save_checkpoint",
    "load_checkpoint",
    "export_torchscript",
    "export_onnx",
    "export_fx_graph_module",
    "save_reports",
    "model_meta",
    # 配置与流水线
    "PruneConfig",
    "TrainConfig",
    "FinetuneConfig",
    "OutputConfig",
    "load_config",
    "run",
    "prepare",
    "PipelineResult",
    # 示例模型与数据
    "build_model",
    "all_models",
    "example_input_for",
    "get_loaders",
]
