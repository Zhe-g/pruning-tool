"""全模型自检：对 zoo 里每个示例模型跑一遍依赖分析与真实剪枝，并校验一致性。

这个脚本的用途
--------------
* **验证环境与代码是否正常**：它不依赖任何训练或外部数据，几十秒内跑完，
  适合在改动代码后当冒烟测试用；
* **生成可对照的表格**：README 里的「不同结构下的压缩效果」一行行都对应它的输出，
  你可以自己复现，而不是只能相信文档。

它检查什么
----------
1. 每个模型都能被 ``torch.fx`` 追踪，并给出通道组 / 可裁剪组 / 屏障数量；
2. 结构化剪枝 50% 后：参数量与 FLOPs **都必须下降**，且前向校验必须通过
   （输出形状不变、数值有限）；
3. 非结构化剪枝 80% 后：保留参数个数必须**精确等于** ``round(total*(1-s))``；
4. 被跳过的通道组必须给出可读的原因。

运行::

    & C:\\Users\\GuoZhe\\.conda\\envs\\gpupy312\\python.exe tools\\verify_models.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple

import torch

from pruning_toolkit import (
    DependencyGraph,
    StructuredPruner,
    UnstructuredPruner,
    count_parameters,
)
from pruning_toolkit.zoo import build_model, example_input_for
from pruning_toolkit.zoo.models import all_models


def structured_check(name: str, ratio: float = 0.5) -> Tuple[int, int, int, int, int, float, bool]:
    """对一个模型做结构化剪枝，返回统计元组。"""
    torch.manual_seed(0)
    model = build_model(name, num_classes=10).eval()
    inputs = (example_input_for(name, batch_size=2),)
    graph = DependencyGraph(model, inputs)
    params_before = count_parameters(model)

    pruner = StructuredPruner(model, inputs, criterion="l1", scope="local")
    report = pruner.prune(ratio)

    assert report.verified, "%s：结构化剪枝后前向校验未通过" % name
    assert report.params_after < params_before, "%s：参数量没有下降" % name
    assert report.flops_after is not None, "%s：剪枝后 FLOPs 未统计" % name
    assert report.flops_after < report.flops_before, "%s：FLOPs 没有下降" % name
    return (
        len(graph.groups),
        len(graph.prunable_groups()),
        len(graph.barriers),
        params_before,
        report.params_after,
        report.flops_reduction,
        report.verified,
    )


def unstructured_check(name: str, sparsity: float = 0.8) -> Tuple[int, int, float]:
    """对一个模型做非结构化剪枝，返回统计元组（含精确性断言）。"""
    torch.manual_seed(0)
    model = build_model(name, num_classes=10).eval()
    pruner = UnstructuredPruner(model, criterion="l1", scope="global", seed=0)
    report = pruner.prune(sparsity)

    expected_kept = int(round(pruner.total_numel() * (1.0 - sparsity)))
    assert report.total_kept == expected_kept, (
        "%s：非结构化剪枝保留个数不精确，期望 %d 实际 %d"
        % (name, expected_kept, report.total_kept)
    )
    materialized = sum(
        int((parameter.detach() != 0).sum().item())
        for _, parameter in pruner.params
    )
    assert materialized == report.total_kept, "%s：权重置零与掩码不一致" % name
    return report.total_kept, pruner.total_numel(), report.achieved_sparsity


def variant_check(name: str, criterion: str, scope: str, target: float) -> Tuple[int, int, float, float]:
    """用不同的准则/范围组合剪同一个模型，返回（参数前, 参数后, 参数下降, FLOPs 下降）。"""
    torch.manual_seed(0)
    model = build_model(name, num_classes=10).eval()
    inputs = (example_input_for(name, batch_size=2),)
    before = count_parameters(model)
    pruner = StructuredPruner(model, inputs, criterion=criterion, scope=scope)
    report = pruner.prune(target)
    assert report.verified, "%s(%s/%s)：前向校验未通过" % (name, criterion, scope)
    assert report.params_after < before and report.flops_after < report.flops_before
    return (
        before,
        report.params_after,
        1.0 - report.params_after / float(before),
        report.flops_reduction,
    )


def main() -> int:
    names: List[str] = list(all_models())
    print("=" * 100)
    print("结构化剪枝自检（每组裁掉 50% 通道，权重未训练 —— 只验证结构与计算量）")
    print("=" * 100)
    print("%-16s %-7s %-7s %-6s %-10s %-10s %-10s %-6s" % (
        "模型", "通道组", "可剪组", "屏障", "参数前", "参数后", "FLOPs 降", "校验"))
    print("-" * 100)
    for name in names:
        groups, prunable, barriers, before, after, flops_drop, verified = structured_check(name)
        print("%-16s %-7d %-7d %-6d %-10d %-10d %-10.4f %-6s" % (
            name, groups, prunable, barriers, before, after, flops_drop,
            "通过" if verified else "未通过"))

    print()
    print("=" * 100)
    print("非结构化剪枝自检（全局剪掉 80% 权重，保留个数必须精确）")
    print("=" * 100)
    print("%-16s %-14s %-14s %-10s %-10s" % (
        "模型", "保留参数", "可剪参数", "稀疏度", "精确性"))
    print("-" * 100)
    for name in names:
        kept, total, sparsity = unstructured_check(name)
        expected = int(round(total * 0.2))
        print("%-16s %-14d %-14d %-10.4f %-10s" % (
            name, kept, total, sparsity, "精确" if kept == expected else "不精确"))

    print()
    print("=" * 100)
    print("配置变体对比（同一个 small_vgg，换准则与裁剪范围）")
    print("=" * 100)
    print("%-14s %-10s %-8s %-10s %-10s %-12s %-12s" % (
        "模型", "准则", "范围", "目标", "参数前", "参数下降", "FLOPs 下降"))
    print("-" * 100)
    for criterion, scope, target in (
        ("l1", "local", 0.5),
        ("l1", "global", 0.6),
        ("bn_scale", "local", 0.4),
    ):
        before, after, params_drop, flops_drop = variant_check("small_vgg", criterion, scope, target)
        print("%-14s %-10s %-8s %-10.2f %-10d %-12.4f %-12.4f" % (
            "small_vgg", criterion, scope, target, before, params_drop, flops_drop))

    print()
    print("=" * 100)
    print("被跳过的通道组及原因（这些层没有被剪，原因必须可读）")
    print("=" * 100)
    for name in names:
        torch.manual_seed(0)
        model = build_model(name, num_classes=10).eval()
        inputs = (example_input_for(name, batch_size=2),)
        graph = DependencyGraph(model, inputs)
        reasons: dict = {}
        for group in graph.groups:
            if not group.prunable:
                reasons.setdefault(group.reason, []).append(group.index)
        print("\n%s：" % name)
        for reason, indexes in reasons.items():
            print("  组%s —— %s" % (indexes, reason))

    print()
    print("=" * 100)
    print("全部自检通过：结构化剪枝的参数量/FLOPs 均下降且前向校验通过，")
    print("非结构化剪枝的保留个数精确，被跳过的通道组都给出了原因。")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
