"""示例 4：敏感度分析与逐层稀疏度分配（含代理指标的局限性对照）。

为什么需要它
------------
「每层都剪掉同样比例」几乎总是次优：有的层稍微一剪精度就崩（敏感层），
有的层剪掉一半也几乎无损（冗余层）。

本示例做三件事：
1. 用**权重重建误差**（快，不需要数据）扫描每个通道组的敏感度；
2. 用**真实精度**（慢，需要数据）扫描同一批通道组，并对比两者的结论；
3. 在同一总目标下对比「均匀分配 / 代理指标分配 / 真实指标分配」的实测精度。

⚠️ 重要结论（本示例会实测出来）：重建误差是**代理指标**，它只看权重本身、不看这一层
在任务里的作用，因此可能与真实精度排序不一致。工具同时提供两种扫描，就是为了让你
在「快」和「准」之间自己选，而不是把代理指标当成真理。

运行::

    & C:\\Users\\GuoZhe\\.conda\\envs\\gpupy312\\python.exe examples\\04_sensitivity_allocation.py
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pruning_toolkit import (
    StructuredPruner,
    count_flops,
    count_parameters,
    recommend_sparsity,
    structured_sensitivity,
)
from pruning_toolkit.zoo import get_loaders
from pruning_toolkit.zoo.models import small_vgg
from pruning_toolkit.zoo.train import evaluate, fit

SILENT = lambda message: None  # noqa: E731

TARGET = 0.6
PROBE_SPARSITIES = (0.1, 0.3, 0.5, 0.7, 0.9)


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    seed = 0
    torch.manual_seed(seed)

    header("1) 数据与基线")
    train_loader, test_loader, meta = get_loaders(
        "tiny_images", batch_size=64, seed=seed, n_samples=1500, noise=0.6
    )
    model = small_vgg(in_channels=3, num_classes=int(meta["num_classes"]))
    example_inputs = (next(iter(test_loader))[0][:4],)
    fit(model, train_loader, test_loader, epochs=8, lr=0.01, seed=seed, log=SILENT)
    baseline = evaluate(model, test_loader)
    print("基线：test_acc=%.4f 参数量=%d FLOPs=%s" % (
        baseline["acc"], count_parameters(model), count_flops(model, example_inputs)))

    header("2) 快速扫描：权重重建误差（不需要数据）")
    proxy_report = structured_sensitivity(
        model, example_inputs, sparsities=PROBE_SPARSITIES, criterion="l1"
    )
    print(proxy_report.to_report().to_text())

    header("3) 逐通道组裁后，用真实测试精度扫描同一批通道组")
    accuracy_report = structured_sensitivity(
        model,
        example_inputs,
        sparsities=PROBE_SPARSITIES,
        criterion="l1",
        proxy="accuracy",
        eval_fn=lambda candidate: evaluate(candidate, test_loader),
        metric_name="acc",
    )
    print(accuracy_report.to_report().to_text())

    header("4) 两种敏感度排序对比（越靠上越不该剪）")
    proxy_rank = [name for name, _ in proxy_report.rank()]
    accuracy_rank = [name for name, _ in accuracy_report.rank()]
    print("%-6s %-44s %s" % ("名次", "重建误差排序（代理）", "真实精度排序"))
    print("-" * 78)
    for index in range(max(len(proxy_rank), len(accuracy_rank))):
        left = proxy_rank[index] if index < len(proxy_rank) else "-"
        right = accuracy_rank[index] if index < len(accuracy_rank) else "-"
        print("%-6d %-44s %s" % (index + 1, left[:42], right[:42]))
    same = proxy_rank == accuracy_rank
    print("\n两种排序是否一致：%s" % ("一致" if same else "不一致 —— 代理指标确实会误导"))

    header("5) 自动分配逐层目标（目标 %.2f，spread=0.5）" % TARGET)
    proxy_allocation = recommend_sparsity(
        model, TARGET, criterion="l1", sensitivity=proxy_report
    )
    accuracy_allocation = recommend_sparsity(
        model, TARGET, criterion="l1", sensitivity=accuracy_report
    )
    for label, allocation in (("代理指标分配", proxy_allocation), ("真实指标分配", accuracy_allocation)):
        print("%s：加权平均=%.6f（请求 %.6f，可行=%s）" % (
            label, allocation.achieved, allocation.requested, allocation.feasible))
        for name, value in sorted(allocation.per_layer.items(), key=lambda item: -item[1]):
            print("    %-44s %.4f" % (name[:42], value))

    header("6) 三种分配方案的实测对比（同一目标、同样微调轮数）")
    results = {}
    plans = (
        ("均匀分配", None),
        ("代理指标分配", proxy_allocation.per_layer),
        ("真实指标分配", accuracy_allocation.per_layer),
    )
    for label, per_group in plans:
        candidate = copy.deepcopy(model)
        pruner = StructuredPruner(candidate, example_inputs, criterion="l1", scope="local")
        report = pruner.prune(TARGET, per_group=per_group)
        after_prune = evaluate(candidate, test_loader)
        fit(candidate, train_loader, test_loader, epochs=6, lr=0.005, seed=seed, log=SILENT)
        after_finetune = evaluate(candidate, test_loader)
        results[label] = {
            "剪枝后": after_prune["acc"],
            "微调后": after_finetune["acc"],
            "参数量": count_parameters(candidate),
            "FLOPs": count_flops(candidate, example_inputs),
            "通道裁剪": report.channel_reduction,
            "报告": report,
        }
        print("%-14s 剪枝后 acc=%.4f → 微调后 acc=%.4f 通道裁剪=%.4f 参数=%d FLOPs=%s" % (
            label, after_prune["acc"], after_finetune["acc"], report.channel_reduction,
            results[label]["参数量"], results[label]["FLOPs"]))

    header("7) 总结")
    print("%-14s %-10s %-10s %-12s %-12s %-12s" % (
        "分配方案", "剪枝后", "微调后", "通道裁剪", "参数量", "FLOPs"))
    print("-" * 74)
    for label, data in results.items():
        print("%-14s %-10.4f %-10.4f %-12.4f %-12d %-12s" % (
            label, data["剪枝后"], data["微调后"], data["通道裁剪"],
            data["参数量"], data["FLOPs"]))

    parameters = {label: data["参数量"] for label, data in results.items()}
    spread = max(parameters.values()) - min(parameters.values())
    best = max(results.items(), key=lambda item: item[1]["微调后"])
    print("\n本示例中微调后精度最高的是：%s（acc=%.4f）" % (best[0], best[1]["微调后"]))

    print("\n怎么读这个结果（重要，别误读）：")
    print("1. 「目标 0.60」是**按参数量加权的单层稀疏度目标**，但通道裁剪会在层间级联：")
    print("   剪掉某个卷积的输出通道，同时就削掉了下游卷积的输入通道，参数量是二次下降的。")
    print("   所以三种方案的最终参数量并不相等（本示例 %s），精度差异里**同时混进了压缩率差异**，"
          % " vs ".join("%d" % value for value in parameters.values()))
    print("   这不是一份干净的对照实验。要干净地比较分配策略，应固定最终参数量（或固定 FLOPs），")
    print("   再调各自的目标值去逼近它。")
    print("2. 代理指标（重建误差）与真实精度的排序不一致时，代理指标可能给出更差的结果 ——")
    print("   本示例正是如此。它只看权重、不看这一层在任务里的作用，便宜但有风险。")
    print("3. 真实精度扫描在本任务上无法区分通道组（各探测强度下精度都接近 1.0），")
    print("   于是自动退化为均匀分配 —— 这是**正确的保守行为**：没有信号就不要乱分配。")
    print("4. 换一个更难的模型/数据，或者把探测强度推向更极端的位置，才可能看出分配策略的收益。")
    print("   请在你自己任务上重跑本脚本，用实测数字下判断。")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "runs", "example4")
    os.makedirs(out_dir, exist_ok=True)
    proxy_report.to_report().save(os.path.join(out_dir, "sensitivity_proxy.md"))
    accuracy_report.to_report().save(os.path.join(out_dir, "sensitivity_accuracy.md"))
    proxy_allocation.to_report().save(os.path.join(out_dir, "allocation_proxy.md"))
    accuracy_allocation.to_report().save(os.path.join(out_dir, "allocation_accuracy.md"))
    for label, data in results.items():
        safe = {"均匀分配": "uniform", "代理指标分配": "proxy", "真实指标分配": "accuracy"}[label]
        data["报告"].to_report("%s（目标 %.2f）" % (label, TARGET)).save(
            os.path.join(out_dir, "prune_%s.md" % safe)
        )
    print("\n报告已写入 %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
