"""示例 2：结构化剪枝 —— 真正把 VGG 的通道删掉。

与非结构化剪枝的区别
--------------------
* 张量形状真的变小：卷积输出通道、BN 的 ``gamma/beta/running_mean/running_var``、
  下游层的输入通道会**用同一组索引**一起裁；
* 因此参数量与 FLOPs 都下降，延迟才可能真的降低；
* 工具会在剪枝后立刻跑一次前向校验（形状不变 + 数值有限），不通过就直接报错，
  而不是交给你一个前向会崩的模型。

运行::

    & C:\\Users\\GuoZhe\\.conda\\envs\\gpupy312\\python.exe examples\\02_structured_cnn.py
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pruning_toolkit import (
    DependencyGraph,
    StructuredPruner,
    benchmark_latency,
    count_flops,
    count_parameters,
    export_torchscript,
)
from pruning_toolkit.zoo import get_loaders
from pruning_toolkit.zoo.models import small_vgg
from pruning_toolkit.zoo.train import evaluate, fit

SILENT = lambda message: None  # noqa: E731


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    seed = 0
    torch.manual_seed(seed)

    header("1) 数据与模型")
    train_loader, test_loader, meta = get_loaders(
        "tiny_images", batch_size=64, seed=seed, n_samples=1500, noise=0.6
    )
    model = small_vgg(in_channels=int(meta["in_channels"]), num_classes=int(meta["num_classes"]))
    example_inputs = (next(iter(test_loader))[0][:4],)
    print("模型 %s：参数量 %d，示例输入 %s" % (
        type(model).__name__, count_parameters(model), tuple(example_inputs[0].shape)))

    header("2) 训练基线")
    fit(model, train_loader, test_loader, epochs=8, lr=0.01, seed=seed, log=SILENT)
    baseline = evaluate(model, test_loader)
    baseline_params = count_parameters(model)
    baseline_flops = count_flops(model, example_inputs)
    baseline_latency = benchmark_latency(model, example_inputs)
    print("基线：test_acc=%.4f 参数=%d FLOPs=%s 延迟p50=%.4f ms" % (
        baseline["acc"], baseline_params, baseline_flops, baseline_latency["p50_ms"]))

    header("3) 通道依赖图：哪些层能安全剪？")
    backup = copy.deepcopy(model)  # 结构化剪枝是破坏性的，留一份基线做对照
    graph = DependencyGraph(model, example_inputs)
    graph_report = graph.summary()
    print(graph_report.to_text())

    header("4) 结构化剪枝：每组裁掉 50% 通道")
    pruner = StructuredPruner(model, example_inputs, criterion="l1", scope="local")
    report = pruner.prune(0.5)
    print(report.to_report().to_text())
    assert report.verified, "剪枝后前向校验未通过"

    pruned = evaluate(model, test_loader)
    pruned_params = count_parameters(model)
    pruned_flops = count_flops(model, example_inputs)
    pruned_latency = benchmark_latency(model, example_inputs)
    print("\n剪枝后：test_acc=%.4f 参数=%d（-%.2f%%）FLOPs=%s（-%.2f%%）延迟p50=%.4f ms" % (
        pruned["acc"], pruned_params, 100 * (1 - pruned_params / baseline_params),
        pruned_flops, 100 * report.flops_reduction, pruned_latency["p50_ms"]))

    header("5) 微调恢复")
    fit(model, train_loader, test_loader, epochs=6, lr=0.005, seed=seed, log=SILENT)
    finetuned = evaluate(model, test_loader)

    header("6) 对照总结")
    print("%-10s %-10s %-10s %-14s %-14s" % ("阶段", "test_acc", "参数量", "FLOPs", "延迟 p50(ms)"))
    print("-" * 62)
    for name, net, metrics in (
        ("基线", backup, (baseline_params, baseline_flops, baseline_latency["p50_ms"])),
        ("剪枝后", model, (pruned_params, pruned_flops, pruned_latency["p50_ms"])),
    ):
        accuracy = baseline["acc"] if name == "基线" else pruned["acc"]
        print("%-10s %-10.4f %-10d %-14s %-14.4f" % (name, accuracy, metrics[0], metrics[1], metrics[2]))
    print("%-10s %-10.4f %-10d %-14s %-14s" % (
        "微调后", finetuned["acc"], count_parameters(model),
        count_flops(model, example_inputs), "-"))
    print("\n注意：参数量下降 %d → %d，但延迟只从 %.4f ms 降到 %.4f ms。" % (
        baseline_params, pruned_params, baseline_latency["p50_ms"], pruned_latency["p50_ms"]))
    print("小模型在 CPU 上受算子启动开销与内存带宽主导，参数量下降不会等比例变成延迟下降；")
    print("想要延迟收益，需要模型足够大、算子足够规整，并配合推理引擎/量化等手段。")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "example2")
    os.makedirs(out_dir, exist_ok=True)
    graph_report.save(os.path.join(out_dir, "graph.md"))
    report.to_report().save(os.path.join(out_dir, "structured_report.md"))
    export_torchscript(model, example_inputs, os.path.join(out_dir, "pruned_model.ts"))
    print("\n报告与 TorchScript 已写入 %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
