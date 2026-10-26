"""示例 1：非结构化剪枝 —— 把 MLP 的 90% 权重置零。

演示要点
--------
* 用全局幅值剪枝（``scope="global"``）把 90% 的权重置零；
* 剪枝会让精度掉下来，**掩码感知微调**（每个 step 后把掩码压回权重）能把它救回来；
* 非结构化剪枝不改变稠密算子的 FLOPs —— 本示例直接把「稠密参数量」和
  「非零参数量（有效参数量）」两个口径都打印出来对照。

运行::

    & C:\\Users\\GuoZhe\\.conda\\envs\\gpupy312\\python.exe examples\\01_unstructured_mlp.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pruning_toolkit import (
    MaskBook,
    UnstructuredPruner,
    count_nonzero_parameters,
    count_parameters,
    model_sparsity,
)
from pruning_toolkit.zoo import get_loaders
from pruning_toolkit.zoo.models import mlp
from pruning_toolkit.zoo.train import evaluate, fit

#: 传给 fit 的空日志函数，用来静默训练过程
SILENT = lambda message: None  # noqa: E731


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    seed = 0
    torch.manual_seed(seed)

    header("1) 准备数据与模型")
    train_loader, test_loader, meta = get_loaders(
        "blobs", batch_size=64, seed=seed, n_samples=1500, n_classes=3
    )
    model = mlp(in_features=int(meta["in_features"]), num_classes=int(meta["num_classes"]))
    print("数据集 blobs：训练 %d 测试 %d 类别 %d 输入维度 %d" % (
        meta["n_train"], meta["n_test"], meta["num_classes"], meta["in_features"]))
    print("模型参数量（稠密）：%d" % count_parameters(model))

    header("2) 训练基线模型")
    history = fit(model, train_loader, test_loader, epochs=30, lr=0.05, seed=seed, log=SILENT)
    baseline = evaluate(model, test_loader)
    print("基线：test_acc=%.4f  test_loss=%.4f" % (baseline["acc"], baseline["loss"]))

    header("3) 全局幅值剪枝 90%")
    pruner = UnstructuredPruner(model, criterion="l1", scope="global", seed=seed)
    report = pruner.prune(0.9)
    print(report.to_report().to_text())

    pruned = evaluate(model, test_loader)
    print("\n剪枝后：test_acc=%.4f" % pruned["acc"])
    print("  稠密参数量 = %d（没变，因为张量形状不变）" % count_parameters(model))
    print("  非零参数量 = %d（这就是真正省下来的部分）" % count_nonzero_parameters(model))
    print("  全模型稀疏度 = %.4f" % model_sparsity(model))

    header("4) 掩码感知微调")
    masks = pruner.masks
    assert isinstance(masks, MaskBook)
    print("微调时每个 optimizer step 之后都调用 masks.apply(model)，")
    print("这样被剪掉的权重不会被优化器重新激活。")
    fit(
        model,
        train_loader,
        test_loader,
        epochs=20,
        lr=0.02,
        seed=seed,
        post_step_hook=lambda current: masks.apply(current),
        log=SILENT,
    )
    pruner.materialize()  # 训练循环结束后再压一次，确保万无一失
    finetuned = evaluate(model, test_loader)

    # 硬校验：所有掩码为 0 的位置，权重必须仍然是 0
    leftovers = 0
    for name, param in model.named_parameters():
        mask = masks.get(name)
        if mask is None:
            continue
        leftovers += int(torch.count_nonzero(param.detach() * (~mask)).item())
    assert leftovers == 0, "掩码感知微调失效：%d 个被剪权重变成非零" % leftovers

    print("\n微调后：test_acc=%.4f（被剪权重仍为 0 的位置数：%d）" % (finetuned["acc"], leftovers))

    header("5) 结果对照")
    print("%-12s %-12s %-14s %-12s" % ("阶段", "test_acc", "稠密参数量", "非零参数量"))
    print("-" * 56)
    print("%-12s %-12.4f %-14d %-12d" % ("基线", baseline["acc"],
                                          count_parameters(model), count_parameters(model)))
    print("%-12s %-12.4f %-14d %-12d" % ("剪枝后", pruned["acc"],
                                          count_parameters(model),
                                          report.total_kept + (count_parameters(model) - report.total_numel)))
    print("%-12s %-12.4f %-14d %-12d" % ("微调后", finetuned["acc"],
                                          count_parameters(model), count_nonzero_parameters(model)))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "example1")
    os.makedirs(out_dir, exist_ok=True)
    masks.save(os.path.join(out_dir, "masks.pt"))
    report.to_report().save(os.path.join(out_dir, "unstructured_report.md"))
    print("\n掩码与报告已写入 %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
