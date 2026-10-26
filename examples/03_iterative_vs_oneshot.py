"""示例 3：迭代式剪枝 vs 一次性剪枝（同等微调预算）。

这是迭代剪枝存在的理由
----------------------
一次性把 95% 权重剪掉，网络会瞬间「失忆」，剩下的 5% 参数往往不足以重建原来的函数，
再微调也救不回来；而分多轮、每轮剪一点再微调，网络可以逐轮适应新的稀疏结构。

本示例在**同一总微调轮数**下对比两种策略，避免「谁微调得多谁赢」的假结论。

运行::

    & C:\\Users\\GuoZhe\\.conda\\envs\\gpupy312\\python.exe examples\\03_iterative_vs_oneshot.py
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pruning_toolkit import (
    IterativePruner,
    UnstructuredPruner,
    count_nonzero_parameters,
    count_parameters,
)
from pruning_toolkit.zoo import get_loaders
from pruning_toolkit.zoo.models import mlp
from pruning_toolkit.zoo.train import evaluate, fit

SILENT = lambda message: None  # noqa: E731

SPARSITY = 0.95
TOTAL_FINETUNE_EPOCHS = 20
ROUNDS = 4


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    seed = 0
    torch.manual_seed(seed)

    header("1) 数据与基线模型（two_moons 是非线性可分任务，比 blobs 更考验容量）")
    train_loader, test_loader, meta = get_loaders(
        "two_moons", batch_size=64, seed=seed, n_samples=1500, noise=0.2
    )
    baseline_model = mlp(in_features=2, hidden=(32, 32), num_classes=2)
    fit(baseline_model, train_loader, test_loader, epochs=40, lr=0.05, seed=seed, log=SILENT)
    baseline = evaluate(baseline_model, test_loader)
    print("基线：test_acc=%.4f 参数量=%d" % (baseline["acc"], count_parameters(baseline_model)))

    header("2) 方案 A：一次性剪到 %.0f%%，再微调 %d 轮" % (SPARSITY * 100, TOTAL_FINETUNE_EPOCHS))
    one_shot = copy.deepcopy(baseline_model)
    pruner = UnstructuredPruner(one_shot, criterion="l1", scope="global", seed=seed)
    report_a = pruner.prune(SPARSITY)
    after_prune_a = evaluate(one_shot, test_loader)
    print("  剪枝瞬间：test_acc=%.4f（非零参数 %d/%d）" % (
        after_prune_a["acc"], count_nonzero_parameters(one_shot), count_parameters(one_shot)))
    masks_a = pruner.masks
    fit(
        one_shot, train_loader, test_loader,
        epochs=TOTAL_FINETUNE_EPOCHS, lr=0.02, seed=seed,
        post_step_hook=lambda current: masks_a.apply(current), log=SILENT,
    )
    pruner.materialize()
    result_a = evaluate(one_shot, test_loader)
    print("  微调 %d 轮后：test_acc=%.4f" % (TOTAL_FINETUNE_EPOCHS, result_a["acc"]))
    header("3) 方案 B：分 %d 轮迭代剪枝，每轮微调 %d 轮（总轮数相同）"
           % (ROUNDS, TOTAL_FINETUNE_EPOCHS // ROUNDS))
    iterative_model = copy.deepcopy(baseline_model)

    def finetune(current, context):
        """每轮剪枝后的微调：post_step_hook 保证被剪权重不会被重新激活。"""
        masks = context.masks
        hook = (lambda target: masks.apply(target)) if masks is not None else None
        fit(
            current, train_loader, test_loader,
            epochs=TOTAL_FINETUNE_EPOCHS // ROUNDS, lr=0.02, seed=seed + context.round_index,
            post_step_hook=hook, log=SILENT,
        )
        return {"acc": evaluate(current, test_loader)["acc"]}

    iteration = IterativePruner(
        iterative_model,
        mode="unstructured",
        criterion="l1",
        scope="global",
        schedule="cubic",
        rounds=ROUNDS,
        final_sparsity=SPARSITY,
        seed=seed,
        eval_fn=lambda current: evaluate(current, test_loader),
        finetune_fn=finetune,
        verbose=False,
        log=None,
    ).run()
    result_b = evaluate(iterative_model, test_loader)
    print(iteration.to_report().to_text())

    header("4) 结论")
    print("%-22s %-12s %-12s %-14s" % ("方案", "test_acc", "非零参数", "微调总轮数"))
    print("-" * 62)
    print("%-22s %-12.4f %-12d %-14d" % (
        "基线（未剪枝）", baseline["acc"], count_parameters(baseline_model), 0))
    print("%-22s %-12.4f %-12d %-14d" % (
        "一次性剪枝 + 微调", result_a["acc"], count_nonzero_parameters(one_shot),
        TOTAL_FINETUNE_EPOCHS))
    print("%-22s %-12.4f %-12d %-14d" % (
        "迭代剪枝 + 微调", result_b["acc"], count_nonzero_parameters(iterative_model),
        TOTAL_FINETUNE_EPOCHS))
    print("\n差距 = %+.4f（迭代 - 一次性）。" % (result_b["acc"] - result_a["acc"]))
    worst_iterative = min(
        record.metrics_after_prune.get("acc", 1.0) for record in iteration.records
    )
    print("参考「剪枝瞬间」的损伤：一次性剪到 %.0f%% 时 acc=%.4f；"
          % (SPARSITY * 100, after_prune_a["acc"]))
    print("迭代过程中各轮剪枝瞬间的最低 acc=%.4f（两者稀疏度不同，不可直接比较，"
          % worst_iterative)
    print("仅用来说明「一刀切」与「逐步加码」在过程上的差别）。")
    print("\n说明：在这么小的模型 + 这么简单的任务上，两种策略的最终差距很小（本示例实测 %+.4f）。" % (
        result_b["acc"] - result_a["acc"]))
    print("差距会随模型规模、稀疏度、任务难度而放大，而这类结论**必须在自己的模型上实测**——")
    print("本工具的作用正是让这个对照实验几行代码就能跑，而不是凭经验假设。")
    print("换模型/数据时把本脚本的 SPARSITY / ROUNDS / 数据集改掉重跑即可。")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "runs", "example3")
    os.makedirs(out_dir, exist_ok=True)
    iteration.to_report("迭代剪枝明细").save(os.path.join(out_dir, "iterative_vs_oneshot.md"))
    report_a.to_report("一次性剪枝明细").save(os.path.join(out_dir, "oneshot_report.md"))
    print("\n报告已写入 %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
