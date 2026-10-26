"""示例 5：把工具接到**你自己的**模型上（最小上手路径）。

本示例不依赖 zoo 里的示例模型，直接定义一个普通的 PyTorch 网络，演示最短流程：

    1. 建通道依赖图，看清哪些层能安全剪、哪些被屏障挡住；
    2. 结构化剪枝到指定比例（自带前向校验）；
    3. 看参数量 / FLOPs 变化，导出检查点与 TorchScript。

运行::

    & C:\\Users\\GuoZhe\\.conda\\envs\\gpupy312\\python.exe examples\\05_quickstart_own_model.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from pruning_toolkit import (
    DependencyGraph,
    StructuredPruner,
    UnstructuredPruner,
    benchmark_latency,
    count_flops,
    count_nonzero_parameters,
    count_parameters,
    export_torchscript,
    load_checkpoint,
    save_checkpoint,
)


class MyNet(nn.Module):
    """一个普通的 MNIST 风格小网络（完全可以换成你自己的模型）。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    torch.manual_seed(0)
    model = MyNet(num_classes=10).eval()

    # 示例输入：形状必须与真实推理一致（只看形状，内容不重要）
    example_inputs = (torch.zeros(2, 1, 28, 28),)

    header("1) 先看通道依赖图")
    graph = DependencyGraph(model, example_inputs)
    print(graph.summary().to_text())
    print("\n只有「可裁剪」的通道组会被剪；被屏障挡住时会写明原因，")
    print("例如拼接分支在不同索引空间之间，工具会选择保守跳过而不是冒险剪坏。")

    header("2) 结构化剪枝：每个通道组裁掉 50%")
    pruner = StructuredPruner(model, example_inputs, criterion="l1", scope="local")
    report = pruner.prune(0.5)
    print(report.to_report().to_text())
    assert report.verified, "前向校验没通过，不应该继续"

    print("\n参数量 %d → %d（下降 %.2f%%）" % (
        report.params_before, report.params_after, 100 * report.param_reduction))
    print("FLOPs    %s → %s（下降 %.2f%%）" % (
        report.flops_before, report.flops_after, 100 * report.flops_reduction))
    latency = benchmark_latency(model, example_inputs)
    print("延迟     p50 = %.4f ms（示例输入 %s）" % (latency["p50_ms"], tuple(example_inputs[0].shape)))

    header("3) 导出与复现")
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = save_checkpoint(
            model, os.path.join(directory, "pruned.pt"), meta={"note": "示例 5 的剪枝结果"}
        )
        script = export_torchscript(model, example_inputs, os.path.join(directory, "pruned.ts"))
        payload = load_checkpoint(checkpoint)
        traced = torch.jit.load(script)
        with torch.no_grad():
            reference = model(*example_inputs)
            exported = traced(*example_inputs)
        print("检查点         : %s" % checkpoint)
        print("  保存的模型元信息：参数量=%s 稀疏度=%.4f" % (
            payload["model_meta"]["params"], payload["model_meta"]["sparsity"]))
        print("TorchScript    : %s" % script)
        print("  导出结果与剪枝后模型一致：%s" % (
            "是" if torch.allclose(reference, exported, atol=1e-5) else "否！"))
        print("\n注意：结构化剪枝改变了张量形状，所以**不能**把剪枝后的 state_dict 直接加载回")
        print("原始结构的模型类。要复用权重，请按剪枝后的结构重建模型（形状可查")
        print("checkpoint 里的 model_meta['layers']），或直接使用导出的 TorchScript；")
        print("非结构化剪枝形状不变，可以直接 load_state_dict，但别忘了一并恢复掩码。")

    header("4) 如果只想做非结构化剪枝（3 行）")
    sparse_model = MyNet(num_classes=10).eval()
    sparse_pruner = UnstructuredPruner(sparse_model, criterion="l1", scope="global")
    sparse_report = sparse_pruner.prune(0.8)
    print("稀疏度=%.4f 非零参数=%d/%d（稠密参数量与 FLOPs 不变）" % (
        sparse_report.achieved_sparsity,
        count_nonzero_parameters(sparse_model),
        count_parameters(sparse_model),
    ))

    header("5) 接自己的模型时的注意事项")
    print("""\
* 模型要能被 torch.fx 追踪：forward 里不要有依赖张量数值的 if/while，
  也不要用 fx 不支持的 Python 结构；否则 DependencyGraph 会抛 UnsupportedGraphError。
* 自定义算子（fx 不认识的函数/方法）会被当作「屏障」保守跳过，
  用 graph.explain() / graph.summary() 能看到是哪个节点挡住了传播。
* 拼接（torch.cat）两侧若来自不同索引空间，相关通道组会被跳过；
  这是为了保证剪枝后前向一定正确，而不是工具「做不到」。
* 用一小批真实形状的数据做 example_inputs；形状不对会导致通道维定位失败。
* 结构化剪枝会就地改模型结构，想保留基线请先 copy.deepcopy。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
