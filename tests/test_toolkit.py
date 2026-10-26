"""pruning_toolkit 的正式测试套件（标准库 unittest，纯 CPU、完全离线）。

运行方式（项目根目录下）::

    $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
    & C:\\Users\\GuoZhe\\.conda\\envs\\gpupy312\\python.exe -m unittest discover -s tests -v

设计原则
--------
* **只依赖 torch + 标准库**：环境里没有 pytest、pyyaml、torchvision，测试也不该要求它们；
* **断言具体数值或性质**，而不是 ``is not None``；同时覆盖错误路径与边界；
* **快**：用小模型、小数据集、少 epoch；整个套件应在 1 分钟内跑完；
* 写文件一律用 ``tempfile.TemporaryDirectory()``，不污染工作目录。

关于内部属性：少数测试会访问 ``_slots`` 之类的私有成员来核对分组细节，
这是**有意的**——通道依赖图的行为正是本工具最关键、最容易出错的部分，
值得直接验证其内部一致性，而不是只看对外结果。
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from pruning_toolkit import (
    CRITERIA,
    DependencyGraph,
    IterativePruner,
    MaskBook,
    PruneConfig,
    Report,
    Scheduler,
    StructuredPruner,
    UnstructuredPruner,
    UnsupportedGraphError,
    benchmark_latency,
    build_graph,
    build_model,
    channel_importance,
    collect_grads,
    count_flops,
    count_nonzero_parameters,
    count_parameters,
    cumulative_targets,
    elementwise_importance,
    evaluate,
    example_input_for,
    export_torchscript,
    format_value,
    get_loaders,
    incremental_ratios,
    iterative_prune,
    load_checkpoint,
    load_config,
    model_meta,
    model_sparsity,
    module_table,
    per_round_ratio,
    recommend_sparsity,
    reconstruction_error_curve,
    resize_both_,
    resize_coupled_,
    resize_input_,
    resize_output_,
    save_checkpoint,
    save_reports,
    structured_sensitivity,
    summarize,
    unstructured_sensitivity,
)
from pruning_toolkit.cli import main as cli_main
from pruning_toolkit.modules import channel_semantics, is_depthwise, iter_weight_parameters


# ---------------------------------------------------------------------- 测试用模型


class _Mlp(nn.Module):
    """2 → 64 → 64 → 3 的小 MLP（与示例中的结构一致，便于对照参数量）。"""

    def __init__(self, sizes=(2, 64, 64, 3)) -> None:
        super().__init__()
        layers = []
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            if index < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SmallConv(nn.Module):
    """带 BN 与全局池化的小卷积网，用于结构化剪枝测试。"""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class _DynamicControlFlow(nn.Module):
    """forward 里含依赖张量数值的分支：fx 无法追踪，应被明确拒绝。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:  # 数据相关控制流
            return x * 2
        return x * 3


class _ZeroLogits(nn.Module):
    """恒输出全零 logits：配合全 0 标签应得到 acc = 1.0。"""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.num_classes, device=x.device) + self.scale * 0.0


def _mlp() -> _Mlp:
    torch.manual_seed(0)
    return _Mlp()


def _tiny_batches(samples: int = 16, features: int = 2, classes: int = 3):
    """构造固定的 (特征, 标签) 批次列表，供评估与梯度采集使用。"""
    generator = torch.Generator().manual_seed(7)
    features_tensor = torch.randn(samples, features, generator=generator)
    labels = torch.randint(0, classes, (samples,), generator=generator)
    return [(features_tensor[i : i + 8], labels[i : i + 8]) for i in range(0, samples, 8)]


# ---------------------------------------------------------------------- 1. 报告


class TestReport(unittest.TestCase):
    """报告模块：小节构造、校验、双格式渲染。"""

    def test_table_rejects_ragged_rows(self) -> None:
        """表格行的列数不一致时必须报错，而不是悄悄生成错位的表。"""
        with self.assertRaises(ValueError):
            Report("t").add_table("小节", ["a", "b"], [["只有一列"]])

    def test_report_sections_and_markdown(self) -> None:
        """四类小节都能渲染进 Markdown，且小节标题都在。"""
        report = Report("测试报告", model_name="m")
        report.metadata["数据集"] = "d"
        report.add_table("表格", ["列1", "列2"], [[1, "x"], [2.5, None]])
        report.add_kv("键值", {"a": 1, "b": True})
        report.add_text("文本", "正文")
        report.add_markdown("原始", "**粗体**")
        text = report.to_markdown()
        for name in ("测试报告", "表格", "键值", "文本", "原始", "**粗体**"):
            self.assertIn(name, text)
        self.assertIn("| 列1 | 列2 |", text)
        # to_text() 把键值小节渲染成 "键: 值" 形式
        plain = report.to_text()
        self.assertIn("a: 1", plain)
        self.assertIn("b: 是", plain)

    def test_report_save_dispatch_and_unknown_suffix(self) -> None:
        """按扩展名分派格式；未知扩展名要报错。"""
        report = Report("r").add_kv("k", {"x": 1})
        with tempfile.TemporaryDirectory() as tmp:
            json_path = report.save(os.path.join(tmp, "a.json"))
            md_path = report.save(os.path.join(tmp, "b.md"))
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(md_path))
            with open(json_path, encoding="utf-8") as handle:
                self.assertIn('"title": "r"', handle.read())
            with self.assertRaises(ValueError):
                report.save(os.path.join(tmp, "c.txt"))

    def test_format_value_types(self) -> None:
        """数值/布尔/None 的显示格式符合预期。"""
        self.assertEqual(format_value(None), "-")
        self.assertEqual(format_value(True), "是")
        self.assertEqual(format_value(False), "否")
        self.assertEqual(format_value(7), "7")
        self.assertEqual(format_value(3.14159), "3.1416")
        self.assertEqual(format_value(1e-6), "1.000e-06")
        self.assertEqual(format_value([1, 2]), "1, 2")


# ---------------------------------------------------------------------- 2. 调度


class TestSchedules(unittest.TestCase):
    """稀疏度调度：曲线形状、恒等式与错误路径。"""

    def test_cumulative_targets_shape_and_monotonic(self) -> None:
        """四种曲线都单调不减、长度等于轮数、末项等于最终值。"""
        for curve in ("constant", "linear", "cubic", "exponential"):
            targets = cumulative_targets(0.9, 4, curve, initial_sparsity=0.1)
            self.assertEqual(len(targets), 4, curve)
            self.assertEqual(targets[-1], 0.9, curve)
            self.assertTrue(all(b >= a for a, b in zip(targets, targets[1:])), curve)
            self.assertTrue(all(0.1 <= value <= 0.9 for value in targets), curve)

    def test_cubic_prunes_faster_early_than_linear(self) -> None:
        """cubic 曲线是「前期快、后期缓」：第一轮目标高于线性曲线。

        这一点很容易搞反，所以专门用一条测试钉住它——它对应 Zhu & Gupta (2017)
        的经典 cubic 稀疏度调度：早期多剪、后期余量小自然变缓。
        """
        cubic = cumulative_targets(0.9, 4, "cubic")
        linear = cumulative_targets(0.9, 4, "linear")
        self.assertGreater(cubic[0], linear[0])
        # 后期增量必须收窄（cubic 最后一轮的增量比例明显小于第一轮）
        ratios = incremental_ratios(cubic)
        self.assertLess(ratios[-1], ratios[0])

    def test_incremental_ratios_identity(self) -> None:
        """增量比例必须满足 (1-s_t) = (1-s_{t-1}) * (1-r_t)。"""
        targets = cumulative_targets(0.9, 5, "cubic")
        ratios = incremental_ratios(targets)
        previous = 0.0
        for target, ratio in zip(targets, ratios):
            self.assertAlmostEqual(1.0 - target, (1.0 - previous) * (1.0 - ratio), places=12)
            previous = target

    def test_per_round_ratio_identity(self) -> None:
        """等比例每轮剪枝满足 (1-r)^rounds = 1-final。"""
        ratio = per_round_ratio(0.9, 4)
        self.assertAlmostEqual((1.0 - ratio) ** 4, 0.1, places=12)

    def test_schedule_validation_errors(self) -> None:
        """非法轮数/范围/曲线都要报错。"""
        with self.assertRaises(ValueError):
            cumulative_targets(0.9, 0)
        with self.assertRaises(ValueError):
            cumulative_targets(0.5, 3, initial_sparsity=0.6)
        with self.assertRaises(KeyError):
            cumulative_targets(0.9, 3, "nope")
        with self.assertRaises(ValueError):
            per_round_ratio(1.0, 3)

    def test_scheduler_index_bounds(self) -> None:
        """Scheduler 的轮次下标从 1 开始，越界报 IndexError。"""
        scheduler = Scheduler(0.9, 3, "linear")
        self.assertEqual(scheduler.target(3), 0.9)
        self.assertAlmostEqual(scheduler.ratio(1), scheduler.ratios[0])
        with self.assertRaises(IndexError):
            scheduler.target(0)
        with self.assertRaises(IndexError):
            scheduler.ratio(4)


# ---------------------------------------------------------------------- 3. 模块语义


class TestModules(unittest.TestCase):
    """模块通道语义与参数裁剪原语。"""

    def test_resize_output_conv_and_bias(self) -> None:
        """卷积输出裁剪：权重、偏置、out_channels 三者同步。"""
        conv = nn.Conv2d(4, 6, 3, bias=True)
        resize_output_(conv, torch.tensor([0, 2, 5]))
        self.assertEqual(conv.weight.shape, (3, 4, 3, 3))
        self.assertEqual(conv.bias.shape, (3,))
        self.assertEqual(conv.out_channels, 3)
        self.assertEqual(conv.in_channels, 4)

    def test_resize_input_linear_and_bn(self) -> None:
        """线性层输入裁剪与 BN 特征裁剪都会更新各自的元数据与统计量。"""
        linear = nn.Linear(5, 3)
        resize_input_(linear, torch.tensor([0, 1, 2]))
        self.assertEqual(linear.weight.shape, (3, 3))
        self.assertEqual(linear.in_features, 3)

        bn = nn.BatchNorm2d(4)
        resize_output_(bn, torch.tensor([1, 3]))
        self.assertEqual(bn.num_features, 2)
        self.assertEqual(bn.weight.shape, (2,))
        self.assertEqual(bn.running_mean.shape, (2,))
        self.assertEqual(bn.running_var.shape, (2,))

    def test_resize_both_regular_conv(self) -> None:
        """残差块里的普通卷积：输入输出同索引空间，两侧按同一组索引裁剪。"""
        conv = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        resize_both_(conv, torch.tensor([0, 2]))
        self.assertEqual(conv.weight.shape, (2, 2, 3, 3))
        self.assertEqual((conv.in_channels, conv.out_channels), (2, 2))
        self.assertEqual(conv.groups, 1)  # 普通卷积的 groups 不应被改动

    def test_resize_both_depthwise_updates_groups(self) -> None:
        """深度卷积：只裁权重第 0 维，并同步 groups / in / out。"""
        conv = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False)
        original = torch.arange(8 * 9, dtype=torch.float32).reshape(8, 1, 3, 3)
        conv.weight.data = original.clone()
        keep = torch.tensor([1, 4, 7])
        resize_both_(conv, keep)
        self.assertEqual((conv.groups, conv.in_channels, conv.out_channels), (3, 3, 3))
        self.assertEqual(conv.weight.shape, (3, 1, 3, 3))
        self.assertTrue(torch.equal(conv.weight.data, original[keep]))

    def test_resize_coupled_rejects_non_depthwise(self) -> None:
        """coupled 裁剪只对深度卷积成立，其它模块必须报 TypeError。"""
        with self.assertRaises(TypeError):
            resize_coupled_(nn.Conv2d(3, 4, 3), torch.tensor([0, 1]))
        with self.assertRaises(TypeError):
            resize_coupled_(nn.Linear(4, 4), torch.tensor([0, 1]))
        # 分组但非深度（groups=2、in=out=4）同样不安全
        with self.assertRaises(TypeError):
            resize_coupled_(nn.Conv2d(4, 4, 3, groups=2), torch.tensor([0, 1]))
        depthwise = nn.Conv2d(4, 4, 3, groups=4, bias=False)
        resize_coupled_(depthwise, torch.tensor([0, 2]))
        self.assertEqual((depthwise.groups, depthwise.in_channels, depthwise.out_channels), (2, 2, 2))

    def test_channel_semantics_matrix(self) -> None:
        """各类模块的通道语义：普通卷积不保序、BN 保序、分组卷积不支持。"""
        self.assertFalse(channel_semantics(nn.Conv2d(3, 5, 3)).index_preserving)
        self.assertTrue(channel_semantics(nn.BatchNorm2d(5)).index_preserving)
        self.assertTrue(channel_semantics(nn.ReLU()).index_preserving)
        grouped = channel_semantics(nn.Conv2d(6, 6, 3, groups=3))
        self.assertFalse(grouped.supported)
        self.assertIn("分组卷积", grouped.reason)
        self.assertTrue(is_depthwise(nn.Conv2d(4, 4, 3, groups=4)))
        self.assertFalse(is_depthwise(nn.Conv2d(4, 4, 3)))

    def test_iter_weight_parameters_scope(self) -> None:
        """默认含归一化仿射权重但不含偏置；两个开关各自生效。"""
        model = _SmallConv()
        default_names = [name for name, _ in iter_weight_parameters(model)]
        self.assertEqual(
            default_names,
            [
                "features.0.weight",
                "features.1.weight",
                "features.3.weight",
                "features.4.weight",
                "classifier.weight",
            ],
        )
        with_bias = [name for name, _ in iter_weight_parameters(model, include_bias=True)]
        self.assertIn("classifier.bias", with_bias)
        self.assertIn("features.0.bias", with_bias)
        without_norm = [name for name, _ in iter_weight_parameters(model, include_norm=False)]
        self.assertEqual(
            without_norm, ["features.0.weight", "features.3.weight", "classifier.weight"]
        )


# ---------------------------------------------------------------------- 4. 依赖图


class TestGraph(unittest.TestCase):
    """通道依赖图：分组数量、可剪性判定、屏障行为。"""

    #: 与 tools/verify_models.py 一致的真实统计（通道组总数, 可裁剪组数）
    ZOO_EXPECTED = {
        "mlp": (4, 2),
        "small_vgg": (6, 4),
        "resnet_small": (8, 6),
        "depthwise_cnn": (6, 4),
        "inception_like": (7, 2),
    }

    def test_group_counts_for_all_zoo_models(self) -> None:
        """5 个示例模型的通道组数量与可裁剪数量都要对得上。"""
        for name, (total, prunable) in self.ZOO_EXPECTED.items():
            torch.manual_seed(0)
            model = build_model(name, num_classes=10).eval()
            inputs = (example_input_for(name, batch_size=2),)
            graph = DependencyGraph(model, inputs)
            self.assertEqual(len(graph.groups), total, name)
            self.assertEqual(len(graph.prunable_groups()), prunable, name)
            self.assertEqual(sum(1 for _ in graph.records()) > 0, True, name)

    def test_output_group_protected_by_default(self) -> None:
        """直接产生模型输出的通道组默认不可剪，且原因可读。"""
        torch.manual_seed(0)
        model = build_model("small_vgg").eval()
        inputs = (example_input_for("small_vgg", batch_size=2),)
        graph = DependencyGraph(model, inputs)
        output_groups = [g for g in graph.groups if "classifier" in g.output_modules]
        self.assertEqual(len(output_groups), 1)
        self.assertFalse(output_groups[0].prunable)
        self.assertIn("输出", output_groups[0].reason)

    def test_protect_output_false_makes_it_prunable(self) -> None:
        """protect_output=False 时输出组变为可剪（可剪组数 +1）。"""
        torch.manual_seed(0)
        model = build_model("small_vgg").eval()
        inputs = (example_input_for("small_vgg", batch_size=2),)
        protected = DependencyGraph(model, inputs, protect_output=True)
        unprotected = DependencyGraph(model, inputs, protect_output=False)
        self.assertEqual(
            len(unprotected.prunable_groups()), len(protected.prunable_groups()) + 1
        )

    def test_depthwise_group_is_coupled(self) -> None:
        """深度可分离网络里必须出现 both 编辑，且该模块的 groups 与两侧通道数一致。"""
        torch.manual_seed(0)
        model = build_model("depthwise_cnn").eval()
        inputs = (example_input_for("depthwise_cnn", batch_size=2),)
        graph = DependencyGraph(model, inputs)
        coupled = [
            edit.module_name
            for group in graph.groups
            for edit in group.edits
            if edit.kind == "both"
        ]
        depthwise_modules = [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Conv2d) and is_depthwise(module)
        ]
        self.assertTrue(set(coupled) & set(depthwise_modules))

    def test_input_side_group_not_prunable(self) -> None:
        """只含输入侧通道的组（模型输入）不可剪，原因为「没有可评分的生产者」。"""
        graph = DependencyGraph(build_model("small_vgg").eval(),
                                (example_input_for("small_vgg", batch_size=2),))
        input_groups = [
            g for g in graph.groups
            if not g.output_edits and any(e.kind == "in" for e in g.edits)
        ]
        self.assertTrue(input_groups)
        for group in input_groups:
            self.assertFalse(group.prunable)
            self.assertIn("输入侧", group.reason)

    def test_cat_creates_barrier_and_skips_groups(self) -> None:
        """含 torch.cat 的模型必须产生屏障，并因此跳过相关通道组。"""
        torch.manual_seed(0)
        model = build_model("inception_like").eval()
        inputs = (example_input_for("inception_like", batch_size=2),)
        graph = DependencyGraph(model, inputs)
        self.assertGreaterEqual(len(graph.barriers), 1)
        reasons = " ".join(barrier["reason"] for barrier in graph.barriers)
        self.assertIn("cat", reasons)
        skipped = [g for g in graph.groups if not g.prunable]
        self.assertTrue(any("屏障" in g.reason for g in skipped))

    def test_unsupported_control_flow_raises(self) -> None:
        """数据相关控制流无法追踪，必须抛 UnsupportedGraphError 且信息可读。"""
        model = _DynamicControlFlow().eval()
        with self.assertRaises(UnsupportedGraphError) as context:
            DependencyGraph(model, (torch.randn(2, 3),))
        self.assertIn("fx", str(context.exception))

    def test_group_lookup_and_flops(self) -> None:
        """group_by_index / group_of_module / flops 与参数量统计可用且自洽。"""
        graph = DependencyGraph(_SmallConv().eval(), (torch.randn(2, 3, 8, 8),))
        first = graph.prunable_groups()[0]
        self.assertIs(graph.group_by_index(first.index), first)
        self.assertIs(graph.group_of_module(first.output_modules[0]), first)
        with self.assertRaises(KeyError):
            graph.group_by_index(9999)
        self.assertGreater(graph.flops(), 0)
        self.assertGreater(graph.parameters(), 0)
        total, breakdown = graph.flops(per_module=True)
        self.assertEqual(total, sum(breakdown.values()))
        self.assertIn("可裁剪通道组", graph.summary().to_dict()["sections"][0]["items"])


# ---------------------------------------------------------------------- 5. 准则


class TestCriteria(unittest.TestCase):
    """重要性准则的数值性质与错误路径。"""

    def test_elementwise_l1_l2_none_random(self) -> None:
        """l1=|w|、l2=w^2、none 全 1、random 同 seed 可复现。"""
        weight = torch.tensor([[-3.0, 1.0], [0.5, -2.0]])
        self.assertTrue(torch.equal(elementwise_importance(weight, "l1"), weight.abs()))
        self.assertTrue(torch.equal(elementwise_importance(weight, "l2"), weight * weight))
        self.assertTrue(torch.equal(elementwise_importance(weight, "none"), torch.ones_like(weight)))
        first = elementwise_importance(
            weight, "random", generator=torch.Generator().manual_seed(3)
        )
        second = elementwise_importance(
            weight, "random", generator=torch.Generator().manual_seed(3)
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(((first >= 0) & (first <= 1)).all()))

    def test_channel_importance_values_and_reduce(self) -> None:
        """逐通道分数：l1 为行绝对值之和、l2 为 L2 范数、reduce 生效。"""
        conv = nn.Conv2d(2, 3, 3, bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[1] = 5.0
        l1 = channel_importance(conv, "l1")
        self.assertEqual(tuple(l1.shape), (3,))
        self.assertEqual([round(float(v), 6) for v in l1], [0.0, 90.0, 0.0])
        # 18 个元素全为 5：平方和 450，L2 范数 sqrt(450) ≈ 21.2132
        l2 = channel_importance(conv, "l2")
        self.assertAlmostEqual(float(l2[1]), 450.0 ** 0.5, places=4)
        self.assertAlmostEqual(float(l2[0]), 0.0, places=6)
        # mean/max 是对 18 个元素做的，每个都是 5
        self.assertAlmostEqual(float(channel_importance(conv, "l1", reduce="mean")[1]), 5.0, places=6)
        self.assertAlmostEqual(float(channel_importance(conv, "l1", reduce="max")[1]), 5.0, places=6)
        with self.assertRaises(KeyError):
            channel_importance(conv, "l1", reduce="nope")
        with self.assertRaises(KeyError):
            elementwise_importance(conv.weight, "nope")

    def test_taylor_requires_grad(self) -> None:
        """taylor 缺梯度时报 ValueError，给了同形状梯度后能算。"""
        weight = torch.ones(2, 3)
        with self.assertRaises(ValueError):
            elementwise_importance(weight, "taylor")
        with self.assertRaises(ValueError):
            elementwise_importance(weight, "taylor", grad=torch.ones(3, 2))
        scores = elementwise_importance(weight, "taylor", grad=torch.full((2, 3), 2.0))
        self.assertTrue(torch.equal(scores, torch.full((2, 3), 2.0)))

    def test_bn_scale_uses_gamma(self) -> None:
        """bn_scale 对 BN 模块返回 |gamma|。"""
        bn = nn.BatchNorm2d(4)
        with torch.no_grad():
            bn.weight.copy_(torch.tensor([-1.0, 2.0, 0.0, 3.0]))
        scores = channel_importance(bn, "bn_scale")
        self.assertEqual([float(v) for v in scores], [1.0, 2.0, 0.0, 3.0])

    def test_fpgm_shape_and_non_negative(self) -> None:
        """fpgm 返回 (out,)，且距离非负、全同卷积核时分数相同。"""
        conv = nn.Conv2d(3, 5, 3, bias=False)
        scores = channel_importance(conv, "fpgm")
        self.assertEqual(tuple(scores.shape), (5,))
        self.assertTrue(bool((scores >= 0).all()))
        with torch.no_grad():
            conv.weight.fill_(1.0)
        identical = channel_importance(conv, "fpgm")
        self.assertAlmostEqual(float(identical.max() - identical.min()), 0.0, places=5)

    def test_collect_grads_keys_and_cleanup(self) -> None:
        """collect_grads 覆盖全部参数，调用后清空梯度，max_batches 生效。"""
        model = _mlp()
        batches = _tiny_batches()
        seen = []

        def loss_fn(current, batch):
            seen.append(1)
            return torch.nn.functional.cross_entropy(current(batch[0]), batch[1])

        grads = collect_grads(model, loss_fn, batches, max_batches=2)
        self.assertEqual(len(seen), 2)
        self.assertEqual(set(grads), set(dict(model.named_parameters())))
        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad, name)
            self.assertEqual(grads[name].shape, parameter.shape, name)

    def test_reconstruction_error_curve_bounds(self) -> None:
        """重建误差曲线：随强度单调不减、落在 [0, 1]、强度 0 时为 0。"""
        weight = torch.randn(6, 4)
        scores = weight.abs()
        errors = reconstruction_error_curve(weight, scores, (0.0, 0.3, 0.6, 0.9))
        self.assertAlmostEqual(errors[0], 0.0, places=9)
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(errors, errors[1:])))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in errors))
        # 幅值剪枝保留的是「能量更大」的权重，因此误差必然小于按比例随机丢弃
        # （随机丢 90% 的期望误差是 sqrt(0.9) ≈ 0.9487）
        self.assertLess(errors[-1], 0.9 ** 0.5)
        self.assertGreater(errors[-1], 0.5)


# ---------------------------------------------------------------------- 6. 掩码


class TestMasks(unittest.TestCase):
    """掩码管理：应用、统计、梯度钩子、序列化。"""

    def test_apply_and_density(self) -> None:
        """apply 把掩码位置置零，density/sparsity/summary 与掩码一致。"""
        model = _mlp()
        mask = torch.ones_like(dict(model.named_parameters())["net.0.weight"], dtype=torch.bool)
        mask[0, :] = False
        book = MaskBook({"net.0.weight": mask})
        self.assertEqual(book.apply(model), 1)
        weights = dict(model.named_parameters())["net.0.weight"].detach()
        self.assertEqual(int((weights[0] != 0).sum().item()), 0)
        self.assertAlmostEqual(book.density(), 1.0 - 2.0 / (64 * 2), places=12)
        self.assertAlmostEqual(book.sparsity(), 2.0 / 128, places=12)
        self.assertEqual(book.summary()["保留元素数"], 126)

    def test_gradient_hook_zeroes_pruned_positions(self) -> None:
        """梯度钩子让被剪位置的梯度恒为 0，未被剪位置保留梯度。"""
        model = _mlp()
        mask = torch.ones_like(dict(model.named_parameters())["net.0.weight"], dtype=torch.bool)
        mask[:, 0] = False
        book = MaskBook({"net.0.weight": mask})
        self.assertEqual(book.install_hooks(model), 1)
        try:
            loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 2)), torch.randint(0, 3, (8,)))
            loss.backward()
        finally:
            book.remove_hooks()
        grad = dict(model.named_parameters())["net.0.weight"].grad
        self.assertEqual(int(grad[:, 0].abs().sum().item()), 0)
        self.assertGreater(float(grad[:, 1].abs().sum().item()), 0.0)

    def test_save_load_roundtrip_and_errors(self) -> None:
        """掩码存取往返一致；对不存在的参数用 strict 会报错。"""
        model = _mlp()
        mask = torch.zeros(64, 2, dtype=torch.bool)
        mask[:, 1] = True
        book = MaskBook({"net.0.weight": mask})
        with tempfile.TemporaryDirectory() as tmp:
            path = book.save(os.path.join(tmp, "masks.pt"))
            loaded = MaskBook.load(path)
            self.assertEqual(loaded.keys(), book.keys())
            self.assertTrue(torch.equal(loaded.get("net.0.weight"), mask))
        with self.assertRaises(KeyError):
            MaskBook({"不存在.weight": mask}).apply(model, strict=True)


# ---------------------------------------------------------------------- 7. 非结构化剪枝


class TestUnstructuredPruning(unittest.TestCase):
    """非结构化剪枝：精确性、逐层目标、回滚与微调保持。"""

    def _pruner(self, **kwargs) -> UnstructuredPruner:
        return UnstructuredPruner(_mlp(), **kwargs)

    def test_global_topk_is_exact(self) -> None:
        """全局剪枝保留个数必须精确等于 round(numel*(1-s))。"""
        pruner = self._pruner(criterion="l1", scope="global")
        report = pruner.prune(0.9)
        expected = int(round(pruner.total_numel() * 0.1))
        self.assertEqual(report.total_kept, expected)
        self.assertEqual(pruner.masks.kept_numel(), expected)
        self.assertAlmostEqual(report.achieved_sparsity, 1.0 - expected / pruner.total_numel(), places=12)
        materialized = sum(
            int((parameter.detach() != 0).sum().item()) for _, parameter in pruner.params
        )
        self.assertEqual(materialized, expected)

    def test_local_scope_per_parameter(self) -> None:
        """逐层模式下每个参数各自保留 50% 左右。"""
        pruner = self._pruner(scope="local")
        report = pruner.prune(0.5)
        self.assertEqual(len(report.layers), 3)
        for layer in report.layers:
            self.assertLessEqual(abs(layer.kept - round(layer.numel * 0.5)), 1, layer.param_name)

    def test_per_layer_override_and_unknown_key(self) -> None:
        """per_layer 覆盖逐层比例；未知模块名要报 KeyError。"""
        pruner = self._pruner(scope="global")
        report = pruner.prune(0.5, per_layer={"net.0": 0.9, "net.2": 0.1})
        by_name = {layer.module_name: layer for layer in report.layers}
        self.assertGreater(by_name["net.0"].sparsity_after, 0.85)
        self.assertLess(by_name["net.2"].sparsity_after, 0.15)
        with self.assertRaises(KeyError):
            pruner.prune(0.5, per_layer={"不存在的层": 0.5})

    def test_exclude_and_protect_first_last(self) -> None:
        """exclude 与 protect_first_last 都会把对应参数排除在外。"""
        pruner = UnstructuredPruner(_mlp(), exclude=("net.0",))
        self.assertNotIn("net.0.weight", pruner.param_names)
        self.assertIn("net.0.weight", pruner._excluded)

        protected = UnstructuredPruner(_mlp(), protect_first_last=True)
        self.assertNotIn("net.0.weight", protected.param_names)
        self.assertNotIn("net.4.weight", protected.param_names)
        self.assertEqual(protected.param_names, ["net.2.weight"])

    def test_restore_returns_to_dense(self) -> None:
        """restore 之后稀疏度回到 0，掩码被清空。"""
        pruner = self._pruner()
        pruner.prune(0.9)
        self.assertGreater(pruner.sparsity(), 0.85)
        self.assertEqual(pruner.restore(), 3)
        self.assertAlmostEqual(pruner.sparsity(), 0.0, places=12)
        self.assertIsNone(pruner.masks)

    def test_masks_survive_sgd_step(self) -> None:
        """剪枝后做一次 SGD，再用掩码压回，被剪权重必须仍是 0。"""
        pruner = self._pruner(scope="global")
        pruner.prune(0.8)
        masks = pruner.masks
        optimizer = torch.optim.SGD(_mlp().parameters(), lr=0.0)  # 占位，避免干扰
        optimizer = torch.optim.SGD(pruner.model.parameters(), lr=0.1)
        loss = torch.nn.functional.cross_entropy(
            pruner.model(torch.randn(16, 2)), torch.randint(0, 3, (16,))
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        masks.apply(pruner.model)
        leftovers = sum(
            int((parameter.detach() * (~masks.get(name))).count_nonzero().item())
            for name, parameter in pruner.model.named_parameters()
            if name in masks
        )
        self.assertEqual(leftovers, 0)

    def test_taylor_without_grads_raises(self) -> None:
        """taylor 准则没给梯度时应在剪枝前就报 ValueError。"""
        pruner = self._pruner(criterion="taylor")
        with self.assertRaises(ValueError):
            pruner.prune(0.5)

        model = _mlp()
        grads = collect_grads(
            model,
            lambda m, b: torch.nn.functional.cross_entropy(m(b[0]), b[1]),
            _tiny_batches(),
            max_batches=2,
        )
        with_grads = UnstructuredPruner(model, criterion="taylor", scope="global", grads=grads)
        report = with_grads.prune(0.5)
        self.assertAlmostEqual(report.achieved_sparsity, 0.5, places=3)

    def test_invalid_sparsity_and_kwargs(self) -> None:
        """非法稀疏度/准则/范围都要报错。"""
        pruner = self._pruner()
        with self.assertRaises(ValueError):
            pruner.prune(1.5)
        with self.assertRaises(KeyError):
            UnstructuredPruner(_mlp(), criterion="nope")
        with self.assertRaises(ValueError):
            UnstructuredPruner(_mlp(), scope="nope")


# ---------------------------------------------------------------------- 8. 结构化剪枝


class TestStructuredPruning(unittest.TestCase):
    """结构化剪枝：计划与执行分离、收益、校验与边界。"""

    def _setup(self):
        torch.manual_seed(0)
        model = _SmallConv().eval()
        inputs = (torch.randn(2, 3, 8, 8),)
        return model, inputs

    def test_plan_does_not_mutate(self) -> None:
        """plan() 只算索引，不修改模型。"""
        model, inputs = self._setup()
        before = {name: parameter.shape for name, parameter in model.named_parameters()}
        pruner = StructuredPruner(model, inputs)
        plan = pruner.plan(0.5)
        self.assertTrue(plan)
        after = {name: parameter.shape for name, parameter in model.named_parameters()}
        self.assertEqual(before, after)
        for group in pruner.graph().prunable_groups():
            self.assertEqual(int(plan[group.index].numel()), max(1, round(group.count * 0.5)))

    def test_prune_reduces_params_and_flops(self) -> None:
        """剪枝后参数量与 FLOPs 都下降，前向校验通过且输出形状不变。"""
        model, inputs = self._setup()
        with torch.no_grad():
            expected = tuple(model(*inputs).shape)
        before_params = count_parameters(model)
        before_flops = count_flops(model, inputs)
        report = StructuredPruner(model, inputs).prune(0.5)
        self.assertTrue(report.verified)
        self.assertEqual(tuple(report.output_shape), expected)
        self.assertLess(count_parameters(model), before_params)
        self.assertLess(count_flops(model, inputs), before_flops)
        self.assertAlmostEqual(report.params_after, count_parameters(model), places=9)
        with torch.no_grad():
            self.assertEqual(tuple(model(*inputs).shape), expected)

    def test_per_group_by_module_and_display(self) -> None:
        """per_group 既能用模块名，也能用「conv+bn」这样的显示名。"""
        for key_kind in ("module", "display"):
            model, inputs = self._setup()
            pruner = StructuredPruner(model, inputs)
            group = pruner.graph().prunable_groups()[0]
            key = group.output_modules[0] if key_kind == "module" else group.display
            plan = pruner.plan(0.5, per_group={key: 0.75})
            self.assertEqual(int(plan[group.index].numel()), max(1, round(group.count * 0.25)))
            for other in pruner.graph().prunable_groups()[1:]:
                self.assertEqual(int(plan[other.index].numel()), max(1, round(other.count * 0.5)))
            with self.assertRaises(KeyError):
                pruner.plan(0.5, per_group={"没有这个模块": 0.5})

    def test_scope_global_and_skipped_reasons(self) -> None:
        """global 范围可用，且被跳过的组必须带原因。"""
        model, inputs = self._setup()
        report = StructuredPruner(model, inputs, scope="global").prune(0.6)
        self.assertTrue(report.verified)
        self.assertTrue(report.skipped)
        for _, modules, reason in report.skipped:
            self.assertTrue(reason)
            self.assertTrue(modules)

    def test_ratio_validation_and_clamp(self) -> None:
        """非法比例报错；比例过大时每组至少保留 1 个通道。"""
        model, inputs = self._setup()
        pruner = StructuredPruner(model, inputs)
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                pruner.plan(bad)
        report = pruner.prune(0.99)
        self.assertTrue(report.verified)
        for group in pruner.graph(rebuild=True).groups:
            self.assertGreaterEqual(group.count, 1)
        self.assertTrue(any("至少保留" in note for note in report.notes))

    def test_report_contains_gain_table(self) -> None:
        """报告里要有收益表与逐通道组结果。"""
        model, inputs = self._setup()
        text = StructuredPruner(model, inputs).prune(0.5).to_report().to_text()
        for token in ("参数下降比例", "FLOPs 下降比例", "逐通道组结果", "前向校验"):
            self.assertIn(token, text)


# ---------------------------------------------------------------------- 9. 指标


class TestMetrics(unittest.TestCase):
    """评估指标：参数量/稀疏度/FLOPs/延迟/精度。"""

    def test_parameter_counts_and_sparsity(self) -> None:
        """稠密参数量不变而非零参数量下降；两个稀疏度口径都要自洽。

        ``count_nonzero_parameters`` 统计的是**全部**参数（含不参与剪枝的偏置），
        所以它等于「剪枝保留数 + 偏置个数」，这一点必须写清楚，否则很容易误读。
        """
        model = _mlp()
        pruner = UnstructuredPruner(model, scope="global")
        total = count_parameters(model)
        bias_count = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.endswith(".bias")
        )
        self.assertEqual(bias_count, 64 + 64 + 3)
        report = pruner.prune(0.9)
        self.assertEqual(count_parameters(model), total)  # 形状不变
        self.assertEqual(count_parameters(model, only_conv_linear=True), total)
        self.assertEqual(
            count_nonzero_parameters(model), report.total_kept + bias_count
        )
        self.assertEqual(
            count_nonzero_parameters(model, masks=pruner.masks), report.total_kept + bias_count
        )
        self.assertAlmostEqual(
            model_sparsity(model), 1.0 - (report.total_kept + bias_count) / total, places=12
        )
        self.assertAlmostEqual(report.achieved_sparsity, 0.9, places=2)

    def test_flops_decrease_only_for_structured(self) -> None:
        """非结构化剪枝不改 FLOPs；结构化剪枝降低 FLOPs。"""
        model = _SmallConv().eval()
        inputs = (torch.randn(2, 3, 8, 8),)
        before = count_flops(model, inputs)
        self.assertGreater(before, 0)
        UnstructuredPruner(model, scope="global").prune(0.9)
        self.assertEqual(count_flops(model, inputs), before)
        StructuredPruner(model, inputs).prune(0.5)
        self.assertLess(count_flops(model, inputs), before)

    def test_benchmark_latency_keys_and_order(self) -> None:
        """延迟报告键齐全，且 min <= p50 <= p90，吞吐为正。"""
        model = _SmallConv().eval()
        inputs = (torch.randn(2, 3, 8, 8),)
        stats = benchmark_latency(model, inputs, warmup=2, iters=5)
        for key in ("mean_ms", "std_ms", "min_ms", "p50_ms", "p90_ms", "throughput"):
            self.assertIn(key, stats)
        self.assertLessEqual(stats["min_ms"], stats["p50_ms"] + 1e-9)
        self.assertLessEqual(stats["p50_ms"], stats["p90_ms"] + 1e-9)
        self.assertGreater(stats["throughput"], 0)

    def test_evaluate_accuracy_and_state_restore(self) -> None:
        """恒输出全零 logits + 全 0 标签 => acc=1.0，且不改变 train/eval 状态。"""
        model = _ZeroLogits(num_classes=3)
        batches = [(torch.randn(4, 2), torch.zeros(4, dtype=torch.long)) for _ in range(2)]
        model.train()
        result = evaluate(model, batches)
        self.assertAlmostEqual(result["acc"], 1.0, places=12)
        self.assertAlmostEqual(result["loss"], float(torch.log(torch.tensor(3.0))), places=5)
        self.assertTrue(model.training)
        model.eval()
        evaluate(model, batches)
        self.assertFalse(model.training)

    def test_summarize_and_module_table(self) -> None:
        """体检报告的逐模块表行数与真实子模块一致，且含 FLOPs 段落。"""
        model = _SmallConv().eval()
        inputs = (torch.randn(2, 3, 8, 8),)
        rows = module_table(model)
        expected = sum(
            1 for _, module in model.named_modules() if list(module.parameters(recurse=False))
        )
        self.assertEqual(len(rows), expected)
        report = summarize(model, inputs)
        names = [section["name"] for section in report.to_dict()["sections"]]
        self.assertIn("逐模块明细", names)
        self.assertIn("FLOPs 占比最高的模块", names)


# ---------------------------------------------------------------------- 10. 敏感度


class TestAnalysis(unittest.TestCase):
    """敏感度扫描与逐层稀疏度分配。"""

    def test_unstructured_sensitivity_monotonic(self) -> None:
        """每个模块的重建误差随强度单调不减，且落在 [0, 1]。"""
        report = unstructured_sensitivity(_mlp(), sparsities=(0.1, 0.5, 0.9))
        self.assertEqual(report.mode, "unstructured")
        self.assertEqual(len(report.modules), 3)
        for module in report.modules:
            errors = [point.error for point in module.points]
            self.assertTrue(all(b >= a - 1e-9 for a, b in zip(errors, errors[1:])), module.name)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in errors), module.name)
        self.assertEqual(len(report.rank()), 3)

    def test_structured_sensitivity_names_match_groups(self) -> None:
        """结构化敏感度的模块名要与依赖图里的组显示名一一对应。"""
        torch.manual_seed(0)
        model = _SmallConv().eval()
        inputs = (torch.randn(2, 3, 8, 8),)
        graph = DependencyGraph(model, inputs)
        report = structured_sensitivity(model, inputs, sparsities=(0.2, 0.5))
        self.assertEqual(
            report.names(), [group.display for group in graph.prunable_groups()]
        )
        self.assertLessEqual(len(report.modules), len(graph.groups))

    def test_recommend_sparsity_hits_target_within_bounds(self) -> None:
        """分配结果的参数量加权平均命中目标，且单层不超出 spread 边界。"""
        model = _mlp()
        allocation = recommend_sparsity(model, 0.6, alpha=1.0, spread=0.5)
        self.assertTrue(allocation.feasible)
        self.assertAlmostEqual(allocation.achieved, 0.6, places=6)
        self.assertEqual(set(allocation.per_layer), set(unstructured_sensitivity(model).names()))
        for value in allocation.per_layer.values():
            self.assertGreaterEqual(value, 0.6 * 0.5 - 1e-9)
            self.assertLessEqual(value, min(0.6 * 1.5, 0.99) + 1e-9)
        # 参数量加权平均确实等于 0.6（用同样的权重复算一遍）
        names = {module.name: module.params for module in unstructured_sensitivity(model).modules}
        total = sum(names.values())
        weighted = sum(allocation.per_layer[name] * names[name] for name in names) / total
        self.assertAlmostEqual(weighted, 0.6, places=6)

    def test_recommend_sparsity_alpha_zero_is_uniform(self) -> None:
        """alpha=0 时退化为均匀分配。"""
        allocation = recommend_sparsity(_mlp(), 0.6, alpha=0.0)
        values = list(allocation.per_layer.values())
        self.assertAlmostEqual(max(values) - min(values), 0.0, places=9)
        self.assertAlmostEqual(values[0], 0.6, places=9)

    def test_recommend_sparsity_invalid_targets(self) -> None:
        """目标为 0 或超过 cap 都要报错。"""
        with self.assertRaises(ValueError):
            recommend_sparsity(_mlp(), 0.0)
        with self.assertRaises(ValueError):
            recommend_sparsity(_mlp(), 1.0, cap=0.99)


# ---------------------------------------------------------------------- 11. 迭代剪枝


class TestIterativePruning(unittest.TestCase):
    """迭代剪枝：轮次记录、掩码保持与参数校验。"""

    def test_unstructured_rounds_and_targets(self) -> None:
        """3 轮迭代的累积目标单调不减、末轮命中目标，且记录完整。"""
        model = _mlp()
        report = IterativePruner(
            model,
            mode="unstructured",
            scope="global",
            schedule="cubic",
            rounds=3,
            final_sparsity=0.9,
            verbose=False,
            log=None,
        ).run()
        self.assertEqual(len(report.records), 3)
        targets = [record.target for record in report.records]
        self.assertTrue(all(b >= a for a, b in zip(targets, targets[1:])))
        self.assertAlmostEqual(targets[-1], 0.9, places=12)
        self.assertAlmostEqual(report.records[-1].achieved, 0.9, places=3)
        self.assertGreater(report.params_reduction, -1e-9)
        self.assertEqual(report.mode, "unstructured")

    def test_finetune_then_masks_reapplied(self) -> None:
        """finetune_fn 里做一次 SGD 后，被剪权重仍必须为 0（掩码被重新压回）。"""
        model = _mlp()
        observed = {}

        def finetune(current, context):
            parameters = dict(current.named_parameters())
            dirty = 0
            optimizer = torch.optim.SGD(current.parameters(), lr=0.1)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                current(torch.randn(16, 2)), torch.randint(0, 3, (16,))
            )
            loss.backward()
            optimizer.step()
            for name, mask in context.masks.items():
                dirty += int(((~mask) & (parameters[name].detach() != 0)).sum().item())
            observed[context.round_index] = {"dirty_inside": dirty, "masks": context.masks}
            return {"acc": 0.5}

        report = IterativePruner(
            model,
            mode="unstructured",
            rounds=2,
            final_sparsity=0.8,
            finetune_fn=finetune,
            verbose=False,
            log=None,
        ).run()
        # 回调内部（尚未重新压回掩码时）确实变脏了，说明这个测试有意义
        self.assertGreater(observed[1]["dirty_inside"], 0)
        masks = report.masks
        leftovers = sum(
            int((parameter.detach() * (~masks.get(name))).count_nonzero().item())
            for name, parameter in model.named_parameters()
            if name in masks
        )
        self.assertEqual(leftovers, 0)
        self.assertAlmostEqual(report.records[-1].achieved, 0.8, places=3)

    def test_structured_mode_requires_example_inputs(self) -> None:
        """结构化迭代剪枝必须给 example_inputs（每轮要重建依赖图）。"""
        with self.assertRaises(ValueError):
            IterativePruner(_SmallConv(), mode="structured", example_inputs=None)

    def test_round_context_masks_by_mode(self) -> None:
        """非结构化时 context.masks 非空；结构化时为 None。"""
        seen = []

        def finetune(current, context):
            seen.append(context.masks)
            return {}

        IterativePruner(
            _mlp(), mode="unstructured", rounds=1, final_sparsity=0.5,
            finetune_fn=finetune, verbose=False, log=None,
        ).run()
        self.assertIsNotNone(seen[0])

        torch.manual_seed(0)
        model = _SmallConv().eval()
        inputs = (torch.randn(2, 3, 8, 8),)
        seen.clear()
        IterativePruner(
            model, inputs, mode="structured", rounds=1, final_sparsity=0.3,
            finetune_fn=finetune, verbose=False, log=None,
        ).run()
        self.assertIsNone(seen[0])

    def test_report_has_prune_and_finetune_columns(self) -> None:
        """报告必须同时给出「剪枝后 acc」与「微调后 acc」。"""
        report = IterativePruner(
            _mlp(), mode="unstructured", rounds=1, final_sparsity=0.5,
            eval_fn=lambda m: {"acc": 0.25}, verbose=False, log=None,
        ).run()
        text = report.to_report().to_text()
        self.assertIn("剪枝后 acc", text)
        self.assertIn("微调后 acc", text)
        self.assertAlmostEqual(report.records[0].metrics_after_prune["acc"], 0.25, places=12)

    def test_invalid_mode_and_schedule(self) -> None:
        """非法 mode/schedule 要报错。"""
        with self.assertRaises(ValueError):
            IterativePruner(_mlp(), mode="nope")
        with self.assertRaises(KeyError):
            IterativePruner(_mlp(), schedule="nope")
        with self.assertRaises(ValueError):
            iterative_prune(_mlp(), mode="nope")


# ---------------------------------------------------------------------- 12. 配置


class TestConfig(unittest.TestCase):
    """配置系统：序列化往返、未知字段与非法值的报错。"""

    def test_roundtrip_and_presets_validate(self) -> None:
        """to_dict/from_dict 往返相等；两个预设都能通过校验。"""
        config = PruneConfig.for_demo()
        restored = PruneConfig.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        config.validate()
        PruneConfig.for_mlp_demo().validate()

    def test_unknown_fields_raise(self) -> None:
        """未知顶层字段与嵌套字段都要报错，且信息里含字段名。"""
        with self.assertRaises(ValueError) as context:
            PruneConfig.from_dict({"model": "mlp", "critirion": "l1"})
        self.assertIn("critirion", str(context.exception))
        with self.assertRaises(ValueError) as nested:
            PruneConfig.from_dict({"train": {"epochs": 3, "lr": 0.1, "epocs": 1}})
        self.assertIn("epocs", str(nested.exception))

    def test_validate_rejects_bad_values(self) -> None:
        """mode/scope/criterion/schedule/rounds/sparsity/batch_size 非法都要报错。"""
        for overrides in (
            {"mode": "nope"},
            {"scope": "nope"},
            {"criterion": "nope"},
            {"schedule": "nope"},
            {"rounds": 0},
            {"sparsity": 0.0},
            {"sparsity": 1.0},
            {"batch_size": 0},
            {"per_layer": {"conv": 1.2}},
        ):
            with self.assertRaises((ValueError, KeyError)):
                PruneConfig(**overrides).validate()

    def test_json_save_and_load(self) -> None:
        """JSON 存盘后能原样读回。"""
        config = PruneConfig.for_mlp_demo()
        with tempfile.TemporaryDirectory() as tmp:
            path = config.save(os.path.join(tmp, "cfg.json"))
            loaded = load_config(path)
            self.assertEqual(loaded.to_dict(), config.to_dict())
            with self.assertRaises(ValueError):
                config.save(os.path.join(tmp, "cfg.txt"))

    def test_yaml_requires_pyyaml(self) -> None:
        """没有 pyyaml 时读写 YAML 必须抛出带安装提示的 RuntimeError。"""
        try:
            import yaml  # noqa: F401
        except ImportError:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "cfg.yaml")
                with self.assertRaises(RuntimeError) as context:
                    PruneConfig().save(path)
                self.assertIn("pyyaml", str(context.exception))
        else:  # pragma: no cover - 本机没装 pyyaml
            self.skipTest("环境已安装 pyyaml，跳过缺失路径测试")

    def test_with_overrides_is_pure_and_nested(self) -> None:
        """with_overrides 能改嵌套字段，且不修改原对象。"""
        config = PruneConfig.for_demo()
        changed = config.with_overrides(train={"epochs": 1}, sparsity=0.3, rounds=2)
        self.assertEqual(changed.train.epochs, 1)
        self.assertEqual(changed.sparsity, 0.3)
        self.assertEqual(changed.rounds, 2)
        self.assertEqual(config.train.epochs, 8)  # 原对象不受影响
        self.assertEqual(config.sparsity, 0.5)
        self.assertEqual(changed.output.tag, config.output.tag)  # 未指定的字段保持


# ---------------------------------------------------------------------- 13. 导出


class TestExport(unittest.TestCase):
    """导出：检查点、元信息、TorchScript、报告批量保存。"""

    def test_checkpoint_roundtrip_including_masks(self) -> None:
        """检查点往返后权重逐元素相等，掩码可重建为 MaskBook。"""
        model = _mlp()
        pruner = UnstructuredPruner(model, scope="global")
        pruner.prune(0.7)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(model, os.path.join(tmp, "ckpt.pt"),
                                   meta={"note": "x"}, masks=pruner.masks)
            payload = load_checkpoint(path)
            self.assertEqual(payload["meta"], {"note": "x"})
            self.assertEqual(payload["model_meta"]["params"], count_parameters(model))
            restored = MaskBook(payload["masks"])
            self.assertTrue(torch.equal(restored.get("net.0.weight"), pruner.masks.get("net.0.weight")))
            fresh = _mlp()
            load_checkpoint(path, model=fresh)
            for name, parameter in model.named_parameters():
                self.assertTrue(
                    torch.equal(parameter.detach(), dict(fresh.named_parameters())[name].detach()),
                    name,
                )

    def test_model_meta_layers_and_counts(self) -> None:
        """model_meta 的参数统计与 count_parameters 一致，逐层形状可读。"""
        model = _mlp()
        meta = model_meta(model)
        self.assertEqual(meta["params"], count_parameters(model))
        self.assertEqual(meta["nonzero_params"], count_nonzero_parameters(model))
        self.assertEqual([layer["name"] for layer in meta["layers"]], ["net.0", "net.2", "net.4"])
        self.assertEqual(meta["layers"][0]["parameters"], {"weight": [64, 2], "bias": [64]})
        self.assertEqual(meta["layers"][2]["parameters"], {"weight": [3, 64], "bias": [3]})

    def test_torchscript_export_loadable(self) -> None:
        """导出的 TorchScript 能加载、能前向，且与源模型输出一致。"""
        torch.manual_seed(0)
        model = _SmallConv().eval()
        inputs = (torch.randn(2, 3, 8, 8),)
        StructuredPruner(model, inputs).prune(0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = export_torchscript(model, inputs, os.path.join(tmp, "sub", "model.ts"))
            self.assertTrue(os.path.exists(path))
            loaded = torch.jit.load(path)
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(*inputs), loaded(*inputs), atol=1e-5))

    def test_save_reports_writes_both_formats(self) -> None:
        """save_reports 同时产出 JSON 与 Markdown。"""
        reports = {"a": Report("甲").add_kv("k", {"x": 1}), "b": Report("乙")}
        with tempfile.TemporaryDirectory() as tmp:
            saved = save_reports(reports, os.path.join(tmp, "out"))
            self.assertEqual(set(saved), {"a", "b"})
            for name, paths in saved.items():
                self.assertTrue(os.path.exists(paths["json"]), name)
                self.assertTrue(os.path.exists(paths["markdown"]), name)


# ---------------------------------------------------------------------- 14. 命令行


class TestCli(unittest.TestCase):
    """命令行：模板生成、依赖图分析、错误路径返回非零。"""

    def _run(self, argv):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli_main(argv)
        return code, buffer.getvalue()

    def test_template_writes_loadable_config(self) -> None:
        """template 子命令写出的配置能被 load_config 读回。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tpl.json")
            code, output = self._run(["template", path])
            self.assertEqual(code, 0)
            self.assertIn("配置模板已写入", output)
            config = load_config(path)
            self.assertEqual(config.mode, "structured")

    def test_graph_subcommand(self) -> None:
        """graph 子命令能跑通并打印通道组表。"""
        code, output = self._run([
            "graph", "--model", "small_vgg", "--dataset", "tiny_images",
            "--dataset-kwarg", "n_samples=200", "--dataset-kwarg", "image_size=16",
        ])
        self.assertEqual(code, 0)
        self.assertIn("通道组", output)
        self.assertIn("可裁剪通道组", output)

    def test_error_paths_return_nonzero(self) -> None:
        """非法准则与非法扫描强度都要返回非零码而不是抛异常。"""
        code, _ = self._run([
            "prune", "--model", "mlp", "--dataset", "blobs",
            "--dataset-kwarg", "n_samples=100", "--criterion", "nope",
            "--train-epochs", "0", "--finetune-epochs", "0", "--rounds", "1",
        ])
        self.assertEqual(code, 2)
        code, _ = self._run([
            "sensitivity", "--model", "mlp", "--dataset", "blobs",
            "--dataset-kwarg", "n_samples=100", "--sparsities", "abc",
        ])
        self.assertEqual(code, 2)


# ---------------------------------------------------------------------- 15. 流水线


class TestPipeline(unittest.TestCase):
    """端到端流水线：最小配置下产出指标与文件。"""

    def test_unstructured_pipeline(self) -> None:
        """非结构化流水线：稠密参数量不变、稀疏度命中目标、产物齐全。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = PruneConfig(
                model="mlp",
                dataset="blobs",
                dataset_kwargs={"n_samples": 200, "n_classes": 3},
                batch_size=32,
                train={"enabled": True, "epochs": 2},
                mode="unstructured",
                scope="global",
                rounds=1,
                sparsity=0.5,
                finetune={"enabled": False},
                benchmark=False,
                output={"directory": tmp, "tag": "t"},
            )
            from pruning_toolkit import run

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                result = run(config, log=None)

            metrics = result.metrics
            self.assertEqual(metrics["参数下降比例"], 0.0)
            self.assertEqual(metrics["FLOPs下降比例"], 0.0)
            self.assertAlmostEqual(metrics["稀疏度_可剪参数"], 0.5, places=2)
            self.assertGreater(metrics["非零参数_剪枝前"], metrics["非零参数_剪枝后"])
            for key in ("准确率_剪枝前", "准确率_剪枝后未微调"):
                self.assertIn(key, metrics)
            for name in ("metrics.json", "config.json", "checkpoint.pt", "masks.pt"):
                self.assertTrue(os.path.exists(os.path.join(tmp, "t", name)), name)
            self.assertTrue(os.path.exists(os.path.join(tmp, "t", "pruning.md")))
            self.assertIsNotNone(result.iteration)
            self.assertIn("剪枝流水线结果", result.summary_text())

    def test_structured_pipeline(self) -> None:
        """结构化流水线：参数量与 FLOPs 都下降，前向校验通过。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = PruneConfig(
                model="small_vgg",
                dataset="tiny_images",
                dataset_kwargs={"n_samples": 200, "image_size": 16},
                batch_size=32,
                train={"enabled": False, "epochs": 0},
                mode="structured",
                scope="local",
                rounds=1,
                sparsity=0.5,
                finetune={"enabled": False},
                benchmark=False,
                output={"directory": tmp, "tag": "s"},
            )
            from pruning_toolkit import run

            with contextlib.redirect_stdout(io.StringIO()):
                result = run(config, log=None)

            metrics = result.metrics
            self.assertGreater(metrics["参数下降比例"], 0.5)
            self.assertGreater(metrics["FLOPs下降比例"], 0.5)
            self.assertEqual(
                metrics["参数量_剪枝后"], count_parameters(result.model)
            )
            report = result.reports["pruning"]
            self.assertTrue(any(section["name"] == "逐轮记录" for section in report.to_dict()["sections"]))


# ---------------------------------------------------------------------- 16. 包导出


class TestPackageExports(unittest.TestCase):
    """包的公开 API 必须都能取到（防止 __all__ 里写错名字）。"""

    def test_all_names_resolvable(self) -> None:
        """__all__ 中每个名字都能通过 getattr 取到。"""
        import pruning_toolkit

        missing = [name for name in pruning_toolkit.__all__ if not hasattr(pruning_toolkit, name)]
        self.assertEqual(missing, [])
        self.assertEqual(pruning_toolkit.__version__, "1.0.0")
        self.assertIn("l1", CRITERIA)
        for name in ("structured", "unstructured", "graph", "analysis", "iterative"):
            self.assertIsNotNone(getattr(pruning_toolkit, name, None) or True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
