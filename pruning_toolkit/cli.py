"""命令行接口：``python -m pruning_toolkit <子命令>``。

子命令
------
``demo``        跑一遍完整演示（含训练、剪枝、微调、评估、导出报告）
``prune``       按配置文件或命令行参数执行剪枝流水线
``inspect``     只看模型体检：参数量、稀疏度、FLOPs、延迟、逐模块明细
``graph``       只做通道依赖图分析：哪些层能安全剪、哪些被屏障挡住及原因
``sensitivity`` 只做敏感度扫描：各层对不同剪枝强度的响应曲线
``template``    导出一份带注释默认值的配置模板（JSON）

示例::

    python -m pruning_toolkit graph --model resnet_small --dataset tiny_images
    python -m pruning_toolkit sensitivity --model small_vgg --mode structured --dataset tiny_images
    python -m pruning_toolkit prune --model small_vgg --mode structured --rounds 3 --sparsity 0.5 --out runs/vgg
    python -m pruning_toolkit demo --preset cnn
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from .analysis import structured_sensitivity, unstructured_sensitivity
from .config import PruneConfig, load_config
from .graph import DependencyGraph, UnsupportedGraphError
from .metrics import benchmark_latency, summarize
from .pipeline import prepare, run
from .zoo.data import DATASET_NAMES
from .zoo.models import all_models

#: 可用的示例模型名（来自 zoo 的注册表）
MODEL_NAMES = tuple(all_models().keys())


def _parse_pairs(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    """把 ``key=value`` 形式的参数解析成字典，值自动推断类型。"""
    result: Dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError("参数 %r 不是 key=value 形式" % item)
        key, _, raw = item.partition("=")
        key = key.strip()
        raw = raw.strip()
        lowered = raw.lower()
        if lowered in ("true", "false"):
            value: Any = lowered == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        result[key] = value
    return result


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="small_vgg", choices=list(MODEL_NAMES),
                        help="示例模型名（默认 small_vgg）")
    parser.add_argument("--dataset", default="tiny_images", choices=list(DATASET_NAMES),
                        help="数据集名（默认 tiny_images）")
    parser.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--dataset-kwarg", action="append", metavar="K=V", default=None,
                        help="透传给数据集构造函数的参数，可重复，如 --dataset-kwarg noise=1.1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pruning-toolkit",
        description="PyTorch 模型剪枝工具箱：通道依赖分析 / 结构化与非结构化剪枝 / "
                    "敏感度分析 / 迭代剪枝与微调 / 评估与报告",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- demo ----
    demo = subparsers.add_parser("demo", help="跑一遍完整演示")
    demo.add_argument("--preset", default="cnn", choices=["cnn", "mlp"],
                      help="演示预设：cnn=结构化剪枝小 VGG；mlp=非结构化剪枝 MLP")
    demo.add_argument("--quick", action="store_true", help="快速模式（减少训练轮数）")
    demo.add_argument("--out", default=None, help="产物目录（覆盖预设值）")
    demo.add_argument("--no-train", action="store_true", help="跳过剪枝前的初始训练")
    demo.add_argument("--torchscript", action="store_true", help="额外导出 TorchScript")
    demo.set_defaults(handler=_cmd_demo)

    # ---- prune ----
    prune = subparsers.add_parser("prune", help="执行剪枝流水线")
    prune.add_argument("--config", default=None, help="配置文件（.json / .yaml）")
    _add_common_flags(prune)
    prune.add_argument("--mode", default=None, choices=["structured", "unstructured"])
    prune.add_argument("--criterion", default=None,
                       help="l1 / l2 / random / taylor / bn_scale / fpgm / none")
    prune.add_argument("--scope", default=None, choices=["global", "local"])
    prune.add_argument("--schedule", default=None, choices=["constant", "linear", "cubic", "exponential"])
    prune.add_argument("--rounds", type=int, default=None, help="迭代轮数")
    prune.add_argument("--sparsity", type=float, default=None,
                       help="非结构化=目标稀疏度；结构化=通道裁剪比例")
    prune.add_argument("--batch-size", type=int, default=None)
    prune.add_argument("--train-epochs", type=int, default=None)
    prune.add_argument("--finetune-epochs", type=int, default=None, help="每轮微调轮数")
    prune.add_argument("--allocate", action="store_true", help="用敏感度分析自动分配逐层目标")
    prune.add_argument("--alpha", type=float, default=None, help="分配力度（越大越按敏感度区分）")
    prune.add_argument("--per-layer", action="append", metavar="K=V", default=None,
                       help="逐层目标，可重复，如 --per-layer features.0=0.3")
    prune.add_argument("--exclude", action="append", default=None, help="排除的模块名，可重复")
    prune.add_argument("--protect-first-last", action="store_true", help="保护首尾可剪模块")
    prune.add_argument("--out", default=None, help="产物目录")
    prune.add_argument("--tag", default=None, help="产物子目录名")
    prune.add_argument("--quiet", action="store_true", help="静默模式")
    prune.add_argument("--torchscript", action="store_true", help="额外导出 TorchScript")
    prune.add_argument("--dump-config", default=None, help="把最终配置写到该路径后退出")
    prune.set_defaults(handler=_cmd_prune)

    # ---- inspect ----
    inspect = subparsers.add_parser("inspect", help="模型体检：参数量 / FLOPs / 延迟")
    _add_common_flags(inspect)
    inspect.add_argument("--benchmark", action="store_true", help="额外实测推理延迟")
    inspect.add_argument("--train-epochs", type=int, default=0, help="先训练若干轮（默认 0）")
    inspect.set_defaults(handler=_cmd_inspect)

    # ---- graph ----
    graph = subparsers.add_parser("graph", help="通道依赖图分析")
    _add_common_flags(graph)
    graph.add_argument("--exclude", action="append", default=None, help="排除的模块名，可重复")
    graph.add_argument("--no-protect-output", action="store_true",
                       help="不保护输出层通道组（会允许裁剪类别维，通常只在特殊场景使用）")
    graph.add_argument("--report", default=None, help="把报告写到该文件（.md / .json）")
    graph.set_defaults(handler=_cmd_graph)

    # ---- sensitivity ----
    sensitivity = subparsers.add_parser("sensitivity", help="敏感度扫描")
    _add_common_flags(sensitivity)
    sensitivity.add_argument("--mode", default="structured", choices=["structured", "unstructured"])
    sensitivity.add_argument("--criterion", default="l1")
    sensitivity.add_argument("--sparsities", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                             help="逗号分隔的扫描强度")
    sensitivity.add_argument("--train-epochs", type=int, default=0, help="先训练若干轮（默认 0）")
    sensitivity.add_argument("--report", default=None, help="把报告写到该文件（.md / .json）")
    sensitivity.set_defaults(handler=_cmd_sensitivity)

    # ---- template ----
    template = subparsers.add_parser("template", help="导出配置模板")
    template.add_argument("path", nargs="?", default="prune_config.json", help="输出路径")
    template.add_argument("--preset", default="cnn", choices=["cnn", "mlp"])
    template.set_defaults(handler=_cmd_template)

    return parser


# ---------------------------------------------------------------------- 子命令实现


def _config_from_args(args: argparse.Namespace, default: Optional[PruneConfig] = None) -> PruneConfig:
    config = load_config(args.config) if getattr(args, "config", None) else (default or PruneConfig())
    overrides: Dict[str, Any] = {
        "model": getattr(args, "model", None),
        "dataset": getattr(args, "dataset", None),
        "device": getattr(args, "device", None),
        "seed": getattr(args, "seed", None),
        "mode": getattr(args, "mode", None),
        "criterion": getattr(args, "criterion", None),
        "scope": getattr(args, "scope", None),
        "schedule": getattr(args, "schedule", None),
        "rounds": getattr(args, "rounds", None),
        "sparsity": getattr(args, "sparsity", None),
        "batch_size": getattr(args, "batch_size", None),
        "alpha": getattr(args, "alpha", None),
    }
    dataset_kwargs = _parse_pairs(getattr(args, "dataset_kwarg", None))
    if dataset_kwargs:
        overrides["dataset_kwargs"] = {**config.dataset_kwargs, **dataset_kwargs}
    per_layer = _parse_pairs(getattr(args, "per_layer", None))
    if per_layer:
        overrides["per_layer"] = {**config.per_layer, **per_layer}
    if getattr(args, "exclude", None):
        overrides["exclude"] = list(args.exclude)
    if getattr(args, "allocate", False):
        overrides["allocate"] = True
    if getattr(args, "protect_first_last", False):
        overrides["protect_first_last"] = True
    train_epochs = getattr(args, "train_epochs", None)
    if train_epochs is not None:
        overrides["train"] = {"enabled": train_epochs > 0, "epochs": train_epochs}
    finetune_epochs = getattr(args, "finetune_epochs", None)
    if finetune_epochs is not None:
        overrides["finetune"] = {"enabled": finetune_epochs > 0, "epochs_per_round": finetune_epochs}
    out = getattr(args, "out", None)
    tag = getattr(args, "tag", None)
    if out is not None or tag is not None:
        output: Dict[str, Any] = {}
        if out is not None:
            output["directory"] = out
            # 用户显式指定了输出目录，就不该再被配置里的 tag 追加一层子目录；
            # 想同时指定就用 --tag 明确说出来。
            if tag is None:
                output["tag"] = ""
        if tag is not None:
            output["tag"] = tag
        overrides["output"] = output
    if getattr(args, "quiet", False):
        overrides["verbose"] = False
    if getattr(args, "torchscript", False):
        overrides["output"] = {**(overrides.get("output") or {}), "export_torchscript": True}
    config = config.with_overrides(**overrides)
    config.validate()
    return config


def _cmd_demo(args: argparse.Namespace) -> int:
    config = PruneConfig.for_mlp_demo() if args.preset == "mlp" else PruneConfig.for_demo()
    if args.quick:
        if args.preset == "mlp":
            config = config.with_overrides(
                train={"epochs": 8}, rounds=2, finetune={"epochs_per_round": 3}
            )
        else:
            config = config.with_overrides(
                train={"epochs": 3}, rounds=2, finetune={"epochs_per_round": 1}
            )
    if args.out:
        # 显式给了输出目录就不再追加预设里的 tag（与 prune 子命令行为一致）
        config = config.with_overrides(output={"directory": args.out, "tag": ""})
    if args.no_train:
        config = config.with_overrides(train={"enabled": False})
    if args.torchscript:
        config = config.with_overrides(output={"export_torchscript": True})

    result = run(config)
    print(result.summary_text())
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if args.dump_config:
        config.save(args.dump_config)
        print("配置已写入 %s" % args.dump_config)
        return 0
    result = run(config, log=None if args.quiet else print)
    print(result.summary_text())
    return 0


def _inspection_context(args: argparse.Namespace):
    config = _config_from_args(args)
    model, train_loader, test_loader, data_meta, example_inputs, device = prepare(
        config, log=print
    )
    if getattr(args, "train_epochs", 0):
        from .zoo.train import fit

        fit(model, train_loader, test_loader, epochs=args.train_epochs, device=device,
            log=print, seed=config.seed)
    return config, model, train_loader, test_loader, data_meta, example_inputs, device


def _cmd_inspect(args: argparse.Namespace) -> int:
    config, model, _, _, data_meta, example_inputs, _ = _inspection_context(args)
    latency = benchmark_latency(model, example_inputs) if args.benchmark else None
    report = summarize(model, example_inputs, latency=latency)
    report.metadata["数据集"] = config.dataset
    report.metadata["训练样本"] = data_meta.get("n_train")
    print(report.to_text())
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    config, model, _, _, _, example_inputs, _ = _inspection_context(args)
    graph = DependencyGraph(
        model,
        example_inputs,
        protect_output=not args.no_protect_output,
        ignore=set(args.exclude or ()),
    )
    report = graph.summary()
    report.metadata["数据集"] = config.dataset
    print(report.to_text())
    if args.report:
        print("报告已写入 %s" % report.save(args.report))
    return 0


def _cmd_sensitivity(args: argparse.Namespace) -> int:
    config, model, _, _, _, example_inputs, _ = _inspection_context(args)
    sparsities = [float(item) for item in args.sparsities.split(",") if item.strip()]
    if args.mode == "structured":
        report = structured_sensitivity(
            model, example_inputs, sparsities=sparsities, criterion=args.criterion,
            exclude=config.exclude,
        )
    else:
        report = unstructured_sensitivity(
            model, sparsities=sparsities, criterion=args.criterion, exclude=config.exclude
        )
    text = report.to_report()
    text.metadata["模型"] = config.model
    text.metadata["数据集"] = config.dataset
    print(text.to_text())
    if args.report:
        print("报告已写入 %s" % text.save(args.report))
    return 0


def _cmd_template(args: argparse.Namespace) -> int:
    config = PruneConfig.for_mlp_demo() if args.preset == "mlp" else PruneConfig.for_demo()
    config.save(args.path)
    print("配置模板已写入 %s" % os.path.abspath(args.path))
    print("可用字段说明见 pruning_toolkit/config.py 的 PruneConfig 定义。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except UnsupportedGraphError as exc:
        print("依赖图分析失败：%s" % exc, file=sys.stderr)
        return 3
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as exc:
        print("错误：%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
