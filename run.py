"""run.py —— 用本仓库的 pruning_toolkit 对 best.pt 做剪枝。

先读这里：这个脚本**能做什么、不能做什么**
================================================
脚本默认读取 `best.pt`（ultralytics 检测模型，Detect 头）。
以作者测试用的模型为例：9 类、imgsz=672、约 3.0M 参数。
针对这类模型，本工具的能力边界是实测过的（见 README 与 `extras/yolov8_torch_pruning/`）：

* ✅ **非结构化剪枝**（本脚本默认走这条）：按重要性把单个权重置零。
  - 有效参数量真的下降，可以配掩码感知微调把精度找回来；
  - 但**稠密 FLOPs 与延迟基本不变**——它省的是存储与有效参数，不是计算量。
* ❌ **结构化剪枝（删通道）**：本工具做不到。两个原因都是硬阻塞：
  1. ultralytics 的 `BaseModel._predict_once` 对 Proxy 做迭代，`torch.fx` 追不了；
  2. 工具把 `torch.cat` / `Upsample` 当作屏障，而 YOLOv8 的 PAN-FPN 颈部全靠这两者。
  脚本会在运行时现场验证这一点并打印原因，而非仅在文档中声明。

若目标是降低 FLOPs（删除通道），请使用已验证可用的方案：
`extras/yolov8_torch_pruning/reproduce.py`（torch-pruning + C2f 改造，实测 MACs -54%）。

用法
====
    # 需要已安装 ultralytics 的 Python 环境
    python run.py --sparsity 0.5

    # 剪枝 + 评估 mAP + 掩码感知微调
    python run.py --data <dataset.yaml> --sparsity 0.5 --finetune-epochs 3

    # Windows 终端若中文乱码，先设置 UTF-8
    #   PowerShell: $env:PYTHONIOENCODING="utf-8"
    #   bash:       export PYTHONIOENCODING=utf-8

**dataset.yaml 必须与权重的训练数据一致**：

* 评估与微调都会读取 `--data` 指定的 yaml；用错数据集会得到没有参考价值的 mAP。
* 权重中的 `train_args['data']` 字段记录了训练时使用的 yaml 路径，可据此确认。
* 同一工程内可能存在多份数据集配置，注意不要把「模型未训练过的数据」当作验证集。

参考数据（`--sparsity 0.5 --finetune-epochs 3`，RTX 4060，私有数据集，仅作量级参考）：

| 阶段 | mAP50 | mAP50-95 |
| --- | --- | --- |
| 剪枝前 | 0.9950 | **0.9751** |
| 剪 50% 权重后（未微调） | 0.9949 | 0.9244 |
| 掩码感知微调 3 轮后 | 0.9950 | **0.9824** |

即：50% 权重置零仅损失约 0.05 mAP，微调 3 轮后略高于原始模型。
需要注意稠密 FLOPs 与延迟不变（约 6 ms，前后差异属测量噪声）。

产物写到 `runs/prune_best/`：可被 `YOLO(path)` 直接加载的权重（已微调且严格稀疏）、
掩码、以及 Markdown / JSON 报告。
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# 让脚本无论在哪个工作目录下运行都能 import 到本仓库的工具包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruning_toolkit import (
    CRITERIA,
    DependencyGraph,
    MaskBook,
    Report,
    UnstructuredPruner,
    UnsupportedGraphError,
    benchmark_latency,
    collect_grads,
    count_nonzero_parameters,
    count_parameters,
    model_sparsity,
    save_reports,
)

DEFAULT_WEIGHTS = "best.pt"
DEFAULT_OUT_DIR = os.path.join("runs", "prune_best")


# ---------------------------------------------------------------------- 命令行


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 pruning_toolkit 对 ultralytics 检测模型做非结构化剪枝",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="输入权重（ultralytics .pt）")
    parser.add_argument("--sparsity", type=float, default=0.5, help="目标稀疏度（0~1）")
    parser.add_argument("--criterion", default="l1", choices=sorted(CRITERIA),
                        help="重要性准则；taylor 需要梯度（无需数据集，用输出幅值反传）")
    parser.add_argument("--scope", default="global", choices=["global", "local"],
                        help="global=所有参数统一排序，local=逐层等比例")
    parser.add_argument("--include-norm", dest="include_norm", action="store_true", default=False,
                        help="是否把 BN 的 gamma 一起纳入全局排序。默认不纳入：gamma 的量级"
                             "（≈1）与卷积权重差得多，混在一起排序会让卷积层被过度剪")
    parser.add_argument("--protect-head", action=argparse.BooleanOptionalAction, default=True,
                        help="保护检测头（model.NN.*）不参与剪枝；--no-protect-head 则连它一起剪"
                             "（可能破坏回归/分类输出的数值范围，慎用）")
    parser.add_argument("--data", default=None,
                        help="数据集 yaml（给了才会评估 mAP 与微调）；best.pt 里记录的是 dataset.yaml")
    parser.add_argument("--finetune-epochs", type=int, default=0,
                        help="微调轮数（0 表示只剪不训）")
    parser.add_argument("--lr0", type=float, default=None,
                        help="微调初始学习率（默认用 ultralytics 的默认值）")
    parser.add_argument("--imgsz", type=int, default=0, help="推理尺寸（0 表示沿用权重里的设置）")
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0（默认自动）")
    parser.add_argument("--bench-iters", type=int, default=10, help="延迟实测次数")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="产物目录")
    return parser.parse_args()


# ---------------------------------------------------------------------- 工具函数


def resolve_device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_detector(path: str, device: torch.device) -> Tuple[Any, torch.nn.Module, Dict[str, Any]]:
    """加载 ultralytics 权重，返回 (YOLO 包装对象, 纯 nn.Module, 原始 ckpt 字典)。"""
    from ultralytics import YOLO

    yolo = YOLO(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ckpt["model"].float().to(device).eval()
    # ultralytics 存盘时会 strip_optimizer，把 requires_grad 全置 False（推理态）。
    # 需要梯度的路径（taylor 准则、微调）必须重新打开，否则反传会报
    # "element 0 of tensors does not require grad"。
    for param in net.parameters():
        param.requires_grad_(True)
    yolo.model = net  # 让 YOLO.val()/train() 用我们手里的这个模型对象
    return yolo, net, ckpt


def head_prefix(net: torch.nn.Module) -> str:
    """检测头在参数名里的前缀，例如 'model.22'。"""
    index = len(net.model) - 1
    return "model.%d" % index


def probe_structured(net: torch.nn.Module, imgsz: int, device: torch.device) -> bool:
    """现场验证结构化剪枝是否可用——把「不能」变成可复现的事实。"""
    example = (torch.zeros(1, 3, imgsz, imgsz, device=device),)
    try:
        graph = DependencyGraph(net, example)
    except UnsupportedGraphError as exc:
        reason = str(exc).split("。")[0]
        print("  ✗ 无法建立通道依赖图：%s" % reason)
        print("  ✗ 因此**结构化剪枝（删通道）不可用**，本脚本只做非结构化剪枝。")
        print("    （想删通道降 FLOPs：用 extras/yolov8_torch_pruning/reproduce.py，已实测可用）")
        return False
    except Exception as exc:  # noqa: BLE001 - 追踪阶段的任何异常都视为不可用
        print("  ✗ 追踪失败：%s: %s" % (type(exc).__name__, str(exc)[:120]))
        return False
    print("  ✓ 可建立依赖图：通道组 %d，可剪 %d，屏障 %d"
          % (len(graph.groups), len(graph.prunable_groups()), len(graph.barriers)))
    return True


def _sum_all_tensors(obj: Any) -> Optional[torch.Tensor]:
    """递归地把嵌套结构里所有张量求和，忽略其它类型。

    检测模型在不同模式下输出结构完全不同：
    * 训练模式：``Detect.forward`` 返回 ``[分支0, 分支1, 分支2]``（**list of tensor**）；
    * 推理模式：返回 ``(预测, [特征图...])``。
    只取 ``output[0]`` 会漏掉另外两个分支——那样反传出来的梯度不完整，
    表现为「部分参数 grad 为 None」，taylor 准则就会失败。
    """
    if isinstance(obj, torch.Tensor):
        return obj.sum()
    if isinstance(obj, (list, tuple)):
        total: Optional[torch.Tensor] = None
        for item in obj:
            part = _sum_all_tensors(item)
            if part is not None:
                total = part if total is None else total + part
        return total
    return None


def output_sum_loss(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    """无数据集时的替代梯度：对输出幅值求和再反传（上游示例也用这种做法做正则）。

    它不是任务损失，所以 `taylor` 只应看作「带一点梯度信息的幅值剪枝」，
    真正的 taylor 需要真实标签。
    """
    inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
    total = _sum_all_tensors(model(inputs))
    if total is None:
        raise RuntimeError("模型输出里找不到任何张量，无法用输出幅值反传")
    return total


def build_grads(net: torch.nn.Module, criterion: str, imgsz: int, device: torch.device):
    """按需采集梯度（只有 taylor 需要）。"""
    if criterion != "taylor":
        return None
    print("  · 采集替代梯度（对输出幅值反传，不需要数据集）…")
    sample = (torch.zeros(1, 3, imgsz, imgsz, device=device),)
    grads = collect_grads(net, output_sum_loss, [sample], max_batches=1, device=device)
    # 兜底：某些参数可能确实不参与当前前向（例如未被使用的分支），
    # 用零梯度补齐，避免 taylor 准则因为缺项直接报错。零梯度意味着这些权重会被优先剪掉。
    missing = [name for name, _ in net.named_parameters() if name not in grads]
    if missing:
        print("  ! %d 个参数没有梯度（未参与前向），按零梯度处理：%s%s"
              % (len(missing), ", ".join(missing[:3]), " …" if len(missing) > 3 else ""))
        for name, param in net.named_parameters():
            grads.setdefault(name, torch.zeros_like(param))
    return grads


def evaluate_map(yolo: Any, net: torch.nn.Module, data: Optional[str], imgsz: int,
                 device: torch.device, tag: str) -> Optional[float]:
    """有数据集时用 ultralytics 的 validator 算 mAP50-95。

    ⚠️ ultralytics 的 `val()` 会**就地** `fuse()` 模型（把 BatchNorm 折进卷积），
    这会让传入的模型永久失去 BN 参数（参数量、掩码键名都会变）。所以这里评估的是
    `net` 的**深拷贝**，评估完丢弃，保证后续剪枝、微调、保存都作用在未被改动的模型上。
    """
    if not data:
        return None
    if not os.path.exists(data):
        print("  ! 数据集 %s 不存在，跳过 %s 评估" % (data, tag))
        return None
    original = yolo.model
    yolo.model = copy.deepcopy(net).eval()
    try:
        metrics = yolo.val(data=data, imgsz=imgsz, device=str(device), verbose=False,
                           plots=False)
        value = float(metrics.box.map)
        print("  ✓ %s mAP50-95 = %.4f（mAP50 = %.4f）" % (tag, value, float(metrics.box.map50)))
        return value
    except Exception as exc:  # noqa: BLE001
        print("  ! %s 评估失败：%s: %s" % (tag, type(exc).__name__, str(exc)[:100]))
        return None
    finally:
        yolo.model = original


def finetune_with_masks(yolo: Any, net: torch.nn.Module, masks: MaskBook, data: str,
                        epochs: int, imgsz: int, device: torch.device, out_dir: str,
                        lr0: Optional[float] = None) -> torch.nn.Module:
    """掩码感知微调：每个训练 step 之后把掩码压回权重，保证被剪位置不被重新激活。

    ultralytics 的训练循环不暴露 post_step 钩子，这里用它的回调机制实现
    `on_train_batch_end` → masks.apply(trainer.model)。

    返回值很重要：ultralytics 的 `train()` 会**用 cfg 重建一个新模型对象**并赋给
    `yolo.model`，传进来的 `net` 会变成过期引用。必须使用返回的模型继续评估与保存，
    否则会出现「训练日志显示 mAP 恢复、脚本却报告没变」的假象。
    """
    if epochs <= 0:
        return net
    if not data or not os.path.exists(data):
        print("  ! 没有可用数据集，跳过微调（--data 指定 dataset.yaml 才会训练）")
        return net

    def on_train_batch_end(trainer) -> None:  # noqa: ANN001 - ultralytics 回调签名
        masks.apply(trainer.model)

    yolo.model = net
    yolo.add_callback("on_train_batch_end", on_train_batch_end)
    print("  · 开始微调 %d 轮（每步后重新压回掩码）…" % epochs)
    train_kwargs: Dict[str, Any] = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "device": str(device),
        "project": os.path.join(out_dir, "train"),
        "name": "finetune",
        "exist_ok": True,
        "verbose": False,
        "plots": False,
    }
    if lr0 is not None:
        train_kwargs["lr0"] = lr0
    yolo.train(**train_kwargs)

    # 取回真正训练后的模型：优先 trainer 的 best 权重（通常 mAP 最高）
    trained = yolo.model
    best_path = getattr(getattr(yolo, "trainer", None), "best", None)
    if best_path is not None and os.path.exists(str(best_path)):
        from ultralytics import YOLO as _YOLO

        reloaded = _YOLO(str(best_path))
        trained = reloaded.model.float().to(device)
        yolo.model = trained
        print("  · 已加载 trainer 的最佳权重：%s" % best_path)
    # 这份权重是 EMA 的「近似稀疏」，这里再压一次，保证交付的权重严格稀疏
    masks.apply(trained)
    print("  ✓ 微调结束，掩码已重新应用（严格稀疏）")
    return trained


def save_ultralytics_ckpt(net: torch.nn.Module, source_ckpt: Dict[str, Any], path: str) -> str:
    """把剪枝后的模型存成 `YOLO(path)` 能直接加载的格式。

    两个细节必须处理：

    1. `attempt_load_one_weight` 是 `ckpt.get('ema') or ckpt['model']`，即**优先用 EMA**，
       所以必须把 ema 也换成剪枝后的模型（或置空），否则加载到的是旧权重；
    2. ultralytics 官方存盘前会 `.half()`（见 `strip_optimizer`），这里保持一致，
       否则 fp32 存出来的文件会比原始 fp16 权重大一倍，看起来「越剪越大」。
    """
    payload = copy.copy(source_ckpt)
    payload["model"] = copy.deepcopy(net).half().cpu()
    payload["ema"] = None
    for key in ("optimizer", "updates"):
        payload[key] = None
    torch.save(payload, path)
    return path


def build_report(meta: Dict[str, Any], prune_report: Any, structured_ok: bool,
                 maps: Dict[str, Optional[float]]) -> Report:
    report = Report("best.pt 非结构化剪枝报告", model_name=meta["weights"])
    report.metadata["权重来源"] = meta["weights"]
    report.metadata["检测模型"] = "%d 类，imgsz=%d" % (meta["num_classes"], meta["imgsz"])

    overview: Dict[str, Any] = {
        "剪枝模式": "非结构化（权重置零，形状不变）",
        "结构化是否可用": "可用" if structured_ok else "不可用（fx 无法追踪 + cat/upsample 屏障）",
        "重要性准则": meta["criterion"],
        "剪枝范围": meta["scope"],
        "目标稀疏度": meta["sparsity"],
        "实际稀疏度（可剪参数）": prune_report.achieved_sparsity,
        "参数量（稠密，不变）": meta["params_dense"],
        "非零参数（剪枝前）": meta["nonzero_before"],
        "非零参数（剪枝后）": meta["nonzero_after"],
        "全模型稀疏度": meta["sparsity_after"],
        "检测头是否参与剪枝": "否（已保护）" if meta["protect_head"] else "是",
        "耗时(秒)": round(meta["duration"], 2),
    }
    for key, value in maps.items():
        if value is not None:
            overview[key] = value
    report.add_kv("总览", overview)

    report.add_table(
        "逐参数稀疏度",
        ["参数", "形状", "元素数", "保留", "剪前稀疏度", "剪后稀疏度"],
        [[layer.param_name, tuple(layer.shape), layer.numel, layer.kept,
          layer.sparsity_before, layer.sparsity_after] for layer in prune_report.layers],
    )
    if meta.get("latency_before") and meta.get("latency_after"):
        report.add_kv(
            "实测延迟（同一台机器）",
            {
                "剪枝前 p50(ms)": meta["latency_before"],
                "剪枝后 p50(ms)": meta["latency_after"],
            },
            note="非结构化剪枝不改变稠密算子的计算量，延迟差异属于测量噪声；"
                 "要真正降 FLOPs/延迟必须删通道（见 extras/yolov8_torch_pruning）。",
        )
    report.add_text(
        "结论与建议",
        "本模型只完成了非结构化剪枝：省下的是有效参数量与存储，推理速度基本不变。\n"
        "若目标是降低 FLOPs/延迟，请改用结构化（通道）剪枝：本工具因 fx 追踪限制无法处理"
        "该模型，已验证可用的方案是 extras/yolov8_torch_pruning/reproduce.py"
        "（先做 C2f→C2fV2 改造，再 torch-pruning 剪枝）。\n"
        "若目标是压缩后的精度，请务必微调：未微调的稀疏模型 mAP 通常会明显下降。",
    )
    return report


# ---------------------------------------------------------------------- 主流程


def main() -> int:
    args = parse_args()
    started = time.time()
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 68)
    print("步骤 1/6  加载权重：%s（设备 %s）" % (args.weights, device))
    if not os.path.exists(args.weights):
        print("  ✗ 找不到 %s" % args.weights)
        return 2
    yolo, net, source_ckpt = load_detector(args.weights, device)

    train_args = source_ckpt.get("train_args") or {}
    imgsz = args.imgsz or int(train_args.get("imgsz", 640))
    num_classes = int(getattr(net, "yaml", {}).get("nc", 0))
    params_dense = count_parameters(net)
    nonzero_before = count_nonzero_parameters(net)
    print("  ✓ %d 类，参数量 %d（%.2f M），imgsz=%d"
          % (num_classes, params_dense, params_dense / 1e6, imgsz))

    print("\n步骤 2/6  检查结构化剪枝是否可行（这是本工具的能力边界）")
    structured_ok = probe_structured(net, imgsz, device)

    print("\n步骤 3/6  实测剪枝前延迟（%d 次）" % args.bench_iters)
    example = (torch.zeros(1, 3, imgsz, imgsz, device=device),)
    latency_before = benchmark_latency(net, example, warmup=3, iters=args.bench_iters,
                                       device=device)["p50_ms"]
    print("  ✓ p50 = %.2f ms" % latency_before)

    # mAP 基线必须在剪枝之前测，否则就没有对照了
    maps: Dict[str, Optional[float]] = {}
    maps["mAP（剪枝前）"] = evaluate_map(yolo, net, args.data, imgsz, device, "剪枝前")
    if not args.data:
        print("  ! 未提供 --data，跳过 mAP 评估（best.pt 记录的是 dataset.yaml，本机不存在）")

    print("\n步骤 4/6  非结构化剪枝：准则=%s 范围=%s 目标稀疏度=%.2f"
          % (args.criterion, args.scope, args.sparsity))
    exclude = (head_prefix(net),) if args.protect_head else ()
    if exclude:
        print("  · 保护检测头：%s.*（如需剪它用 --no-protect-head）" % exclude[0])
    print("  · BN 权重%s参与排序" % ("会" if args.include_norm else "不"))
    grads = build_grads(net, args.criterion, imgsz, device)
    pruner = UnstructuredPruner(
        net,
        criterion=args.criterion,
        scope=args.scope,
        exclude=exclude,
        include_norm=args.include_norm,
        grads=grads,
    )
    prune_report = pruner.prune(args.sparsity)
    masks = pruner.masks
    assert masks is not None
    nonzero_after = count_nonzero_parameters(net, masks=masks)
    sparsity_after = model_sparsity(net, masks=masks)
    print("  ✓ 可剪参数稀疏度 %.4f，保留 %d / %d 个权重"
          % (prune_report.achieved_sparsity, prune_report.total_kept, prune_report.total_numel))
    print("  ✓ 全模型非零参数 %d → %d（%.1f%% 被置零）"
          % (nonzero_before, nonzero_after, 100 * (1 - nonzero_after / max(nonzero_before, 1))))

    print("\n步骤 5/6  评估与微调")
    maps["mAP（剪枝后未微调）"] = evaluate_map(yolo, net, args.data, imgsz, device, "剪枝后未微调")
    if args.finetune_epochs > 0:
        net = finetune_with_masks(yolo, net, masks, args.data, args.finetune_epochs, imgsz,
                                  device, args.out, lr0=args.lr0)
        maps["mAP（微调后）"] = evaluate_map(yolo, net, args.data, imgsz, device, "微调后")

    print("\n步骤 6/6  保存产物到 %s" % args.out)
    latency_after = benchmark_latency(net, example, warmup=3, iters=args.bench_iters,
                                      device=device)["p50_ms"]
    pruned_path = save_ultralytics_ckpt(net, source_ckpt, os.path.join(args.out, "pruned_best.pt"))
    masks_path = masks.save(os.path.join(args.out, "masks.pt"))

    meta: Dict[str, Any] = {
        "weights": args.weights,
        "num_classes": num_classes,
        "imgsz": imgsz,
        "criterion": args.criterion,
        "scope": args.scope,
        "sparsity": args.sparsity,
        "protect_head": args.protect_head,
        "params_dense": params_dense,
        "nonzero_before": nonzero_before,
        "nonzero_after": nonzero_after,
        "sparsity_after": sparsity_after,
        "duration": time.time() - started,
        "latency_before": latency_before,
        "latency_after": latency_after,
    }
    report = build_report(meta, prune_report, structured_ok, maps)
    saved = save_reports({"prune_report": report}, args.out)

    # 最后校验：产物能不能被 ultralytics 正常加载并前向
    reload_ok = False
    output_finite = False
    try:
        from ultralytics import YOLO

        reloaded = YOLO(pruned_path)
        with torch.no_grad():
            probe_out = reloaded.model.eval()(example[0].cpu())
        output_finite = bool(torch.isfinite(probe_out[0]).all())
        reload_ok = True
    except Exception as exc:  # noqa: BLE001
        print("  ! 重新加载校验失败：%s: %s" % (type(exc).__name__, str(exc)[:100]))

    print("  ✓ 剪枝后权重 : %s（YOLO() 可加载：%s，输出数值有限：%s）"
          % (pruned_path, "是" if reload_ok else "否", "是" if output_finite else "否"))
    print("  ✓ 掩码       : %s" % masks_path)
    print("  ✓ 报告       : %s / %s" % (saved["prune_report"]["markdown"], saved["prune_report"]["json"]))

    size_before = os.path.getsize(args.weights) / 1e6
    size_after = os.path.getsize(pruned_path) / 1e6
    print("  · 文件大小   : %.2f MB → %.2f MB（非结构化剪枝是「稠密存零」，"
          "文件不会因为多了零而变小）" % (size_before, size_after))

    print("\n" + report.to_text())
    print("\n提示：以上是**非结构化**剪枝，稠密 FLOPs 与延迟基本不变（实测 %.2f → %.2f ms）。"
          % (latency_before, latency_after))
    print("      要降计算量请用通道剪枝：extras/yolov8_torch_pruning/reproduce.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
