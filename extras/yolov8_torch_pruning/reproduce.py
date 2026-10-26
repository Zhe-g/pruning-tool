"""在 yolov8n 上复现 torch-pruning 官方 recipe：先把 C2f 换成无 chunk 的等价实现，再剪枝。

对应上游 examples/yolov8/yolov8_pruning.py（adapted from issue #147），
这里做了两处适配，以便跑在 ultralytics 8.3.x 上：

1. 直接复用原始 Bottleneck 列表，不再重新构造（新版 Bottleneck 的签名变了）；
2. 只切 cv1 的权重/BN，构造 cv0 与 cv1 两个独立卷积。
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn

import torch_pruning as tp
from ultralytics import YOLO


class C2fV2(nn.Module):
    """与 C2f 数学等价，但把 chunk 拆成两个独立 Conv，避免 split 的索引空间问题。"""

    def __init__(self, cv0, cv1, cv2, bottleneck, hidden: int) -> None:
        super().__init__()
        self.c = hidden
        self.cv0 = cv0
        self.cv1 = cv1
        self.cv2 = cv2
        self.m = bottleneck

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def _half_conv(old_conv, channels: int, part: int):
    """把原始 cv1（输出 2c）切成第 part 个 c 通道的 Conv（含 BN 参数迁移）。"""
    module = type(old_conv)(old_conv.conv.in_channels, channels, 1, 1)
    with torch.no_grad():
        sl = slice(part * channels, (part + 1) * channels)
        module.conv.weight.copy_(old_conv.conv.weight[sl])
        if old_conv.conv.bias is not None:
            module.conv.bias.copy_(old_conv.conv.bias[sl])
        for name in ("weight", "bias", "running_mean", "running_var"):
            source = getattr(old_conv.bn, name, None)
            target = getattr(module.bn, name, None)
            if isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor):
                target.copy_(source[sl])
    return module


def replace_c2f(module: nn.Module) -> int:
    """递归把 C2f 替换成 C2fV2，返回替换个数。"""
    from ultralytics.nn.modules import C2f

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, C2f):
            hidden = int(child.c)
            cv0 = _half_conv(child.cv1, hidden, 0)
            cv1 = _half_conv(child.cv1, hidden, 1)
            new = C2fV2(cv0, cv1, child.cv2, child.m, hidden)
            # ultralytics 的 _predict_once 依赖每层的 f/i/type/np 等元属性，
            # 这些不在 __dict__ 的参数里，必须显式搬过去（官方示例也是这么做的）
            for attr_name in dir(child):
                if attr_name.startswith("_") or attr_name in ("cv0", "cv1", "cv2", "m", "c"):
                    continue
                value = getattr(child, attr_name, None)
                if callable(value):
                    continue
                setattr(new, attr_name, value)
            setattr(module, name, new)
            replaced += 1
        else:
            replaced += replace_c2f(child)
    return replaced


def _max_diff(left, right) -> float:
    """递归比较嵌套的 tensor / list / tuple，返回最大绝对差。"""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return float((left - right).abs().max())
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError("输出结构长度不一致：%d vs %d" % (len(left), len(right)))
        return max([_max_diff(a, b) for a, b in zip(left, right)] or [0.0])
    raise TypeError("无法比较的类型：%s vs %s" % (type(left), type(right)))


def main() -> int:
    net = YOLO("yolov8n.yaml").model
    original = copy.deepcopy(net).eval()          # 保留一份未改造的，用于数值等价性对照
    net.train()
    count = replace_c2f(net)
    print("替换 C2f → C2fV2: %d 个" % count)

    x = torch.zeros(1, 3, 640, 640)

    # 1) 等价性：改造只应改变计算图的写法，不应改变数值
    net.eval()
    with torch.no_grad():
        before = original(x)
        after = net(x)
    print("改造前后最大数值差: %.3e（应接近 0）" % _max_diff(before, after))

    # 2) 不同剪枝比例下的收益（官方 recipe：忽略 Detect 头 + GroupNormPruner）
    base_macs, base_params = tp.utils.count_ops_and_params(original, x)
    print("基线: 参数 %.3f M, MACs %.2f G" % (base_params / 1e6, base_macs / 1e9))
    for ratio in (0.2, 0.5):
        model = copy.deepcopy(original).train()
        replace_c2f(model)
        ignored = [m for m in model.modules() if type(m).__name__ == "Detect"]
        pruner = tp.pruner.GroupNormPruner(
            model,
            x,
            importance=tp.importance.GroupMagnitudeImportance(),
            iterative_steps=1,
            pruning_ratio=ratio,
            ignored_layers=ignored,
        )
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, x)
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(
            "比例 %.1f: 参数 %.3f M (-%.1f%%), MACs %.2f G (-%.1f%%), 前向=%s"
            % (
                ratio,
                params / 1e6,
                100 * (1 - params / base_params),
                macs / 1e9,
                100 * (1 - macs / base_macs),
                isinstance(out, (tuple, list)),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
