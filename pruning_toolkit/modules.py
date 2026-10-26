"""模块语义层：判断一个 ``nn.Module`` 是否携带「通道」语义，以及如何安全地裁剪它。

剪枝工具必须知道两件事：

1. **这个模块的张量里哪一维是通道维**（Conv 是 1，Linear 是最后一维，BN 是 1……）；
2. **裁剪通道时该动哪些参数**（``weight`` / ``bias`` / ``running_mean`` ……）。

本模块把这两件事集中成 :func:`channel_semantics` 返回的 :class:`ChannelSemantics`
描述，并通过 :func:`resize_output_` / :func:`resize_input_` 提供原地修改能力。
所有形状相关的副作用都只在这里发生，其它模块只管调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "ChannelSemantics",
    "channel_semantics",
    "module_kind",
    "out_channels",
    "in_channels",
    "is_channel_preserving",
    "resize_output_",
    "resize_input_",
    "resize_both_",
    "iter_weight_parameters",
    "is_depthwise",
    "CONV_TYPES",
    "NORM_TYPES",
    "LINEAR_TYPES",
]

CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
CONV_TRANSPOSE_TYPES = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
NORM_TYPES = (
    nn.modules.batchnorm._BatchNorm,
    nn.modules.instancenorm._InstanceNorm,
    nn.GroupNorm,
    nn.LayerNorm,
)
LINEAR_TYPES = (nn.Linear,)
POOL_TYPES = (
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.FractionalMaxPool2d,
)
ACTIVATION_TYPES = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.ELU,
    nn.SELU,
    nn.GELU,
    nn.SiLU,
    nn.Mish,
    nn.Sigmoid,
    nn.Tanh,
    nn.Hardtanh,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.Softplus,
    nn.LogSigmoid,
    nn.Identity,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.Flatten,
    nn.Softmax,
    nn.LogSoftmax,
)

#: 默认参与非结构化剪枝的权重类型
PRUNABLE_WEIGHT_TYPES = CONV_TYPES + CONV_TRANSPOSE_TYPES + LINEAR_TYPES


@dataclass(frozen=True)
class ChannelSemantics:
    """描述一个模块的通道语义。

    属性
    ----
    kind:               模块类别，用于报告展示（conv / linear / norm / pool / activation / ...）
    consumes/produces:  输入侧 / 输出侧是否存在通道维
    in_count/out_count: 输入 / 输出通道数
    in_axis/out_axis:   通道维在张量中的位置（标准化后的**非负**下标）
    index_preserving:   输入与输出的通道索引空间是否相同
                        （BN、激活、池化、深度卷积都是；普通卷积不是）
    coupled:            裁剪时输入与输出权重维度必须**同时**按同一组索引裁剪（深度卷积）
    supported:          是否支持结构化裁剪；False 时会被依赖图当作屏障
    reason:             不支持的原因，用于报告
    """

    kind: str
    consumes: bool = False
    produces: bool = False
    in_count: Optional[int] = None
    out_count: Optional[int] = None
    in_axis: Optional[int] = None
    out_axis: Optional[int] = None
    index_preserving: bool = False
    coupled: bool = False
    supported: bool = True
    reason: str = ""

    @property
    def prunable_output(self) -> bool:
        return self.produces and self.supported and (self.out_count or 0) >= 2


def _unsupported(kind: str, reason: str) -> ChannelSemantics:
    return ChannelSemantics(kind=kind, supported=False, reason=reason)


# ---------------------------------------------------------------------- 基本查询


def is_depthwise(module: nn.Module) -> bool:
    """是否为「深度卷积」：分组数等于输入通道数且输入输出通道数相同。

    这种卷积的输入通道 c 只影响输出通道 c，因此输入输出的通道索引空间一致，
    可以在剪掉输出通道时同步剪掉对应输入通道。
    """
    if not isinstance(module, CONV_TYPES + CONV_TRANSPOSE_TYPES):
        return False
    groups = int(getattr(module, "groups", 1))
    return groups > 1 and groups == int(module.in_channels) == int(module.out_channels)


def out_channels(module: nn.Module) -> Optional[int]:
    if isinstance(module, CONV_TYPES + CONV_TRANSPOSE_TYPES):
        return int(module.out_channels)
    if isinstance(module, LINEAR_TYPES):
        return int(module.out_features)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return int(module.num_features)
    if isinstance(module, nn.modules.instancenorm._InstanceNorm):
        return int(module.num_features)
    if isinstance(module, nn.GroupNorm):
        return int(module.num_channels)
    if isinstance(module, nn.LayerNorm):
        shape = module.normalized_shape
        if len(shape) == 1:
            return int(shape[0])
        return None
    if isinstance(module, nn.PReLU):
        return None  # 取决于输入，由调用方按输入通道推断
    return None


def in_channels(module: nn.Module) -> Optional[int]:
    if isinstance(module, CONV_TYPES):
        return int(module.in_channels)
    if isinstance(module, CONV_TRANSPOSE_TYPES):
        return int(module.in_channels)
    if isinstance(module, LINEAR_TYPES):
        return int(module.in_features)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return int(module.num_features)
    if isinstance(module, nn.modules.instancenorm._InstanceNorm):
        return int(module.num_features)
    if isinstance(module, nn.GroupNorm):
        return int(module.num_channels)
    if isinstance(module, nn.LayerNorm):
        shape = module.normalized_shape
        if len(shape) == 1:
            return int(shape[0])
        return None
    return None


def module_kind(module: nn.Module) -> str:
    """给模块分类，仅用于报告与调试。"""
    if isinstance(module, CONV_TYPES):
        return "conv"
    if isinstance(module, CONV_TRANSPOSE_TYPES):
        return "conv_transpose"
    if isinstance(module, LINEAR_TYPES):
        return "linear"
    if isinstance(module, NORM_TYPES):
        return "norm"
    if isinstance(module, nn.PReLU):
        return "prelu"
    if isinstance(module, nn.Embedding):
        return "embedding"
    if isinstance(module, POOL_TYPES):
        return "pool"
    if isinstance(module, ACTIVATION_TYPES):
        return "activation"
    return "other"


def is_channel_preserving(module: nn.Module) -> bool:
    """模块是否逐通道独立（通道数不变且不做通道间混合）。"""
    return channel_semantics(module).index_preserving


# ---------------------------------------------------------------------- 语义解析


def channel_semantics(module: nn.Module) -> ChannelSemantics:
    """返回模块的通道语义。遇到无法安全处理的模块会返回 ``supported=False``。"""
    # ---- 卷积 -------------------------------------------------------------
    if isinstance(module, CONV_TYPES):
        groups = int(getattr(module, "groups", 1))
        in_c, out_c = int(module.in_channels), int(module.out_channels)
        if groups == 1:
            return ChannelSemantics(
                kind="conv",
                consumes=True,
                produces=True,
                in_count=in_c,
                out_count=out_c,
                in_axis=1,
                out_axis=1,
                index_preserving=False,
                coupled=False,
            )
        if groups == in_c == out_c:
            return ChannelSemantics(
                kind="conv",
                consumes=True,
                produces=True,
                in_count=in_c,
                out_count=out_c,
                in_axis=1,
                out_axis=1,
                index_preserving=True,
                coupled=True,
            )
        return _unsupported(
            "conv",
            "分组卷积 groups=%d 既不是普通卷积也不是深度卷积（in=%d, out=%d），"
            "输入输出通道不是一一对应，无法安全传播" % (groups, in_c, out_c),
        )

    if isinstance(module, CONV_TRANSPOSE_TYPES):
        groups = int(getattr(module, "groups", 1))
        in_c, out_c = int(module.in_channels), int(module.out_channels)
        if groups == 1:
            return ChannelSemantics(
                kind="conv_transpose",
                consumes=True,
                produces=True,
                in_count=in_c,
                out_count=out_c,
                in_axis=1,
                out_axis=1,
                index_preserving=False,
            )
        if groups == in_c == out_c:
            return ChannelSemantics(
                kind="conv_transpose",
                consumes=True,
                produces=True,
                in_count=in_c,
                out_count=out_c,
                in_axis=1,
                out_axis=1,
                index_preserving=True,
                coupled=True,
            )
        return _unsupported(
            "conv_transpose",
            "转置分组卷积 groups=%d（in=%d, out=%d）无法安全传播" % (groups, in_c, out_c),
        )

    # ---- 线性 -------------------------------------------------------------
    if isinstance(module, LINEAR_TYPES):
        return ChannelSemantics(
            kind="linear",
            consumes=True,
            produces=True,
            in_count=int(module.in_features),
            out_count=int(module.out_features),
            in_axis=-1,
            out_axis=-1,
            index_preserving=False,
        )

    # ---- 归一化 -----------------------------------------------------------
    if isinstance(module, nn.GroupNorm):
        if int(module.num_groups) != 1:
            return _unsupported(
                "norm",
                "GroupNorm(num_groups=%d) 会在组内混合通道，裁剪单通道会破坏统计量"
                % int(module.num_groups),
            )
        return ChannelSemantics(
            kind="norm",
            consumes=True,
            produces=True,
            in_count=int(module.num_channels),
            out_count=int(module.num_channels),
            in_axis=1,
            out_axis=1,
            index_preserving=True,
        )

    if isinstance(module, nn.LayerNorm):
        if len(module.normalized_shape) != 1:
            return _unsupported(
                "norm",
                "LayerNorm(normalized_shape=%s) 归一化的是多维特征块，无法按通道裁剪"
                % (tuple(module.normalized_shape),),
            )
        count = int(module.normalized_shape[0])
        return ChannelSemantics(
            kind="norm",
            consumes=True,
            produces=True,
            in_count=count,
            out_count=count,
            in_axis=-1,
            out_axis=-1,
            index_preserving=True,
        )

    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm)):
        count = int(module.num_features)
        return ChannelSemantics(
            kind="norm",
            consumes=True,
            produces=True,
            in_count=count,
            out_count=count,
            in_axis=1,
            out_axis=1,
            index_preserving=True,
        )

    # ---- PReLU：num_parameters=1 时是逐元素广播，不涉及通道 -----------------
    if isinstance(module, nn.PReLU):
        num = int(module.num_parameters)
        if num == 1:
            return ChannelSemantics(
                kind="prelu",
                consumes=True,
                produces=True,
                in_axis=1,
                out_axis=1,
                index_preserving=True,
            )
        return ChannelSemantics(
            kind="prelu",
            consumes=True,
            produces=True,
            in_count=num,
            out_count=num,
            in_axis=1,
            out_axis=1,
            index_preserving=True,
        )

    # ---- 池化 / 激活：逐通道，无参数 --------------------------------------
    if isinstance(module, POOL_TYPES) or isinstance(module, ACTIVATION_TYPES):
        kind = "pool" if isinstance(module, POOL_TYPES) else "activation"
        return ChannelSemantics(kind=kind, consumes=True, produces=True, in_axis=1, out_axis=1,
                                index_preserving=True)

    # ---- 其余一律视为屏障 -------------------------------------------------
    if isinstance(module, nn.Embedding):
        return _unsupported("embedding", "Embedding 的「输出维度」是词向量维度，不支持结构化裁剪")
    return _unsupported(
        module_kind(module),
        "%s 不在支持的结构化裁剪模块列表中（可加入 exclude 或改用非结构化剪枝）"
        % type(module).__name__,
    )


# ---------------------------------------------------------------------- 参数修改


def _index_select_parameter(
    param: Optional[nn.Parameter], keep: torch.Tensor, dim: int
) -> Optional[nn.Parameter]:
    if param is None:
        return None
    data = param.data.index_select(dim, keep.to(param.data.device)).contiguous()
    return nn.Parameter(data, requires_grad=param.requires_grad)


def _plain_parameter(
    tensor: Optional[torch.Tensor], keep: torch.Tensor, dim: int
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.index_select(dim, keep.to(tensor.device)).contiguous()


def resize_output_(module: nn.Module, keep: torch.Tensor) -> None:
    """按索引 ``keep`` 原地裁剪模块的**输出通道**及其所有相关参数。"""
    semantics = channel_semantics(module)
    if not semantics.produces or semantics.out_count is None:
        raise TypeError("%s 没有可裁剪的输出通道" % type(module).__name__)
    new_count = int(keep.numel())

    if isinstance(module, CONV_TYPES):
        module.weight = _index_select_parameter(module.weight, keep, 0)
        module.bias = _index_select_parameter(module.bias, keep, 0)
        module.out_channels = new_count
        return

    if isinstance(module, CONV_TRANSPOSE_TYPES):
        # ConvTranspose 的权重布局是 (in_channels, out_channels/groups, *kernel)
        module.weight = _index_select_parameter(module.weight, keep, 1)
        module.bias = _index_select_parameter(module.bias, keep, 0)
        module.out_channels = new_count
        return

    if isinstance(module, LINEAR_TYPES):
        module.weight = _index_select_parameter(module.weight, keep, 0)
        module.bias = _index_select_parameter(module.bias, keep, 0)
        module.out_features = new_count
        return

    if isinstance(module, nn.GroupNorm):
        module.weight = _index_select_parameter(module.weight, keep, 0)
        module.bias = _index_select_parameter(module.bias, keep, 0)
        module.num_channels = new_count
        return

    if isinstance(module, nn.LayerNorm):
        module.weight = _index_select_parameter(module.weight, keep, 0)
        module.bias = _index_select_parameter(module.bias, keep, 0)
        module.normalized_shape = (new_count,)
        return

    if isinstance(module, nn.PReLU):
        if int(module.num_parameters) == 1:
            return  # 广播参数，无需改动
        module.weight = _index_select_parameter(module.weight, keep, 0)
        module.num_parameters = new_count
        return

    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm)):
        module.weight = _index_select_parameter(module.weight, keep, 0)
        module.bias = _index_select_parameter(module.bias, keep, 0)
        if getattr(module, "running_mean", None) is not None:
            module.running_mean = _plain_parameter(module.running_mean, keep, 0)
        if getattr(module, "running_var", None) is not None:
            module.running_var = _plain_parameter(module.running_var, keep, 0)
        module.num_features = new_count
        return

    raise TypeError("模块 %s 不支持输出通道裁剪" % type(module).__name__)


def resize_input_(module: nn.Module, keep: torch.Tensor) -> None:
    """按索引 ``keep`` 原地裁剪模块的**输入通道**及其相关参数。"""
    semantics = channel_semantics(module)
    if not semantics.consumes or semantics.in_count is None:
        raise TypeError("%s 没有可裁剪的输入通道" % type(module).__name__)
    new_count = int(keep.numel())

    if isinstance(module, CONV_TYPES):
        module.weight = _index_select_parameter(module.weight, keep, 1)
        module.in_channels = new_count
        return

    if isinstance(module, CONV_TRANSPOSE_TYPES):
        module.weight = _index_select_parameter(module.weight, keep, 0)
        module.in_channels = new_count
        return

    if isinstance(module, LINEAR_TYPES):
        module.weight = _index_select_parameter(module.weight, keep, 1)
        module.in_features = new_count
        return

    # 归一化 / PReLU 的输入输出索引空间相同，用输出侧实现即可
    resize_output_(module, keep)


def resize_both_(module: nn.Module, keep: torch.Tensor) -> None:
    """输入与输出通道同属一个索引空间时，按同一组索引同时裁剪两侧。

    典型场景：

    * 残差块内部的 ``conv1``：它读入的是块输入张量、输出的张量又经短接回到同一
      通道空间，因此输入输出必须一起裁；
    * 深度卷积：输入通道 c 只影响输出通道 c（此时还会同步更新 ``groups``）。
    """
    semantics = channel_semantics(module)
    new_count = int(keep.numel())

    if semantics.index_preserving and not semantics.coupled:
        # 归一化 / 激活类：只有一个通道维
        resize_output_(module, keep)
        return

    if semantics.coupled:
        # 深度卷积：输入输出 1:1 耦合，但权重第 1 维是 in_channels/groups == 1，
        # 只需按输出通道取索引，同时把 groups 改成新的通道数
        resize_coupled_(module, keep)
        return

    device = None
    weight = None
    if getattr(module, "weight", None) is not None and isinstance(module.weight, nn.Parameter):
        device = module.weight.device
    index = keep.to(device) if device is not None else keep

    if isinstance(module, CONV_TYPES):
        weight = module.weight.data.index_select(0, index).index_select(1, index).contiguous()
        module.weight = nn.Parameter(weight, requires_grad=module.weight.requires_grad)
        module.bias = _index_select_parameter(module.bias, keep, 0)
        module.in_channels = new_count
        module.out_channels = new_count
        return

    if isinstance(module, CONV_TRANSPOSE_TYPES):
        weight = module.weight.data.index_select(0, index).index_select(1, index).contiguous()
        module.weight = nn.Parameter(weight, requires_grad=module.weight.requires_grad)
        module.bias = _index_select_parameter(module.bias, keep, 0)
        module.in_channels = new_count
        module.out_channels = new_count
        return

    if isinstance(module, LINEAR_TYPES):
        weight = module.weight.data.index_select(0, index).index_select(1, index).contiguous()
        module.weight = nn.Parameter(weight, requires_grad=module.weight.requires_grad)
        module.bias = _index_select_parameter(module.bias, keep, 0)
        module.in_features = new_count
        module.out_features = new_count
        return

    raise TypeError("模块 %s 不支持输入输出同时裁剪" % type(module).__name__)


def resize_coupled_(module: nn.Module, keep: torch.Tensor) -> None:
    """深度卷积：输入与输出通道一一对应，必须同时裁剪。

    深度卷积的权重形状是 ``(out_channels, 1, *kernel)``（因为
    ``in_channels / groups == 1``），所以只按输出通道取索引即可，
    但 ``groups`` / ``in_channels`` / ``out_channels`` 都要同步更新为新的通道数。

    只接受真正的深度卷积（``groups == in_channels == out_channels``）；
    普通卷积请用 :func:`resize_output_` / :func:`resize_input_`，
    输入输出同索引空间的普通卷积请用 :func:`resize_both_`。
    """
    if not is_depthwise(module):
        raise TypeError(
            "%s 不是深度卷积（要求 groups == in_channels == out_channels），"
            "不能使用 coupled 裁剪" % type(module).__name__
        )
    new_count = int(keep.numel())
    module.weight = _index_select_parameter(module.weight, keep, 0)
    module.bias = _index_select_parameter(module.bias, keep, 0)
    module.groups = new_count
    module.in_channels = new_count
    module.out_channels = new_count


# ---------------------------------------------------------------------- 参数枚举


def iter_weight_parameters(
    model: nn.Module,
    include_bias: bool = False,
    include_norm: bool = True,
) -> Iterator[Tuple[str, nn.Parameter]]:
    """遍历适合非结构化剪枝的权重参数。

    参数
    ----
    include_bias: 是否包含偏置（默认不含，偏置通常只有很小一部分参数量）
    include_norm: 是否包含归一化层的仿射权重
    """
    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        if isinstance(module, PRUNABLE_WEIGHT_TYPES):
            yield prefix + "weight", module.weight
            if include_bias and getattr(module, "bias", None) is not None:
                yield prefix + "bias", module.bias  # type: ignore[arg-type]
        elif include_norm and isinstance(module, NORM_TYPES):
            if getattr(module, "weight", None) is not None:
                yield prefix + "weight", module.weight  # type: ignore[arg-type]
        elif include_norm and isinstance(module, nn.PReLU) and int(module.num_parameters) > 1:
            yield prefix + "weight", module.weight
