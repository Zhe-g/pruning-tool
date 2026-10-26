"""通道依赖图：结构化剪枝的「安全检查」与「传播规则」。

结构化剪枝必须回答一个问题：**删掉某一层的某个通道，还有哪些地方的维度必须跟着改？**
本模块用 ``torch.fx`` 追踪模型，把「共享同一个通道索引空间」的张量位置（称为 *槽位 slot*）
用并查集合并成 *通道组 ChannelGroup*，再判定每个组是否可以被安全裁剪。

三类角色
--------
* **槽位**：``(节点名, in|out)``，代表某节点输入/输出张量的通道维（含轴位置与通道数）。
* **通道组**：一组索引空间相同的槽位。组内被删的通道号对所有成员同时生效。
* **编辑（GroupEdit）**：裁剪某组时需要执行的参数改动，分 ``out`` / ``in`` / ``both`` 三种。

传播规则
--------
+---------------------------+----------------------------------------------------------+
| 结构                      | 规则                                                     |
+===========================+==========================================================+
| 普通卷积 / 线性层         | 输入槽位与输出槽位是**不同**索引空间，各成一组           |
| 归一化 / 激活 / 池化      | 逐通道独立，输入输出合并为同一组                         |
| 深度卷积                  | 输入通道 c 只对应输出通道 c，合并为一组并同步裁剪         |
| 残差相加 ``add``          | 两路张量通道数相同则合并（残差要求联合裁剪）             |
| 逐元素乘 ``mul``          | 同上（如 SE 门控、注意力权重）                           |
| 拼接 ``cat``              | 各分支索引空间无法对应，作为**屏障**并整体保护            |
| 形状变换（view/permute…） | 作为屏障；被屏障波及的组一律不剪                         |
+---------------------------+----------------------------------------------------------+

屏障（barrier）是这里的安全阀：无法证明安全时**宁可不剪**，并在报告里写明原因，
而不是产出一个前向报错的模型。
"""

from __future__ import annotations

import operator
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.fx as fx
import torch.nn as nn

from . import report as report_mod
from .modules import (
    ChannelSemantics,
    channel_semantics,
    resize_both_,
    resize_input_,
    resize_output_,
)

__all__ = [
    "ChannelGroup",
    "GroupEdit",
    "DependencyGraph",
    "UnsupportedGraphError",
    "build_graph",
]


class UnsupportedGraphError(RuntimeError):
    """torch.fx 无法追踪该模型（存在动态控制流等）时抛出。"""


# ---------------------------------------------------------------------- 算子分类

#: 逐通道直通算子：只改空间尺寸，不改通道数与通道位置
PASS_THROUGH_OPS = frozenset(
    {
        "relu", "relu6", "leaky_relu", "gelu", "silu", "swish", "selu", "elu", "celu", "mish",
        "hardsigmoid", "hardswish", "hardtanh", "sigmoid", "tanh", "softplus", "logsigmoid",
        "softsign", "dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout",
        "feature_alpha_dropout", "max_pool1d", "max_pool2d", "max_pool3d",
        "avg_pool1d", "avg_pool2d", "avg_pool3d", "adaptive_avg_pool1d", "adaptive_avg_pool2d",
        "adaptive_avg_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",
        "fractional_max_pool2d", "lp_pool2d", "flatten", "contiguous", "clone", "detach",
        "softmax", "log_softmax", "normalize", "channel_shuffle_placeholder",
    }
)

#: 逐元素二元算子：两路通道索引空间一致，可以合并（要求通道数相同）
ELEMENTWISE_BINARY_OPS = frozenset(
    {
        "add", "__add__", "iadd", "sub", "__sub__", "isub", "mul", "__mul__", "imul",
        "div", "__truediv__", "itruediv", "rsub", "rmul", "rdiv", "add_", "sub_", "mul_", "div_",
    }
)

#: 归约算子：只有当被归约的维度不包含通道维、且保留维度时才是直通
REDUCTION_OPS = frozenset({"mean", "sum", "amax", "amin", "max", "min", "prod", "var", "std"})

#: 已知会改变索引空间的算子，直接作为屏障
BARRIER_OPS = frozenset(
    {
        "cat", "concat", "stack", "chunk", "split", "unbind", "getitem", "__getitem__",
        "select", "narrow", "slice", "view", "reshape", "permute", "transpose", "swapaxes",
        "flatten_dense", "matmul", "__matmul__", "bmm", "einsum", "interpolate", "upsample",
        "upsample_bilinear", "upsample_nearest", "grid_sampler", "repeat", "expand", "tile",
        "pixel_shuffle", "pixel_unshuffle", "broadcast_tensors", "where", "masked_fill",
        "index_select", "gather", "scatter", "topk", "sort", "argmax", "argmin", "roll", "flip",
        "pad", "constant_pad_nd", "reflection_pad2d", "replication_pad2d", "linspace", "zeros",
        "ones", "full", "arange", "randn", "rand", "conv2d", "linear", "batch_norm",
    }
)


def _op_name(target: Any) -> str:
    """取得算子的短名字，用于分类。"""
    name = getattr(target, "__name__", None)
    if name is None:
        text = str(target)
        name = text.rsplit(".", 1)[-1] if "." in text else text
    if name.endswith("_") and not name.startswith("__"):
        name = name[:-1]
    return name


# ---------------------------------------------------------------------- 数据结构


@dataclass
class SlotInfo:
    """一个通道槽位。"""

    key: str
    node_name: str
    role: str  # 'in' | 'out'
    axis: int
    count: int
    label: str = ""


@dataclass
class GroupEdit:
    """裁剪某个通道组时，要对某个模块执行的参数改动。"""

    module_name: str
    kind: str  # 'out' | 'in' | 'both'

    def __post_init__(self) -> None:
        if self.kind not in ("out", "in", "both"):
            raise ValueError("未知的编辑类型 %r" % self.kind)


@dataclass
class ChannelGroup:
    """一组共享通道索引空间的槽位，是结构化剪枝的最小单位。"""

    index: int
    count: int
    slots: List[str]
    edits: List[GroupEdit]
    prunable: bool = True
    reason: str = ""

    @property
    def module_names(self) -> List[str]:
        seen: List[str] = []
        for edit in self.edits:
            if edit.module_name not in seen:
                seen.append(edit.module_name)
        return seen

    @property
    def output_edits(self) -> List[GroupEdit]:
        """输出侧编辑：这些模块的通道重要性会被评分。"""
        return [edit for edit in self.edits if edit.kind in ("out", "both")]

    @property
    def output_modules(self) -> List[str]:
        return [edit.module_name for edit in self.output_edits]

    @property
    def display(self) -> str:
        return "+".join(self.module_names) if self.module_names else "group%d" % self.index

    def apply(self, model: nn.Module, keep: torch.Tensor) -> None:
        """按保留索引 ``keep`` 原地裁剪本组涉及的所有模块。"""
        for edit in self.edits:
            module = model.get_submodule(edit.module_name)
            if edit.kind == "out":
                resize_output_(module, keep)
            elif edit.kind == "in":
                resize_input_(module, keep)
            else:
                resize_both_(module, keep)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "count": self.count,
            "modules": self.module_names,
            "edits": [{"module": e.module_name, "kind": e.kind} for e in self.edits],
            "prunable": self.prunable,
            "reason": self.reason,
            "slots": list(self.slots),
        }


@dataclass
class NodeRecord:
    """追踪过程中对每个 fx 节点的记录，用于报告与 FLOPs 统计。"""

    node_name: str
    op: str
    module_name: Optional[str] = None
    kind: str = ""
    semantics: Optional[ChannelSemantics] = None
    in_slot: Optional[str] = None
    out_slot: Optional[str] = None
    in_shape: Optional[Tuple[int, ...]] = None
    out_shape: Optional[Tuple[int, ...]] = None
    flops: int = 0
    note: str = ""

    @property
    def needs_resize(self) -> bool:
        sem = self.semantics
        return bool(sem is not None and (sem.in_count is not None or sem.out_count is not None))


# ---------------------------------------------------------------------- 依赖图


class DependencyGraph:
    """基于 ``torch.fx`` 的通道依赖图。

    参数
    ----
    model:           待分析模型（会被建立追踪副本，不改变权重）
    example_inputs:  与模型 ``forward`` 位置参数一致的示例输入，用于形状传播
    protect_output:  是否默认保护直接产生模型输出的通道组（通常是分类头）
    ignore:          需要排除在结构化剪枝之外的模块名

    典型用法::

        graph = DependencyGraph(model, (torch.randn(8, 3, 32, 32),))
        for group in graph.prunable_groups():
            print(group.display, group.count)
    """

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Sequence[torch.Tensor],
        protect_output: bool = True,
        ignore: Iterable[str] = (),
    ) -> None:
        self.model = model
        self.example_inputs = tuple(example_inputs)
        self.protect_output = protect_output
        self.ignore: Set[str] = set(ignore)

        self._parent: Dict[str, str] = {}
        self._slots: "OrderedDict[str, SlotInfo]" = OrderedDict()
        self._records: "OrderedDict[str, NodeRecord]" = OrderedDict()
        self._out_slot_of_node: Dict[str, str] = {}
        self.barriers: List[Dict[str, Any]] = []
        self._barrier_slots: List[Tuple[str, str]] = []  # (槽位键, 原因)
        self._protected_slots: List[str] = []
        #: 输入张量来自模型输入或屏障节点的槽位（这些张量**无法**被缩小）
        self._external_in_slots: List[str] = []

        self.gm: fx.GraphModule = self._trace()
        self._analyze()
        self._build_groups()

    # ------------------------------------------------------------------ 追踪
    def _trace(self) -> fx.GraphModule:
        modes = [(module, module.training) for module in self.model.modules()]
        for module, _ in modes:
            module.eval()  # 避免追踪过程更新 BatchNorm 统计量
        try:
            gm = fx.symbolic_trace(self.model)
            from torch.fx.passes.shape_prop import ShapeProp

            ShapeProp(gm).propagate(*self.example_inputs)
        except UnsupportedGraphError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一转换成可读异常
            raise UnsupportedGraphError(
                "torch.fx 无法追踪该模型（%s: %s）。常见原因是 forward 中使用了"
                "依赖张量数值的动态控制流、或使用了 fx 不支持的 Python 结构。"
                "可以改为只使用非结构化剪枝，或把模型改写成静态图形式。"
                % (type(exc).__name__, exc)
            ) from exc
        finally:
            for module, training in modes:
                module.training = training
        return gm

    # ------------------------------------------------------------------ 形状工具
    def _shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        meta = node.meta.get("tensor_meta")
        if meta is None or not hasattr(meta, "shape"):
            return None
        return tuple(int(dim) for dim in meta.shape)

    def _shape_of(self, arg: Any) -> Optional[Tuple[int, ...]]:
        if isinstance(arg, fx.Node):
            if arg.op == "placeholder":
                return self._placeholder_shape(arg)
            return self._shape(arg)
        return None

    def _placeholder_shape(self, node: fx.Node) -> Optional[Tuple[int, ...]]:
        """占位符（模型输入）的形状：ShapeProp 不写入其 meta，按顺序对应示例输入。"""
        placeholders = [n for n in self.gm.graph.nodes if n.op == "placeholder"]
        try:
            position = placeholders.index(node)
        except ValueError:  # pragma: no cover - 不可能发生
            return None
        if position >= len(self.example_inputs):
            return None
        value = self.example_inputs[position]
        if isinstance(value, torch.Tensor):
            return tuple(int(dim) for dim in value.shape)
        return None

    @staticmethod
    def _tensor_args(node: fx.Node) -> List[fx.Node]:
        """递归收集节点参数中的所有张量节点（跳过常量与嵌套容器）。"""
        found: List[fx.Node] = []

        def walk(value: Any) -> None:
            if isinstance(value, fx.Node):
                found.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)

        for arg in node.args:
            walk(arg)
        for arg in node.kwargs.values():
            walk(arg)
        return found

    def _resolve_axis(
        self, shape: Tuple[int, ...], count: int, preferred: Optional[int]
    ) -> Optional[int]:
        """在 ``shape`` 中定位通道维：优先使用 ``preferred``，否则按通道数唯一匹配。"""
        ndim = len(shape)
        if count is None or count < 1:
            return None
        if preferred is not None:
            axis = preferred if preferred >= 0 else ndim + preferred
            if 0 <= axis < ndim and shape[axis] == count:
                return axis
        candidates = [axis for axis, dim in enumerate(shape) if dim == count]
        if len(candidates) == 1:
            return candidates[0]
        return None

    # ------------------------------------------------------------------ 并查集
    def _find(self, key: str) -> str:
        parent = self._parent.get(key, key)
        while parent != key:
            grandparent = self._parent.get(parent, parent)
            self._parent[key] = grandparent
            key, parent = parent, grandparent
        return key

    def _union(self, left: str, right: str) -> bool:
        """合并两个槽位；通道数不一致时拒绝并记为屏障。"""
        root_left, root_right = self._find(left), self._find(right)
        if root_left == root_right:
            return True
        count_left = self._slots[root_left].count
        count_right = self._slots[root_right].count
        if count_left != count_right:
            self._mark_barrier(
                [left, right],
                "通道数不一致（%d vs %d），无法合并索引空间" % (count_left, count_right),
            )
            return False
        self._parent[root_right] = root_left
        return True

    def _add_slot(self, key: str, node_name: str, role: str, axis: int, count: int, label: str = "") -> str:
        if key not in self._slots:
            self._slots[key] = SlotInfo(key=key, node_name=node_name, role=role, axis=axis,
                                        count=count, label=label)
            self._parent[key] = key
        return key

    def _mark_barrier(self, slot_keys: Sequence[str], reason: str) -> None:
        for key in slot_keys:
            if key in self._slots:
                self._barrier_slots.append((key, reason))

    def _record_barrier(self, node: fx.Node, reason: str) -> None:
        self.barriers.append({"node": node.name, "op": _op_name(node.target) if node.op != "call_module" else str(node.target), "reason": reason})

    # ------------------------------------------------------------------ 分析主流程
    def _analyze(self) -> None:
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                self._analyze_output(node)
                continue
            if node.op == "call_module":
                self._analyze_module_node(node)
            elif node.op == "call_function":
                self._analyze_function_node(node)
            elif node.op == "call_method":
                self._analyze_method_node(node)
            else:  # get_attr 等
                continue

    def _analyze_output(self, node: fx.Node) -> None:
        """模型输出所在张量的通道空间默认保护（分类头不能被剪掉类别数）。

        ``protect_output=False`` 时该通道组会变为可裁剪。请注意：裁剪类别维会让模型
        的输出形状从 ``(N, num_classes)`` 变成 ``(N, keep)``，
        :meth:`StructuredPruner.prune` 的前向校验会因此报错——那是有意的保护，
        确实想剪类别维时请显式传 ``verify=False``。
        """
        if not self.protect_output:
            return
        for arg in self._tensor_args(node):
            slot = self._out_slot_of_node.get(arg.name)
            if slot is not None:
                self._protected_slots.append(slot)

    # ---- 模块节点 --------------------------------------------------------
    def _analyze_module_node(self, node: fx.Node) -> None:
        module_name = str(node.target)
        module = self.gm.get_submodule(module_name)
        sem = channel_semantics(module)
        record = NodeRecord(
            node_name=node.name,
            op="call_module",
            module_name=module_name,
            kind=sem.kind or type(module).__name__,
            semantics=sem,
            in_shape=self._shape_of(self._tensor_args(node)[0]) if self._tensor_args(node) else None,
            out_shape=self._shape(node),
        )
        self._records[node.name] = record

        if not sem.supported:
            record.note = sem.reason
            self._record_barrier(node, sem.reason)
            self._barrier_upstream(node, sem.reason)
            return

        args = self._tensor_args(node)
        in_slot: Optional[str] = None
        out_slot: Optional[str] = None

        # 继承型模块（池化 / 激活 / PReLU(num_parameters=1)）：输入输出同为一片通道空间
        if sem.in_count is None:
            slot = self._inherit_slot(node, args, record)
            if slot is None:
                return
            out_slot = slot
            self._out_slot_of_node[node.name] = slot
            record.out_slot = slot
            self._records[node.name] = record
            return

        # 已知通道数的模块：建立输入 / 输出两个槽位
        if sem.consumes:
            if record.in_shape is None:
                self._fail_module(node, record, "拿不到输入形状，无法定位输入通道维")
                return
            axis = self._resolve_axis(record.in_shape, int(sem.in_count), sem.in_axis)
            if axis is None:
                self._fail_module(
                    node, record,
                    "在输入形状 %s 中无法唯一定位 %d 个输入通道"
                    % (record.in_shape, sem.in_count),
                )
                return
            in_slot = self._add_slot(
                "%s:in" % node.name, node.name, "in", axis, int(sem.in_count),
                label="%s.in" % module_name,
            )
            record.in_slot = in_slot
            producer = self._out_slot_of_node.get(args[0].name) if args else None
            if producer is not None:
                self._union(producer, in_slot)
            else:
                # 生产者没有槽位：输入来自模型输入或屏障节点，这个张量的宽度改不了。
                # 对普通卷积无所谓（它的输入槽位自成一組，本来就不可剪），
                # 但对深度卷积/归一化这类「输入输出同索引空间」的模块是致命的：
                # 它们会把输入维度和输出维度绑在一起剪，而输入根本缩不了。
                self._external_in_slots.append(in_slot)

        if sem.produces:
            if record.out_shape is None:
                self._fail_module(node, record, "拿不到输出形状，无法定位输出通道维")
                return
            axis = self._resolve_axis(record.out_shape, int(sem.out_count), sem.out_axis)
            if axis is None:
                self._fail_module(
                    node, record,
                    "在输出形状 %s 中无法唯一定位 %d 个输出通道"
                    % (record.out_shape, sem.out_count),
                )
                return
            out_slot = self._add_slot(
                "%s:out" % node.name, node.name, "out", axis, int(sem.out_count),
                label="%s.out" % module_name,
            )
            record.out_slot = out_slot
            self._out_slot_of_node[node.name] = out_slot
            if sem.index_preserving and in_slot is not None:
                self._union(in_slot, out_slot)

        record.flops = self._module_flops(node, module, record)
        self._records[node.name] = record

    def _fail_module(self, node: fx.Node, record: NodeRecord, reason: str) -> None:
        record.note = reason
        if record.out_slot:
            self._mark_barrier([record.out_slot], reason)
        if record.in_slot:
            self._mark_barrier([record.in_slot], reason)
        self._record_barrier(node, reason)
        self._barrier_upstream(node, reason)

    def _barrier_upstream(self, node: fx.Node, reason: str) -> None:
        """屏障节点会「污染」其输入侧通道空间：剪了上游就会让屏障后的维度对不上。"""
        slots = []
        for arg in self._tensor_args(node):
            slot = self._out_slot_of_node.get(arg.name)
            if slot is not None:
                slots.append(slot)
        if slots:
            self._mark_barrier(slots, "上游存在屏障节点：%s" % reason)

    def _inherit_slot(
        self, node: fx.Node, args: Sequence[fx.Node], record: NodeRecord
    ) -> Optional[str]:
        """为直通节点建立继承自输入的输出槽位。"""
        producer_slots = [
            self._out_slot_of_node[arg.name] for arg in args if arg.name in self._out_slot_of_node
        ]
        if not producer_slots:
            self._fail_module(node, record, "上游通道空间未知（上游是屏障节点或模型输入）")
            return None
        counts = {self._slots[slot].count for slot in producer_slots}
        if len(counts) != 1:
            self._fail_module(
                node, record, "多输入通道数不一致：%s" % sorted(counts)
            )
            return None
        base = self._slots[producer_slots[0]]
        count = base.count
        out_shape = record.out_shape
        axis = base.axis
        # 直通算子必须保持通道维位置不变。这里刻意**不做**「按通道数重新搜索轴」的兜底：
        # 当 H 或 W 恰好等于通道数时（如 32 通道的 32x32 特征图），搜索会把空间维
        # 误判成通道维。宁可判为屏障，也不产生错误的剪枝结果。
        if out_shape is not None:
            if axis >= len(out_shape) or out_shape[axis] != count:
                self._fail_module(
                    node, record,
                    "输出形状 %s 在第 %d 维上不是 %d 个通道，该算子改变了通道维位置"
                    % (out_shape, axis, count),
                )
                return None
        slot = self._add_slot(
            "%s:out" % node.name, node.name, "out", axis, count, label="%s.out" % record.module_name
        )
        for producer in producer_slots:
            self._union(producer, slot)
        record.out_slot = slot
        # 必须登记到节点表，否则下游节点找不到生产者，通道链会在此处断开
        self._out_slot_of_node[node.name] = slot
        return slot

    # ---- 函数 / 方法节点 -------------------------------------------------
    def _analyze_function_node(self, node: fx.Node) -> None:
        name = _op_name(node.target)
        record = NodeRecord(
            node_name=node.name, op="call_function", kind=name,
            in_shape=self._shape_of(self._tensor_args(node)[0]) if self._tensor_args(node) else None,
            out_shape=self._shape(node),
        )
        self._records[node.name] = record
        args = self._tensor_args(node)

        if node.target is operator.getitem or name == "getitem":
            self._barrier(node, args, "下标/切片会改变索引空间，无法对位传播")
            return
        self._classify_call(node, name, args, record)

    def _analyze_method_node(self, node: fx.Node) -> None:
        name = _op_name(node.target)
        record = NodeRecord(
            node_name=node.name, op="call_method", kind=name,
            in_shape=self._shape_of(self._tensor_args(node)[0]) if self._tensor_args(node) else None,
            out_shape=self._shape(node),
        )
        self._records[node.name] = record
        self._classify_call(node, name, self._tensor_args(node), record)

    def _classify_call(
        self, node: fx.Node, name: str, args: Sequence[fx.Node], record: NodeRecord
    ) -> None:
        """把 call_function / call_method 节点归类成直通、逐元素、归约或屏障。"""
        if name in REDUCTION_OPS:
            if not self._reduction_is_channel_preserving(node, args):
                self._barrier(node, args, "%s 归约涉及通道维，通道空间被破坏" % name)
                return
            self._inherit_slot(node, args, record)
            return

        if name in ELEMENTWISE_BINARY_OPS:
            if not self._inherit_slot(node, args, record):
                return
            # 逐元素运算要求两路通道数一致；不一致通常是广播（如 SE 的 1 通道门控）
            base_count = self._slots[record.out_slot].count if record.out_slot else None
            for arg in args[1:]:
                slot = self._out_slot_of_node.get(arg.name)
                if slot is None:
                    continue
                if base_count is not None and self._slots[slot].count != base_count:
                    self._barrier(
                        node, args,
                        "逐元素 %s 的两路通道数不同（%d vs %d），属于广播语义"
                        % (name, base_count, self._slots[slot].count),
                    )
                    return
                self._union(slot, record.out_slot)  # type: ignore[arg-type]
            return

        if name in PASS_THROUGH_OPS:
            self._inherit_slot(node, args, record)
            return

        if name in BARRIER_OPS:
            self._barrier(node, args, "%s 会改变通道索引空间" % name)
            return

        # 未知算子：保守处理
        self._barrier(node, args, "未识别的算子 %s，为避免破坏模型按屏障处理" % name)

    def _reduction_is_channel_preserving(
        self, node: fx.Node, args: Sequence[fx.Node]
    ) -> bool:
        """判断 ``mean``/``sum`` 之类的归约是否避开了通道维且保留了维度。"""
        if not args:
            return False
        slot_key = self._out_slot_of_node.get(args[0].name)
        if slot_key is None:
            return False
        axis = self._slots[slot_key].axis

        dims: Any = None
        keepdim = False
        if len(node.args) >= 2:
            dims = node.args[1]
        if len(node.args) >= 3:
            keepdim = bool(node.args[2])
        dims = node.kwargs.get("dim", dims)
        keepdim = bool(node.kwargs.get("keepdim", keepdim))

        if dims is None:
            return False  # 全量归约，通道维被抹掉
        if isinstance(dims, int):
            reduce_dims = (dims,)
        elif isinstance(dims, (list, tuple)):
            reduce_dims = tuple(int(d) for d in dims)
        else:
            return False
        if not keepdim:
            return False
        out_shape = self._shape(node)
        if out_shape is None:
            return False
        ndim = len(out_shape)
        normalized = {d if d >= 0 else ndim + d for d in reduce_dims}
        return axis not in normalized

    def _barrier(self, node: fx.Node, args: Sequence[fx.Node], reason: str) -> None:
        self._record_barrier(node, reason)
        self._barrier_upstream(node, reason)

    # ---- FLOPs -----------------------------------------------------------
    def _module_flops(self, node: fx.Node, module: nn.Module, record: NodeRecord) -> int:
        """乘加密集型算子的 FLOPs（1 MAC 记 2 FLOPs）。"""
        in_shape = self._shape_of(self._tensor_args(node)[0]) if self._tensor_args(node) else None
        out_shape = record.out_shape
        if in_shape is None or out_shape is None:
            return 0
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            kernel = 1
            for size in module.kernel_size:
                kernel *= int(size)
            groups = int(module.groups)
            spatial = 1
            for dim in out_shape[2:]:
                spatial *= int(dim)
            return 2 * int(module.out_channels) * (int(module.in_channels) // groups) * kernel * spatial
        if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            kernel = 1
            for size in module.kernel_size:
                kernel *= int(size)
            groups = int(module.groups)
            spatial = 1
            for dim in out_shape[2:]:
                spatial *= int(dim)
            return 2 * int(module.in_channels) * (int(module.out_channels) // groups) * kernel * spatial
        if isinstance(module, nn.Linear):
            return 2 * int(module.in_features) * int(module.out_features)
        return 0

    # ------------------------------------------------------------------ 建组
    def _build_groups(self) -> None:
        edits_by_root: "OrderedDict[str, List[GroupEdit]]" = OrderedDict()
        for record in self._records.values():
            if not record.needs_resize:
                continue
            in_root = self._find(record.in_slot) if record.in_slot else None
            out_root = self._find(record.out_slot) if record.out_slot else None
            if in_root is None and out_root is None:
                continue
            if in_root is not None and in_root == out_root:
                edits_by_root.setdefault(in_root, []).append(GroupEdit(record.module_name, "both"))
            else:
                if out_root is not None:
                    edits_by_root.setdefault(out_root, []).append(GroupEdit(record.module_name, "out"))
                if in_root is not None:
                    edits_by_root.setdefault(in_root, []).append(GroupEdit(record.module_name, "in"))

        barrier_reasons: Dict[str, str] = {}
        for key, reason in self._barrier_slots:
            root = self._find(key)
            barrier_reasons.setdefault(root, reason)
        protected_roots = {self._find(key) for key in self._protected_slots if key in self._slots}
        external_roots = {
            self._find(key) for key in self._external_in_slots if key in self._slots
        }

        self._groups: List[ChannelGroup] = []
        self._root_to_group: Dict[str, ChannelGroup] = {}
        for index, (root, edits) in enumerate(edits_by_root.items()):
            slots = [key for key in self._slots if self._find(key) == root]
            count = self._slots[slots[0]].count
            group = ChannelGroup(index=index, count=count, slots=slots, edits=edits)
            resizes_input = any(edit.kind in ("in", "both") for edit in edits)
            if count < 2:
                group.prunable, group.reason = False, "通道数小于 2，无法裁剪"
            elif not group.output_edits:
                group.prunable = False
                group.reason = (
                    "只包含输入侧通道（上游是模型输入或屏障节点），没有可评分的生产者"
                )
            elif root in barrier_reasons:
                group.prunable, group.reason = False, barrier_reasons[root]
            elif root in external_roots and resizes_input:
                group.prunable = False
                group.reason = (
                    "该组的输入张量来自模型输入或屏障（宽度无法改变），而组内有深度卷积/"
                    "归一化等「输入输出同索引空间」的结构，裁剪会破坏前向，整组跳过"
                )
            elif root in protected_roots:
                group.prunable, group.reason = False, "直接产生模型输出的通道组，默认保护"
            elif self.ignore and set(group.module_names) & self.ignore:
                excluded = sorted(set(group.module_names) & self.ignore)
                group.prunable, group.reason = False, "被 ignore 排除：%s" % ", ".join(excluded)
            else:
                group.prunable, group.reason = True, ""
            self._groups.append(group)
            self._root_to_group[root] = group

    # ------------------------------------------------------------------ 查询
    @property
    def groups(self) -> List[ChannelGroup]:
        return list(self._groups)

    def prunable_groups(self) -> List[ChannelGroup]:
        return [group for group in self._groups if group.prunable]

    def group_of_module(self, module_name: str) -> Optional[ChannelGroup]:
        """返回以该模块输出通道为主体的通道组。"""
        for group in self._groups:
            if module_name in group.output_modules:
                return group
        return None

    def group_by_index(self, index: int) -> ChannelGroup:
        for group in self._groups:
            if group.index == index:
                return group
        raise KeyError("不存在下标为 %d 的通道组" % index)

    def records(self) -> List[NodeRecord]:
        return list(self._records.values())

    def flops(self, per_module: bool = False):
        """统计乘加密集型算子的 FLOPs。"""
        breakdown = OrderedDict(
            (record.module_name or record.node_name, record.flops)
            for record in self._records.values()
            if record.flops
        )
        total = int(sum(breakdown.values()))
        return (total, breakdown) if per_module else total

    def parameters(self, only_conv_linear: bool = False) -> int:
        total = 0
        for record in self._records.values():
            if not record.module_name:
                continue
            module = self.model.get_submodule(record.module_name)
            if only_conv_linear and not isinstance(
                module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
            ):
                continue
            total += sum(param.numel() for param in module.parameters(recurse=False))
        return total

    # ------------------------------------------------------------------ 报告
    def summary(self) -> report_mod.Report:
        """生成依赖图报告。"""
        report = report_mod.Report("通道依赖图分析", model_name=type(self.model).__name__)
        report.add_kv(
            "总览",
            {
                "通道组总数": len(self._groups),
                "可裁剪通道组": len(self.prunable_groups()),
                "屏障节点数": len(self.barriers),
                "乘加 FLOPs（示例输入）": self.flops(),
                "参与统计的参数量": self.parameters(),
            },
        )
        report.add_table(
            "通道组",
            ["组", "通道数", "可裁剪", "涉及模块", "说明"],
            [
                [
                    group.index,
                    group.count,
                    "是" if group.prunable else "否",
                    group.display,
                    group.reason or "-",
                ]
                for group in self._groups
            ],
            note="不可裁剪的通道组会在结构化剪枝时自动跳过。",
        )
        if self.barriers:
            seen: Set[Tuple[str, str]] = set()
            rows = []
            for barrier in self.barriers:
                key = (barrier["node"], barrier["reason"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append([barrier["node"], barrier["op"], barrier["reason"]])
            report.add_table("屏障节点", ["节点", "算子", "原因"], rows)
        report.add_table(
            "逐模块统计",
            ["模块", "类型", "输入形状", "输出形状", "FLOPs"],
            [
                [
                    record.module_name,
                    record.kind,
                    record.in_shape or "-",
                    record.out_shape or "-",
                    record.flops or "-",
                ]
                for record in self._records.values()
                if record.module_name
            ],
        )
        return report

    def describe(self) -> str:
        return self.summary().to_text()

    def __repr__(self) -> str:  # pragma: no cover
        return "DependencyGraph(groups=%d, prunable=%d, barriers=%d)" % (
            len(self._groups),
            len(self.prunable_groups()),
            len(self.barriers),
        )


def build_graph(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    protect_output: bool = True,
    ignore: Iterable[str] = (),
) -> DependencyGraph:
    """便捷函数：``DependencyGraph(model, example_inputs)`` 的等价写法。"""
    return DependencyGraph(
        model, example_inputs, protect_output=protect_output, ignore=ignore
    )
