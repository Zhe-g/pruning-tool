# pruning-toolkit —— PyTorch 模型剪枝工具

一套**能跑、能验证、说真话**的模型剪枝工具：通道依赖分析、结构化与非结构化剪枝、
敏感度分析、迭代剪枝与微调、评估与报告导出。

设计上只依赖 **PyTorch + NumPy**；
示例模型与数据集全部内置且离线可用，克隆下来就能跑。

---

## 1. 它能做什么

| 能力 | 说明 |
| --- | --- |
| **通道依赖分析** | 用 `torch.fx` 追踪模型，把「共享同一通道索引空间」的张量位置合并成**通道组**，据此判断哪些层能安全剪、哪些必须一起剪（残差、深度卷积、BN 耦合） |
| **非结构化剪枝** | 逐权重置零，全局/逐层/自定义逐层比例，保留个数**精确**；配套掩码管理与梯度屏蔽，微调时被剪权重不会被重新激活 |
| **结构化剪枝** | 物理删除通道/神经元，参数与 FLOPs 真实下降；剪枝后**自动跑前向校验**（形状不变 + 数值有限），不通过直接报错 |
| **重要性准则** | `l1` / `l2` / `random` / `taylor`（一阶泰勒，需校准梯度）/ `bn_scale`（Network Slimming）/ `fpgm`（几何中位数）/ `none` |
| **敏感度分析与分配** | 逐层扫描剪枝强度响应；按敏感度自动分配逐层稀疏度，并用二分法保证**参数量加权平均恰好命中目标** |
| **迭代剪枝 + 微调** | 多轮「剪枝 → 微调」，稀疏度调度（constant/linear/cubic/exponential），每轮自动重新压回掩码、重新构建依赖图 |
| **评估** | 参数量、**非零参数量**、稀疏度、乘加 FLOPs、实测延迟（p50/p90/吞吐）、精度 |
| **报告与导出** | 每份结果同时产出 JSON 与 Markdown 报告；检查点（含掩码与形状元信息）、TorchScript、ONNX（需装 onnx）|
| **命令行** | `graph` / `inspect` / `sensitivity` / `prune` / `demo` / `template` 六个子命令 |

---

## 2. 快速开始

### 2.1 环境

```powershell
# 本机实测环境（RTX 4060 Laptop，CUDA 可用）
C:\Users\GuoZhe\.conda\envs\gpupy312\python.exe   # Python 3.12.13 + torch 2.11.0+cu126 + numpy 2.4.3
```

只依赖 `torch`，无需安装本包，直接在项目根目录运行即可：

```powershell
cd E:\ItemCodeFlies\pruning
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'   # Windows 控制台中文输出
& C:\Users\GuoZhe\.conda\envs\gpupy312\python.exe -m pruning_toolkit demo
```

### 2.2 一条命令看效果

```powershell
# 结构化剪枝小 VGG（训练 → 3 轮剪枝 → 每轮微调 → 评估 → 导出报告）
& $py -m pruning_toolkit demo --preset cnn

# 非结构化剪枝 MLP（90% 稀疏度）
& $py -m pruning_toolkit demo --preset mlp

# 只看通道依赖分析：哪些层能剪、哪些被屏障挡住
& $py -m pruning_toolkit graph --model resnet_small --dataset tiny_images
```

### 2.3 最小 API（接到你自己的模型上）

```python
import torch
from pruning_toolkit import DependencyGraph, StructuredPruner

model = MyNet().eval()
example_inputs = (torch.zeros(2, 1, 28, 28),)      # 形状要对，内容不重要

# 1) 先看清能不能剪
graph = DependencyGraph(model, example_inputs)
print(graph.summary().to_text())

# 2) 结构化剪枝：每个通道组裁掉 50%
pruner = StructuredPruner(model, example_inputs, criterion="l1", scope="local")
report = pruner.prune(0.5)                          # 内置前向校验
print("参数 %d → %d，FLOPs 下降 %.2f%%" % (
    report.params_before, report.params_after, 100 * report.flops_reduction))
```

非结构化剪枝更短：

```python
from pruning_toolkit import UnstructuredPruner

pruner = UnstructuredPruner(model, criterion="l1", scope="global")
report = pruner.prune(0.9)                          # 90% 权重置零
masks = pruner.masks                                # 微调时用
# 每个 optimizer step 之后调用一次，保证被剪权重不被重新激活：
masks.apply(model)
```

---

## 3. 实测结果

以下全部是本机实跑的数字（Python 3.12.13 / torch 2.11.0+cu126 / CPU 推理），
不是估算，也不是抄来的论文结论。

### 3.1 结构化剪枝：小 VGG，每组裁 50% 通道

`examples/02_structured_cnn.py`（`small_vgg` + 合成图像，batch=64，8 epoch 训练）

| 阶段 | test_acc | 参数量 | FLOPs | 延迟 p50 |
| --- | --- | --- | --- | --- |
| 基线 | 1.0000 | 61,338 | 38,634,752 | 1.46 ms |
| 剪枝后（未微调） | 0.0933 | 15,698（**-74.4%**） | 9,880,192（**-74.4%**） | 1.11 ms |
| 微调 6 epoch 后 | **1.0000** | 15,698 | 9,880,192 | — |

> 参数量、FLOPs、精度都是确定性结果，重跑可复现；**延迟会随运行环境大幅波动**。
> 同一台机器上同一个模型，实测 p50 在 **1.1 ~ 2.0 ms** 之间跳（取决于 CPU 频率、
> 后台负载等），因此这里给的只是某一次的读数，不要把它当成精确指标。
> 可靠的部分是**方向**：压缩 74% 参数与 FLOPs，并没有带来等比例的延迟下降。
> 比较延迟请多次测量、看 p50，并在同一负载条件下对比。

**怎么读**：50% 的通道比例带来 74% 的参数量下降，因为剪掉某层输出通道会同时削掉下游层的
输入通道，参数是二次下降的。剪枝瞬间精度从 1.00 崩到 0.09，微调 6 个 epoch 后完全恢复。

**延迟只从 1.46 ms 降到 1.11 ms**：小模型在 CPU 上的耗时由算子启动开销与内存带宽主导，
参数量下降不会等比例变成延迟下降。想让压缩真正变成加速，需要模型足够大、算子足够规整，
并配合推理引擎/量化。这是很多剪枝教程不会告诉你的事。

### 3.2 不同结构下的压缩效果

用 zoo 里 5 个示例模型做结构化剪枝（每组裁 50% 通道），
**这张表可以用 `tools/verify_models.py` 一条命令复现**。
权重未训练（随机初始化），所以只看**结构与计算量**的变化；前向校验验证的是
「剪枝后模型仍能正确前向、输出形状不变、数值有限」。

| 模型 | 通道组 | 可剪组 | 屏障 | 参数量 | FLOPs 下降 |
| --- | --- | --- | --- | --- | --- |
| `mlp` | 4 | 2 | 0 | 5,002 → 1,482（-70.4%） | -71.1% |
| `small_vgg`（链式 CNN） | 6 | 4 | 0 | 61,338 → 15,698（-74.4%） | -74.4% |
| `resnet_small`（残差） | 8 | 6 | 0 | 170,410 → 43,226（-74.6%） | -74.7% |
| `depthwise_cnn`（深度可分离） | 6 | 4 | 0 | 8,741 → 2,725（-68.8%） | -70.1% |
| `inception_like`（含 `cat`） | 7 | 2 | **1** | 2,682 → 1,362（-49.2%） | -50.0% |

换成别的准则/范围（同一个 `small_vgg`）：

| 配置 | 参数量下降 | FLOPs 下降 |
| --- | --- | --- |
| `l1` + `scope="local"`，目标 0.5 | -74.41% | -74.43% |
| `l1` + `scope="global"`，目标 0.6 | **-83.44%** | **-83.63%** |
| `bn_scale` + `scope="local"`，目标 0.4 | -64.01% | -63.67% |

为什么 `inception_like` 只压得动 49%？因为它有 1 个屏障（`torch.cat`）导致 2 个通道组被跳过。
每个模型还会有 1~2 个组因「输入侧通道不可改」或「输出层默认保护」被跳过；
`depthwise_cnn` 另外有 1 个组被跳过——它的首层深度卷积直接作用在 3 通道输入上，
剪它会把输入通道改掉，工具识别出这一点后整组跳过（详见 `tools/verify_models.py` 输出的原因表）。
这些跳过不是失败，而是**保证剪枝后前向一定正确**的代价。

### 3.3 非结构化剪枝：MLP 全局剪掉 90%

`examples/01_unstructured_mlp.py`（`blobs` 数据集，30 epoch 基线训练）

| 阶段 | test_acc | 稠密参数量 | 非零参数量 |
| --- | --- | --- | --- |
| 基线 | 0.9833 | 4,611 | 4,611 |
| 剪枝 90% 后 | 0.9667 | 4,611 | **579** |
| 掩码感知微调 20 epoch 后 | 0.9867 | 4,611 | 579 |

剪枝后工具会**硬校验**：所有掩码为 0 的位置，微调后权重必须仍然为 0（实测残留 0 个）。

注意两个口径：剪枝只作用于权重，偏置不参与，所以「可剪参数稀疏度」是 0.90，
而「全模型参数稀疏度」是 0.8744。报告里两个数都会给出。

### 3.4 迭代剪枝 vs 一次性剪枝（同等微调预算）

`examples/03_iterative_vs_oneshot.py`（`two_moons` + MLP，目标 95% 稀疏度，
两种策略都微调 20 个 epoch，避免"谁微调得多谁赢"的假结论）

| 方案 | test_acc | 非零参数 | 剪枝瞬间的 acc |
| --- | --- | --- | --- |
| 基线（未剪枝） | 0.9667 | 1,218 | — |
| 一次性剪到 95% + 微调 20 轮 | 0.9733 | 124 | 0.7467 |
| 分 4 轮迭代剪到 95% + 微调 20 轮 | **0.9800** | 124 | 最低 0.9600 |

**诚实的结论**：在这个小模型 + 简单任务上，两种策略最终只差 **+0.0067**，差距很小；
但"剪枝瞬间的损伤"差别很大（0.75 vs 0.96），迭代策略的过程更平滑。
差距会随模型规模、稀疏度、任务难度放大——**而这类结论必须在自己的模型上实测**。
本工具的价值正是让这个对照实验几行代码就能跑起来，而不是让你凭经验假设。

### 3.5 敏感度分析与逐层分配：一个"反面教材"

`examples/04_sensitivity_allocation.py` 对比了三种分配方案（目标 0.60）：

| 分配方案 | 剪枝后 acc | 微调后 acc | 通道裁剪 | 参数量 |
| --- | --- | --- | --- | --- |
| 均匀分配 | 0.0933 | **1.0000** | 0.5966 | 10,402 |
| 重建误差代理分配 | 0.1100 | 0.8000 | 0.6477 | 8,213 |
| 真实精度扫描分配 | 0.0933 | **1.0000** | 0.5966 | 10,402 |

这里有三个必须说清楚的点：

1. **代理指标会误导**。权重重建误差只看权重本身，不看这一层在任务里的作用，
   实测中它的排序与真实精度排序**不一致**，按它分配反而更差（0.80 vs 1.00）。
2. **这不是干净的对照实验**。三种方案最终参数量并不相等（10402 vs 8213），
   因为通道裁剪在层间级联。要干净比较，应固定最终参数量或 FLOPs，再各自调目标去逼近。
3. **没有信号时保守退化是正确的**。真实精度扫描在这个简单任务上无法区分通道组
   （各探测强度下精度都接近 1.0），工具于是退化为均匀分配，而不是强行乱分配。

工具同时提供两种扫描（`proxy="reconstruction"` 快、`proxy="accuracy"` 准），
就是为了让你在"快"和"准"之间自己选，并把风险摆在明面上。

---

## 4. 核心概念

### 4.1 通道依赖图（`pruning_toolkit.graph`）

结构化剪枝必须先回答：**删掉某层的某个通道，还有哪些地方的维度必须跟着改？**
工具用 `torch.fx` 追踪模型，把「共享同一通道索引空间」的张量位置（*槽位*）用并查集
合并成**通道组**，再判定每个组能否安全裁剪。

传播规则：

| 结构 | 规则 |
| --- | --- |
| 普通卷积 / 线性层 | 输入槽位与输出槽位是**不同**索引空间，各成一组 |
| 归一化 / 激活 / 池化 | 逐通道独立，输入输出合并为同一组 |
| 深度卷积 | 输入通道 c 只对应输出通道 c，合并为一组并同步更新 `groups` |
| 残差相加 `add` | 两路通道数相同则合并（残差块因此要求输入输出联合裁剪） |
| 逐元素乘 `mul` | 同上（如 SE 门控、注意力权重） |
| 拼接 `cat` | 各分支索引空间无法对应，作为**屏障**并整体保护 |
| 形状变换 `view/permute/getitem…` | 作为屏障；被屏障波及的组一律不剪 |

一个真实的残差网络分组结果（`resnet_small`，示例输入 4×3×32×32，共 8 组 / 6 组可剪）：

```
组0 通道=32 可剪   stem.0+stem.1+stages.0.conv1+stages.0.conv2+stages.0.bn2
                   +stages.1.conv1+stages.1.conv2+stages.1.bn2+stages.2.downsample.0+stages.2.conv1
组1 通道=3  跳过   stem.0                       ← 只包含输入侧通道
组2 通道=32 可剪   stages.0.conv1+stages.0.bn1+stages.0.conv2
组3 通道=32 可剪   stages.1.conv1+stages.1.bn1+stages.1.conv2
组4 通道=64 可剪   stages.2.downsample.0+stages.2.downsample.1+stages.2.conv2+stages.2.bn2
                   +stages.3.conv1+stages.3.conv2+stages.3.bn2+classifier
组5 通道=64 可剪   stages.2.conv1+stages.2.bn1+stages.2.conv2
组6 通道=64 可剪   stages.3.conv1+stages.3.bn1+stages.3.conv2
组7 通道=10 跳过   classifier                   ← 直接产生模型输出的组，默认保护
```

组 0 之所以把 stem 和多个 conv 合并在一起，是因为残差短接把「块输入通道空间」和
「块输出通道空间」连成了同一个索引空间——**这些层必须一起剪**，否则前向就会维度不匹配。
组 4 里出现 `classifier` 也是正确的：分类头被剪的是**输入维度**（64），
而它的输出维度（10 个类别）在组 7 里被保护起来。

直接产生模型输出的通道组默认受保护，可以用 `protect_output=False` 关闭。
但关闭后若真去裁剪类别维，模型输出会从 `(N, num_classes)` 变成 `(N, keep)`，
`prune()` 的前向校验会因此报错——这是有意的保护，不是 bug；
确实想剪类别维时请显式传 `prune(..., verify=False)`（实测：可剪组从 4 个变成 5 个，
输出形状变为 `(2, 5)`）。

### 4.2 屏障（barrier）：安全阀

无法证明安全时**宁可不剪**，并在报告里写明原因，而不是产出一个前向会崩的模型。
被屏障挡住的组会出现在报告的「跳过的通道组」表里，原因可读，例如：

> 上游存在屏障节点：cat 会改变通道索引空间

### 4.3 掩码 vs 形状

| | 非结构化剪枝 | 结构化剪枝 |
| --- | --- | --- |
| 权重张量形状 | 不变 | **真的变小** |
| 稠密 FLOPs | 不变 | 下降 |
| 参数量 | 稠密口径不变，有效参数量下降 | 真的下降 |
| 需要掩码 | 需要（微调时保持稀疏） | 不需要（结构已经没了） |
| 可回滚 | 可以（`pruner.restore()`） | 不可以（破坏性，先 `deepcopy`） |
| 前向校验 | 不适用 | **自动执行** |

---

## 5. 模块与 API

```
pruning_toolkit/
├── report.py          报告：table/kv/text/markdown → JSON + Markdown
├── schedules.py       稀疏度调度：cumulative_targets / incremental_ratios / Scheduler
├── modules.py         模块通道语义与参数裁剪原语（conv/linear/norm/深度卷积）
├── graph.py           通道依赖图：DependencyGraph / ChannelGroup / 屏障判定
├── criteria.py        重要性准则 + 校准梯度收集 collect_grads
├── masks.py           掩码管理：MaskBook（apply / 梯度钩子 / 存取）
├── unstructured.py    非结构化剪枝：UnstructuredPruner
├── structured.py      结构化剪枝：StructuredPruner
├── metrics.py         参数量 / 非零参数 / FLOPs / 延迟 / 精度评估
├── analysis.py        敏感度扫描与稀疏度分配
├── iterative.py       迭代剪枝 + 微调：IterativePruner / RoundContext
├── export.py          检查点 / TorchScript / ONNX / 报告批量保存
├── config.py          配置（JSON / YAML）+ 预设 for_demo / for_mlp_demo
├── pipeline.py        端到端流水线：run() / prepare() / PipelineResult
├── cli.py             命令行
└── zoo/               示例模型、离线数据集、最小训练器
```

### 5.1 常用入口

```python
from pruning_toolkit import (
    # 分析与剪枝
    DependencyGraph, StructuredPruner, UnstructuredPruner,
    # 准则与掩码
    CRITERIA, collect_grads, MaskBook,
    # 敏感度与迭代
    structured_sensitivity, unstructured_sensitivity, recommend_sparsity, IterativePruner,
    # 指标与报告
    count_parameters, count_nonzero_parameters, model_sparsity, count_flops,
    benchmark_latency, summarize,
    # 导出与流水线
    save_checkpoint, export_torchscript, save_reports, PruneConfig, run,
)
```

`IterativePruner` 的微调回调签名是 `finetune_fn(model, context)`，
`context` 是 `RoundContext(round_index, target, achieved, masks)`：

```python
def finetune(model, ctx):
    hook = (lambda m: ctx.masks.apply(m)) if ctx.masks else None   # 非结构化时保持稀疏
    fit(model, train_loader, test_loader, epochs=2, post_step_hook=hook)
    return {"acc": evaluate(model, test_loader)["acc"]}

report = IterativePruner(
    model, example_inputs, mode="structured",
    rounds=3, final_sparsity=0.5, schedule="cubic",
    eval_fn=lambda m: evaluate(m, test_loader),
    finetune_fn=finetune,
).run()
print(report.to_report().to_text())   # 含「剪枝后 acc」与「微调后 acc」两列
```

### 5.2 稀疏度调度曲线的形状

四条曲线只描述「累积目标怎么爬升」，每轮真正要剪的比例由
`incremental_ratios()` 换算。以 `final=0.9, rounds=4` 为例，第一轮的目标是：

| 曲线 | 第一轮目标 | 说明 |
| --- | --- | --- |
| `constant` | 0 | 前面几轮不动，**最后一轮**才剪到 0.9（等价于在末尾做一次性剪枝） |
| `linear` | 0.225 | 线性爬升：0.225 → 0.45 → 0.675 → 0.9 |
| `cubic`（默认） | **0.520** | **前期快、后期缓**：0.520 → 0.788 → 0.886 → 0.9，对应 Zhu & Gupta (2017) 的经典 cubic 调度 |
| `exponential` | 0.056 | 起点为 0 时退化为平方曲线（否则按几何增长） |

⚠️ `cubic` 很容易被误解成「前期剪得少」——**恰恰相反**，它在第一轮就剪掉一半以上，
后面余量越来越小自然放缓。想验证这一点可以直接跑：

```python
from pruning_toolkit import cumulative_targets, incremental_ratios
print(cumulative_targets(0.9, 4, "cubic"))       # [0.52, 0.787, 0.886, 0.9]
print(incremental_ratios(cumulative_targets(0.9, 4, "cubic")))
# [0.5203, 0.557, 0.4632, 0.1233] —— 不是严格单调，但后期动作明显收窄
```

### 5.3 准则一览

| 准则 | 粒度 | 是否需要梯度 | 说明 |
| --- | --- | --- | --- |
| `l1` | 元素/通道 | 否 | 绝对值；最常用、最稳 |
| `l2` | 元素/通道 | 否 | 平方 / L2 范数 |
| `random` | 元素/通道 | 否 | 随机基线（同 seed 可复现）|
| `taylor` | 元素/通道 | **是** | 一阶泰勒 `|w·grad|`，用 `collect_grads` 采校准梯度 |
| `bn_scale` | 通道组 | 否 | 组内 BN 的 `|gamma|`（Network Slimming），无 BN 时退化为 `l2` |
| `fpgm` | 通道 | 否 | 到卷积核几何中位数的 L2 距离 |
| `none` | 元素 | 否 | 恒为 1，等价于按顺序裁剪 |

---

## 6. 命令行

```powershell
& $py -m pruning_toolkit <子命令> [选项]

demo         跑一遍完整演示（--preset cnn|mlp，--quick 快速模式）
prune        按配置或参数执行流水线（--config / --model / --mode / --rounds / --sparsity …）
inspect      模型体检：参数量、FLOPs、延迟、逐模块明细（--benchmark）
graph        只做通道依赖分析（--report 落盘 .md/.json）
sensitivity  只做敏感度扫描（--mode structured|unstructured --sparsities 0.1,...,0.9）
template     导出配置模板（JSON）
```

实例：

```powershell
# 依赖分析并落盘报告
& $py -m pruning_toolkit graph --model resnet_small --dataset tiny_images --report runs/graph.md

# 结构化剪枝：4 轮、目标 55%、用 bn_scale 准则、自动分配逐层目标
& $py -m pruning_toolkit prune --model small_vgg --mode structured --rounds 4 `
    --sparsity 0.55 --criterion bn_scale --allocate --out runs --tag vgg-bn

# 用配置文件跑（配置里带数据集参数、训练/微调设置、产物设置）
& $py -m pruning_toolkit prune --config configs/allocated_vgg_bn_scale.json

# 先看配置长什么样
& $py -m pruning_toolkit template configs/my.json
```

数据集参数可透传，例如 `--dataset-kwarg noise=1.1 --dataset-kwarg n_samples=2000`。

---

## 7. 示例脚本

| 脚本 | 内容 |
| --- | --- |
| `examples/01_unstructured_mlp.py` | 非结构化剪枝 90% + 掩码感知微调，对照稠密/非零参数量 |
| `examples/02_structured_cnn.py` | 结构化剪枝 50% 通道，通道依赖图 + 参数量/FLOPs/延迟对照 |
| `examples/03_iterative_vs_oneshot.py` | 迭代剪枝 vs 一次性剪枝（同等微调预算的实测对照）|
| `examples/04_sensitivity_allocation.py` | 敏感度扫描（代理指标 vs 真实精度）+ 逐层分配，含"代理指标会误导"的实测证据 |
| `examples/05_quickstart_own_model.py` | 接到自己模型上的最小路径 + 导出 + 注意事项 |
| `tools/verify_models.py` | 全模型自检：依赖分析 → 结构化/非结构化剪枝 → 一致性断言 + 可复现表格 |

全部脚本都可直接运行，产物写到 `runs/exampleN/`。
`tools/verify_models.py` 不需要训练与数据，几十秒跑完，适合作为改动后的冒烟测试。

---

## 8. 配置文件

JSON 为默认格式（YAML 需要 `pip install pyyaml`，未安装时会给出明确提示）。
未知字段会**报错**而不是被静默忽略，避免把 `criterion` 写成 `critirion` 却毫无察觉。

```json
{
  "model": "small_vgg",
  "dataset": "tiny_images",
  "dataset_kwargs": { "n_samples": 1500, "noise": 0.6 },
  "batch_size": 64,
  "device": "cpu",
  "seed": 0,
  "mode": "structured",
  "criterion": "l1",
  "scope": "local",
  "schedule": "cubic",
  "rounds": 3,
  "sparsity": 0.5,
  "allocate": false,
  "per_layer": {},
  "exclude": [],
  "protect_output": true,
  "benchmark": true,
  "train":    { "enabled": true, "epochs": 8, "lr": 0.01 },
  "finetune": { "enabled": true, "epochs_per_round": 2, "lr": 0.005 },
  "output":   { "directory": "runs", "tag": "demo", "export_torchscript": false }
}
```

内置三份可直接用：`configs/structured_vgg.json`、`configs/unstructured_mlp.json`、
`configs/allocated_vgg_bn_scale.json`。

> 注意 `sparsity` 的含义随 `mode` 变化：非结构化=目标稀疏度；结构化=通道裁剪比例。

---

## 9. 设计取舍与已知边界

这一节是刻意写详细的——工具的边界比它的功能更值得知道。

1. **模型必须能被 `torch.fx` 追踪**。`forward` 里若有依赖张量数值的 `if/while`、
   或 fx 不支持的 Python 结构，`DependencyGraph` 会抛 `UnsupportedGraphError` 并说明原因。
   这种情况下非结构化剪枝仍然可用（它不需要依赖图）。
2. **拼接（`cat`）不传播**。`torch.cat` 两侧来自不同索引空间时，无法对位裁剪下游输入通道，
   相关通道组会被保守跳过并在报告里写明。这是为保证剪枝后前向一定正确。
   实测：`inception_like` 的参数量只降 49.2%（2,682 → 1,362），而链式 `small_vgg`
   降 74.4%——差距主要来自这一个屏障。
3. **输入侧与深度卷积的组合会被整组跳过**。若某个「输入输出同索引空间」的模块
   （深度卷积、归一化）直接作用在模型输入上，剪它会改变输入宽度，工具会整组跳过
   （`depthwise_cnn` 的首层就是这种情况，见 §3.2 与 `tools/verify_models.py`）。
4. **分组卷积（`groups` 既不是 1 也不是深度卷积）不支持结构化剪枝**，会被当作屏障。
   深度卷积（`groups == in == out`）是支持的，会同步更新 `groups` 与 `in/out_channels`。
5. **`nn.MultiheadAttention` / `TransformerEncoderLayer` 等复合模块**会被 fx 视为叶子模块，
   因而不在支持列表里，结构化剪枝时作为屏障；非结构化剪枝不受影响。
   要剪 Transformer 的注意力头，需要把它们展开成显式的 `Linear`。
6. **非结构化剪枝不会加速稠密推理**。它省的是有效参数量与存储；
   想要 FLOPs 与延迟收益请用结构化剪枝。报告里会明确写出这一点。
7. **小模型的延迟数字不可靠**。CPU 上小模型单次前向只有几十微秒，抖动很大；
   延迟对比建议看 p50，并意识到算子启动开销会主导结果。
8. **结构化剪枝是破坏性的**。模型结构被就地改变，无法把剪枝后的 `state_dict`
   加载回原始结构的模型类；请在剪枝前 `copy.deepcopy` 保留基线，
   部署时用导出的 TorchScript 或按 `checkpoint` 里的形状元信息重建结构。
9. **敏感度的权重重建误差是代理指标**，实测会与真实精度排序不一致（见 §3.5）。
   需要可靠信号时用 `proxy="accuracy"` 配合 `eval_fn` 做任务感知扫描，代价是慢。
10. **逐层分配的目标是「参数量加权平均」**，而通道裁剪在层间级联，
    因此不同分配方案最终得到的参数量可能并不相等。要做干净的策略对比，
    应先固定最终参数量或 FLOPs，再各自调目标去逼近（见 §3.5 的说明）。
11. **ONNX 导出路径本机未实测**（环境缺 `onnx` 包），代码会在缺失时给出安装提示；
    TorchScript 导出已实测可用。
12. **示例数据集是离线合成的**（`blobs` / `two_moons` / `tiny_images`），
    目的是让工具本身可复现、无外部依赖；接真实数据集时把 `example_inputs` 换成
    一批真实形状的数据、并把评估回调换成你的 `DataLoader` 即可。

---

## 10. 目录结构

```
pruning/
├── README.md                    本文档
├── pyproject.toml / requirements.txt
├── pruning_toolkit/             工具包（见 §5）
├── examples/                    5 个可运行示例
├── configs/                     3 份可直接用的配置
├── tools/verify_models.py       全模型自检（依赖分析 + 剪枝 + 一致性断言）
├── tests/test_toolkit.py        单元测试（unittest，无第三方依赖）
└── runs/                        运行产物：报告、指标、检查点、掩码
```

自检（几十秒，不训练、不下载数据，适合改完代码后冒烟）：

```powershell
& C:\Users\GuoZhe\.conda\envs\gpupy312\python.exe tools\verify_models.py
```

运行测试（**没有 pytest，用的是标准库 unittest**）：

```powershell
cd E:\ItemCodeFlies\pruning
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
& C:\Users\GuoZhe\.conda\envs\gpupy312\python.exe -m unittest discover -s tests -v
```

当前 **81 个用例，实测约 1.8 秒全部通过**（覆盖报告/调度/模块语义/依赖图/准则/掩码/
两种剪枝/指标/敏感度/迭代/配置/导出/CLI/流水线/公开 API，含错误路径与边界）。

---

## 11. 常见问题

**Q：`DependencyGraph` 报 `UnsupportedGraphError` 怎么办？**
先用非结构化剪枝，或把 `forward` 改写成静态图形式（去掉依赖数值的分支）。
异常信息里会带上原始的 fx 报错原因。

**Q：为什么我的模型有些层没被剪？**
跑 `python -m pruning_toolkit graph --model ...` 看「跳过的通道组」和原因。
常见原因是位于 `cat` 之后、是分组卷积、或是输出层（默认保护，可用
`protect_output=False` 关闭）。

**Q：剪枝后精度掉了怎么办？**
这是正常的——先看报告里「剪枝后 acc」与「微调后 acc」两列，确认微调是否奏效。
若微调也救不回来，降低 `sparsity`、改用 `rounds>1` 的迭代剪枝，
或先用 `sensitivity` 看哪一层最敏感。

**Q：怎么微调才不会把剪掉的权重又激活？**
非结构化剪枝必须做两件事：`finetune_fn` 里给 `fit(..., post_step_hook=lambda m: ctx.masks.apply(m))`，
并且 `IterativePruner` 每轮结束会自动再压一次掩码。工具还会硬校验残留零的个数
（见 `examples/01`）。

**Q：参数量下降很多，为什么延迟没降？**
见 §3.1 与 §9 的第 6、7 条：稠密算子不会因为权重里有零而变快；
小模型受算子启动开销与内存带宽主导。

**Q：能剪 Transformer / LLM 吗？**
当前版本对复合模块（`MultiheadAttention` 等）只做屏障处理。剪注意力头需要把
`q_proj/k_proj/v_proj/out_proj` 展开成显式 `Linear` 后再用本工具的通道组机制——
这属于后续工作，工具当前不承诺支持。
