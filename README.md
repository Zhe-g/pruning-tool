# pruning-toolkit

PyTorch 模型剪枝工具。先用依赖图判断「哪些层能安全剪」，再执行结构化（删通道）或
非结构化（权重置零）剪枝，并配套敏感度分析、逐层稀疏度分配、迭代剪枝与微调、
评估与报告导出。

- 核心引擎只依赖 **PyTorch**，不依赖任何第三方剪枝库；
- 内置示例模型与离线合成数据集，克隆后即可运行，不需要下载数据；
- 每次结构化剪枝都会自动做前向校验（形状不变 + 数值有限）；
  无法证明安全的结构会被跳过并在报告里写明原因——**正确性优先于压缩率**；
- 所有评估结果统一输出 JSON 与 Markdown 两种格式。

---

## 目录

- [1. 功能特性](#1-功能特性)
- [2. 环境要求与安装](#2-环境要求与安装)
- [3. 快速开始](#3-快速开始)
- [4. 核心概念](#4-核心概念)
- [5. 参考数据](#5-参考数据)
- [6. 命令行](#6-命令行)
- [7. 示例与自检](#7-示例与自检)
- [8. 配置文件](#8-配置文件)
- [9. 用于检测模型（ultralytics / YOLO）](#9-用于检测模型ultralytics--yolo)
- [10. 模块结构](#10-模块结构)
- [11. 已知限制](#11-已知限制)
- [12. 常见问题](#12-常见问题)
- [13. 贡献](#13-贡献)
- [14. 许可证](#14-许可证)

---

## 1. 功能特性

| 能力 | 说明 |
| --- | --- |
| **通道依赖分析** | 用 `torch.fx` 追踪模型，把「共享同一通道索引空间」的张量位置合并成通道组，据此判断哪些层可独立剪、哪些必须联合剪（残差、深度卷积、BN 耦合） |
| **非结构化剪枝** | 逐权重置零，支持全局 / 逐层 / 自定义逐层比例，保留个数精确；配套掩码管理与梯度屏蔽，微调时被剪权重不会被重新激活 |
| **结构化剪枝** | 物理删除通道或神经元，参数量与 FLOPs 真实下降；剪枝后自动执行前向校验 |
| **重要性准则** | `l1` / `l2` / `random` / `taylor`（一阶泰勒，需校准梯度）/ `bn_scale`（Network Slimming）/ `fpgm`（几何中位数）/ `none` |
| **敏感度分析与分配** | 逐层扫描剪枝强度响应，按敏感度自动分配逐层稀疏度，并用二分法保证参数量加权平均命中目标 |
| **迭代剪枝 + 微调** | 多轮「剪枝 → 微调」，支持 constant / linear / cubic / exponential 四种稀疏度调度；每轮自动重新压回掩码、重新构建依赖图 |
| **评估** | 参数量、非零参数量、稀疏度、乘加 FLOPs、实测延迟（p50 / p90 / 吞吐）、精度或自定义指标 |
| **报告与导出** | JSON + Markdown 报告、检查点（含掩码与形状元信息）、TorchScript、ONNX（需另装 `onnx`） |
| **命令行** | `demo` / `prune` / `inspect` / `graph` / `sensitivity` / `template` 六个子命令 |

---

## 2. 环境要求与安装

### 2.1 依赖

| 项 | 要求 |
| --- | --- |
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0（CPU 可用，CUDA 可选） |
| 可选 | `pyyaml`（读写 YAML 配置）、`onnx`（导出 ONNX） |

> 本仓库在 Python 3.12 + PyTorch 2.11 上完成验证；表中的版本下限为声明值，未逐一测试。

### 2.2 安装

方式一：**不安装，直接在仓库根目录运行**（推荐，最简单）

```bash
git clone <仓库地址> && cd pruning
python -m pruning_toolkit demo
```

方式二：**开发模式安装**（会注册 `prune-toolkit` 命令，便于在其他目录使用）

```bash
pip install -e .
prune-toolkit demo
```

两种方式都要求当前 Python 环境已安装 PyTorch。

### 2.3 中文输出（可选）

命令行与报告包含中文。若终端出现乱码，按平台设置 UTF-8：

```powershell
# Windows PowerShell
$env:PYTHONIOENCODING = "utf-8"
```

```bash
# Linux / macOS
export PYTHONIOENCODING=utf-8
```

---

## 3. 快速开始

### 3.1 命令行

```bash
# 结构化剪枝示例：训练 → 3 轮剪枝 → 每轮微调 → 评估 → 导出报告
python -m pruning_toolkit demo --preset cnn

# 非结构化剪枝示例：全局置零 90%
python -m pruning_toolkit demo --preset mlp

# 只看通道依赖分析：哪些层可剪、哪些被屏障挡住
python -m pruning_toolkit graph --model resnet_small --dataset tiny_images
```

产物默认写到 `runs/`。

### 3.2 Python API

结构化剪枝（删除通道）：

```python
import torch
from pruning_toolkit import DependencyGraph, StructuredPruner

model = MyNet().eval()
example_inputs = (torch.zeros(2, 1, 28, 28),)      # 形状需与真实推理一致

# 1) 先确认结构可被分析
graph = DependencyGraph(model, example_inputs)
print(graph.summary().to_text())

# 2) 每个通道组裁掉 50%，prune() 内置前向校验
pruner = StructuredPruner(model, example_inputs, criterion="l1", scope="local")
report = pruner.prune(0.5)
print("参数量 %d → %d，FLOPs 下降 %.2f%%" % (
    report.params_before, report.params_after, 100 * report.flops_reduction))
```

非结构化剪枝（权重置零）：

```python
from pruning_toolkit import UnstructuredPruner

pruner = UnstructuredPruner(model, criterion="l1", scope="global")
report = pruner.prune(0.9)          # 90% 权重置零
masks = pruner.masks                # 微调阶段使用

# 每个 optimizer step 之后调用，保证被剪权重不被重新激活
masks.apply(model)
```

### 3.3 验证安装

```bash
# 不需要训练与数据，几十秒完成
python tools/verify_models.py

# 单元测试
python -m unittest discover -s tests -v
```

---

## 4. 核心概念

### 4.1 通道依赖图

结构化剪枝需要先回答一个问题：**删掉某一层的某个通道后，还有哪些地方的维度必须跟着改？**

工具用 `torch.fx` 追踪模型，把共享同一通道索引空间的张量位置（槽位）用并查集合并为
通道组，再判定每个组是否可安全裁剪。传播规则如下：

| 结构 | 规则 |
| --- | --- |
| 普通卷积 / 线性层 | 输入槽位与输出槽位属于不同索引空间，各自成组 |
| 归一化 / 激活 / 池化 | 逐通道独立，输入输出合并为同一组 |
| 深度卷积 | 输入通道 c 仅对应输出通道 c，合并为一组并同步更新 `groups` |
| 残差相加 `add` | 两路通道数相同时合并（残差块因此要求输入输出联合裁剪） |
| 逐元素乘 `mul` | 同上（如 SE 门控、注意力权重） |
| 拼接 `cat` | 各分支索引空间无法对应，作为屏障并整体保护 |
| 形状变换 `view` / `permute` / `getitem` | 作为屏障；受屏障影响的组一律跳过 |

一个实际的分组结果（`resnet_small`，示例输入 4×3×32×32，共 8 组，其中 6 组可剪）：

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

组 0 之所以把 stem 与多个卷积合并在一起，是因为残差短接把「块输入通道空间」与
「块输出通道空间」连成了同一索引空间，这些层必须一起剪，否则前向时维度不匹配。
组 4 中出现 `classifier` 同样是正确的：它被裁剪的是输入维度（64），
输出维度（10 个类别）在组 7 中受保护。

直接产生模型输出的通道组默认受保护，可通过 `protect_output=False` 关闭。
关闭后若确实裁剪类别维，输出形状会从 `(N, num_classes)` 变为 `(N, keep)`，
`prune()` 的前向校验会因此报错；确实需要该行为时显式传入 `prune(..., verify=False)`。

### 4.2 屏障

无法证明安全的裁剪不做，并记录原因，而不是产生一个前向会报错的模型。
受屏障影响的组会出现在报告的「跳过的通道组」表中，例如：

> 上游存在屏障节点：cat 会改变通道索引空间

### 4.3 掩码与形状

| | 非结构化剪枝 | 结构化剪枝 |
| --- | --- | --- |
| 权重张量形状 | 不变 | 真正变小 |
| 稠密 FLOPs | 不变 | 下降 |
| 参数量 | 稠密口径不变，有效参数量下降 | 真的下降 |
| 是否需要掩码 | 需要（微调时保持稀疏） | 不需要（结构已删除） |
| 是否可回滚 | 可以（`pruner.restore()`） | 不可以（破坏性，建议先 `copy.deepcopy`） |
| 前向校验 | 不适用 | 自动执行 |

---

## 5. 参考数据

下列数据均由仓库内脚本直接产生，可在相同环境复现。
测试环境为 Python 3.12 / PyTorch 2.11 / RTX 4060 Laptop（推理分别统计 CPU 与 GPU），
**不同硬件与负载下延迟读数会有明显差异，请以量级而非精确值参考**。

### 5.1 结构化剪枝：小 VGG，每组裁 50% 通道

`examples/02_structured_cnn.py`（`small_vgg` + 合成图像，batch=64，训练 8 epoch）

| 阶段 | test_acc | 参数量 | FLOPs | 延迟 p50 |
| --- | --- | --- | --- | --- |
| 基线 | 1.0000 | 61,338 | 38,634,752 | 1.46 ms |
| 剪枝后（未微调） | 0.0933 | 15,698（**-74.4%**） | 9,880,192（**-74.4%**） | 1.11 ms |
| 微调 6 epoch 后 | **1.0000** | 15,698 | 9,880,192 | — |

参数量与 FLOPs 为确定性结果，重跑可复现；延迟读数在同一台机器上的波动区间约为
1.1 ~ 2.0 ms（受 CPU 频率与后台负载影响），仅作量级参考。

**解读**：50% 的通道比例带来 74% 的参数量下降，原因是剪掉某层输出通道会同时削减
下游层的输入通道，参数量呈二次下降。剪枝瞬间精度由 1.00 降至 0.09，微调 6 个 epoch
后完全恢复。

延迟仅从 1.46 ms 降至 1.11 ms。小模型在 CPU 上的耗时主要由算子启动开销与内存带宽决定，
参数量下降不会等比例转化为延迟下降；要让压缩真正变成加速，通常需要模型足够大、
算子足够规整，并配合推理引擎或量化。

### 5.2 不同结构下的压缩效果

对 zoo 中 5 个示例模型做结构化剪枝（每组裁 50% 通道），
可用 `python tools/verify_models.py` 一条命令复现。权重为随机初始化，
因此只反映结构与计算量的变化；前向校验验证的是「剪枝后模型仍能正确前向、
输出形状不变、数值有限」。

| 模型 | 通道组 | 可剪组 | 屏障 | 参数量 | FLOPs 下降 |
| --- | --- | --- | --- | --- | --- |
| `mlp` | 4 | 2 | 0 | 5,002 → 1,482（-70.4%） | -71.1% |
| `small_vgg`（链式 CNN） | 6 | 4 | 0 | 61,338 → 15,698（-74.4%） | -74.4% |
| `resnet_small`（残差） | 8 | 6 | 0 | 170,410 → 43,226（-74.6%） | -74.7% |
| `depthwise_cnn`（深度可分离） | 6 | 4 | 0 | 8,741 → 2,725（-68.8%） | -70.1% |
| `inception_like`（含 `cat`） | 7 | 2 | **1** | 2,682 → 1,362（-49.2%） | -50.0% |

同一模型（`small_vgg`）换用不同准则与裁剪范围：

| 配置 | 参数量下降 | FLOPs 下降 |
| --- | --- | --- |
| `l1` + `scope="local"`，目标 0.5 | -74.41% | -74.43% |
| `l1` + `scope="global"`，目标 0.6 | **-83.44%** | **-83.63%** |
| `bn_scale` + `scope="local"`，目标 0.4 | -64.01% | -63.67% |

`inception_like` 压缩率偏低，是因为其中 1 个屏障（`torch.cat`）导致 2 个通道组被跳过。
此外每个模型通常有 1~2 个组因「输入侧通道不可改」或「输出层默认保护」被跳过；
`depthwise_cnn` 另有 1 个组被跳过——其首层深度卷积直接作用在 3 通道输入上，
裁剪会改变输入宽度，工具识别后整组跳过（详见 `tools/verify_models.py` 输出的原因表）。
这些跳过是保证剪枝后前向正确的代价，而非执行失败。

### 5.3 非结构化剪枝：MLP 全局置零 90%

`examples/01_unstructured_mlp.py`（`blobs` 数据集，基线训练 30 epoch）

| 阶段 | test_acc | 稠密参数量 | 非零参数量 |
| --- | --- | --- | --- |
| 基线 | 0.9833 | 4,611 | 4,611 |
| 剪枝 90% 后 | 0.9667 | 4,611 | **579** |
| 掩码感知微调 20 epoch 后 | 0.9867 | 4,611 | 579 |

剪枝后工具会校验掩码：所有被剪位置在微调后权重仍须为 0，实测残留 0 个。

注意两个统计口径：剪枝只作用于权重、偏置不参与，因此「可剪参数稀疏度」为 0.90，
而「全模型参数稀疏度」为 0.8744。报告中两个数值都会给出。

### 5.4 迭代剪枝与一次性剪枝（同等微调预算）

`examples/03_iterative_vs_oneshot.py`（`two_moons` + MLP，目标 95% 稀疏度，
两种策略均微调 20 个 epoch，以排除「微调更多因而更好」的干扰）

| 方案 | test_acc | 非零参数 | 剪枝瞬间的 acc |
| --- | --- | --- | --- |
| 基线（未剪枝） | 0.9667 | 1,218 | — |
| 一次性剪到 95% + 微调 20 轮 | 0.9733 | 124 | 0.7467 |
| 分 4 轮迭代剪到 95% + 微调 20 轮 | **0.9800** | 124 | 最低 0.9600 |

在该小模型与简单任务上，两种策略最终仅相差 **+0.0067**，差距很小；
但「剪枝瞬间的损失」差别明显（0.75 与 0.96），迭代策略的过程更平滑。
该差距会随模型规模、稀疏度与任务难度变化，因此这类结论应在目标模型上实测，
而不是直接套用。工具的作用是让这类对照实验只需几行代码即可完成。

### 5.5 敏感度分配：一个值得注意的结果

`examples/04_sensitivity_allocation.py` 对比了三种逐层分配方案（目标 0.60）：

| 分配方案 | 剪枝后 acc | 微调后 acc | 通道裁剪 | 参数量 |
| --- | --- | --- | --- | --- |
| 均匀分配 | 0.0933 | **1.0000** | 0.5966 | 10,402 |
| 重建误差代理分配 | 0.1100 | 0.8000 | 0.6477 | 8,213 |
| 真实精度扫描分配 | 0.0933 | **1.0000** | 0.5966 | 10,402 |

有三点需要说明：

1. **代理指标可能误导**。权重重建误差只反映权重本身，不反映该层在任务中的作用；
   实测中其排序与真实精度排序不一致，按它分配反而更差（0.80 与 1.00）。
2. **这不是严格的对照实验**。三种方案最终参数量并不相等（10,402 与 8,213），
   因为通道裁剪在层间级联。若需严格比较，应先固定最终参数量或 FLOPs，再分别调目标逼近。
3. **无有效信号时退化为均匀分配是正确行为**。真实精度扫描在该任务上无法区分通道组
   （各探测强度下精度均接近 1.0），工具因此退化为均匀分配，而非强行分配。

工具同时提供两种扫描方式（`proxy="reconstruction"` 快、`proxy="accuracy"` 准），
由使用者按时间预算与可靠性要求自行选择。

---

## 6. 命令行

```bash
python -m pruning_toolkit <子命令> [选项]

demo         运行完整演示（--preset cnn|mlp，--quick 快速模式）
prune        按配置或参数执行流水线（--config / --model / --mode / --rounds / --sparsity …）
inspect      模型概况：参数量、FLOPs、延迟、逐模块明细（--benchmark）
graph        仅做通道依赖分析（--report 落盘 .md/.json）
sensitivity  仅做敏感度扫描（--mode structured|unstructured --sparsities 0.1,...,0.9）
template     导出配置模板（JSON）
```

示例：

```bash
# 依赖分析并落盘报告
python -m pruning_toolkit graph --model resnet_small --dataset tiny_images --report runs/graph.md

# 结构化剪枝：4 轮、目标 55%、bn_scale 准则、自动分配逐层目标
python -m pruning_toolkit prune --model small_vgg --mode structured --rounds 4 \
    --sparsity 0.55 --criterion bn_scale --allocate --out runs --tag vgg-bn

# 使用配置文件
python -m pruning_toolkit prune --config configs/allocated_vgg_bn_scale.json

# 生成配置模板
python -m pruning_toolkit template configs/my.json
```

数据集参数可通过 `--dataset-kwarg` 透传，例如
`--dataset-kwarg noise=1.1 --dataset-kwarg n_samples=2000`。

---

## 7. 示例与自检

| 脚本 | 内容 |
| --- | --- |
| `examples/01_unstructured_mlp.py` | 非结构化剪枝 90% + 掩码感知微调，对照稠密/非零参数量 |
| `examples/02_structured_cnn.py` | 结构化剪枝 50% 通道，依赖图 + 参数量/FLOPs/延迟对照 |
| `examples/03_iterative_vs_oneshot.py` | 迭代剪枝与一次性剪枝的对照（同等微调预算） |
| `examples/04_sensitivity_allocation.py` | 敏感度扫描（代理指标与真实精度对比）+ 逐层分配 |
| `examples/05_quickstart_own_model.py` | 接入自有模型的最小路径、导出与注意事项 |
| `tools/verify_models.py` | 全模型自检：依赖分析 → 两类剪枝 → 一致性断言 → 可复现表格 |

示例脚本产物写到 `runs/exampleN/`。`tools/verify_models.py` 不需要训练与数据，
几十秒完成，适合作为修改后的冒烟测试。

单元测试使用标准库 `unittest`，无需安装 pytest：

```bash
python -m unittest discover -s tests -v
```

当前共 81 个用例，通常数秒内完成，覆盖报告、调度、模块语义、依赖图、准则、掩码、
两类剪枝、指标、敏感度、迭代、配置、导出、CLI、流水线与公开 API，并包含错误路径与边界。

---

## 8. 配置文件

JSON 为默认格式（YAML 需先 `pip install pyyaml`，缺失时会给出明确提示）。
未知字段会报错而非被静默忽略，可避免 `criterion` 误写为 `critirion` 却未被察觉。

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

仓库内置三份可直接使用的配置：`configs/structured_vgg.json`、
`configs/unstructured_mlp.json`、`configs/allocated_vgg_bn_scale.json`。

> `sparsity` 的含义随 `mode` 变化：非结构化为目标稀疏度，结构化为通道裁剪比例。

### 稀疏度调度曲线

四种曲线描述累积目标如何爬升，每轮实际裁剪比例由 `incremental_ratios()` 换算。
以 `final=0.9, rounds=4` 为例：

| 曲线 | 第一轮目标 | 说明 |
| --- | --- | --- |
| `constant` | 0 | 前几轮不动，最后一轮剪到 0.9 |
| `linear` | 0.225 | 线性爬升：0.225 → 0.45 → 0.675 → 0.9 |
| `cubic`（默认） | **0.520** | 前期快、后期缓：0.520 → 0.788 → 0.886 → 0.9，对应 Zhu & Gupta (2017) 的 cubic 调度 |
| `exponential` | 0.056 | 起点为 0 时退化为平方曲线，否则按几何增长 |

`cubic` 常被误解为「前期剪得少」，实际相反：第一轮即剪掉一半以上，后续余量变小而自然放缓。

```python
from pruning_toolkit import cumulative_targets, incremental_ratios

print(cumulative_targets(0.9, 4, "cubic"))       # [0.52, 0.787, 0.886, 0.9]
print(incremental_ratios(cumulative_targets(0.9, 4, "cubic")))
# [0.5203, 0.557, 0.4632, 0.1233] —— 非严格单调，但后期动作明显收窄
```

### 准则一览

| 准则 | 粒度 | 需要梯度 | 说明 |
| --- | --- | --- | --- |
| `l1` | 元素 / 通道 | 否 | 绝对值，最常用且稳定 |
| `l2` | 元素 / 通道 | 否 | 平方 / L2 范数 |
| `random` | 元素 / 通道 | 否 | 随机基线（同 seed 可复现） |
| `taylor` | 元素 / 通道 | **是** | 一阶泰勒 `|w·grad|`，用 `collect_grads` 采集校准梯度 |
| `bn_scale` | 通道组 | 否 | 组内 BN 的 `|gamma|`（Network Slimming），无 BN 时退化为 `l2` |
| `fpgm` | 通道 | 否 | 到卷积核几何中位数的 L2 距离 |
| `none` | 元素 | 否 | 恒为 1，等价于按顺序裁剪 |

---

## 9. 用于检测模型（ultralytics / YOLO）

本工具的依赖图基于 `torch.fx`，对 ultralytics 系列检测模型（YOLOv8 / YOLO11）
**无法建立依赖图**，原因是两个独立的限制：

1. `BaseModel._predict_once` 对 Proxy 做迭代，`torch.fx` 无法追踪；
2. 工具将 `torch.cat` 与 `Upsample` 视为屏障，而这类模型的 PAN-FPN 颈部依赖这两者。

因此对检测模型有以下三条路径：

| 目标 | 方案 | 说明 |
| --- | --- | --- |
| 压缩存储 / 有效参数量 | `run.py`（本仓库） | 对 ultralytics 权重做非结构化剪枝 + 掩码感知微调，附带 mAP 评估；需自备权重与数据集 |
| ��低 FLOPs 与延迟 | `extras/yolov8_torch_pruning/reproduce.py` | 通道剪枝。需先安装 `torch-pruning`，并把 `C2f` 替换为无 `chunk` 的等价实现 |
| 直接更换模型规格 | — | 例如改用更小的官方变体，通常比剪枝更省事 |

`extras/` 中的脚本在 yolov8n 上实测（随机初始化权重，仅验证结构与计算量）：

```
基线: 参数 3.157 M, MACs 4.40 G
比例 0.2: 参数 2.213 M (-29.9%), MACs 3.25 G (-26.2%)
比例 0.5: 参数 1.175 M (-62.8%), MACs 2.02 G (-54.2%)
```

若不替换 `C2f`，`torch-pruning` 会在剪枝阶段抛出
`IndexError: index 384 is out of bounds for dimension 0 with size 256`——
原因是 `C2f` 内部的 `chunk` 产生的索引空间与后续 `cat` 不一致。
替换后数值完全等价（实测最大差异 0），再执行剪枝即可正常进行。

> 一个真实场景的参考：在某 9 类检测任务上，对 ultralytics 检测模型做 50% 非结构化
> 剪枝后，mAP50-95 由 0.975 降至 0.924，微调 3 轮后回到 0.982。该数据依赖私有数据集，
> 无法公开复现，仅作量级参考。

---

## 10. 模块结构

```
pruning_toolkit/
├── report.py          报告：table / kv / text / markdown → JSON + Markdown
├── schedules.py       稀疏度调度：cumulative_targets / incremental_ratios / Scheduler
├── modules.py         模块通道语义与参数裁剪原语（卷积 / 线性 / 归一化 / 深度卷积）
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

常用入口：

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

`IterativePruner` 的微调回调签名为 `finetune_fn(model, context)`，
其中 `context` 是 `RoundContext(round_index, target, achieved, masks)`：

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

---

## 11. 已知限制

1. **模型需可被 `torch.fx` 追踪**。`forward` 中存在依赖张量数值的 `if` / `while`，
   或使用了 fx 不支持的 Python 结构时，`DependencyGraph` 会抛出
   `UnsupportedGraphError` 并说明原因。此情况下非结构化剪枝仍可用（不需要依赖图）。
2. **拼接不传播**。`torch.cat` 两侧来自不同索引空间时，无法对位裁剪下游输入通道，
   相关通道组会被跳过并在报告中说明。
3. **输入侧与深度卷积的组合整组跳过**。若「输入输出同索引空间」的模块
   （深度卷积、归一化）直接作用在模型输入上，裁剪会改变输入宽度，工具将整组跳过。
4. **分组卷积不支持结构化剪枝**。`groups` 既不为 1 也不是深度卷积时作为屏障；
   深度卷积（`groups == in == out`）受支持，会同步更新 `groups` 与 `in/out_channels`。
5. **复合模块作为屏障**。`nn.MultiheadAttention`、`TransformerEncoderLayer` 等被 fx
   视为叶子模块，结构化剪枝时跳过；非结构化剪枝不受影响。若要裁剪注意力头，
   需先展开为显式 `Linear`。
6. **非结构化剪枝不提升稠密推理速度**。它降低有效参数量与存储占用；
   需要 FLOPs 与延迟收益时应使用结构化剪枝。报告会注明这一点。
7. **小模型的延迟数据参考价值有限**。CPU 上小模型单次前向仅数十微秒，抖动明显；
   建议以 p50 比较，并在相同负载条件下多次测量。
8. **结构化剪枝是破坏性的**。模型结构被就地修改，剪枝后的 `state_dict` 无法加载回
   原始结构的模型类；建议剪枝前 `copy.deepcopy` 保留基线，部署时使用导出的
   TorchScript，或依据 checkpoint 中的形状元信息重建结构。
9. **敏感度的重建误差为代理指标**，可能与真实精度排序不一致（见 §5.5）。
   需要可靠信号时使用 `proxy="accuracy"` 配合 `eval_fn` 做任务感知扫描，代价是耗时。
10. **逐层分配的目标为参数量加权平均**，而通道裁剪在层间级联，不同分配方案的最终
    参数量可能并不相等。严格比较时应先固定最终参数量或 FLOPs（见 §5.5）。
11. **ONNX 导出未在本仓库中实测**（开发环境未安装 `onnx`），代码在缺失时会给出
    安装提示；TorchScript 导出已实测可用。
12. **内置数据集为离线合成数据**（`blobs` / `two_moons` / `tiny_images`），
    目的是让工具本身可复现且无外部依赖。接入真实数据时，将 `example_inputs` 换为
    一批真实形状的数据，并把评估回调换成自有 `DataLoader` 即可。

---

## 12. 常见问题

**`DependencyGraph` 抛出 `UnsupportedGraphError` 怎么办？**
可先改用非结构化剪枝，或将 `forward` 改写为静态图形式（去掉依赖数值的分支）。
异常信息中包含原始的 fx 报错原因。

**为什么部分层没有被剪枝？**
运行 `python -m pruning_toolkit graph --model ...` 查看「跳过的通道组」及原因。
常见原因包括：位于 `cat` 之后、属于分组卷积、或为输出层（默认保护，可用
`protect_output=False` 关闭）。

**剪枝后精度下降如何处理？**
这属正常现象。先查看报告中的「剪枝后 acc」与「微调后 acc」两列，确认微调是否有效；
若微调后仍不可接受，可降低 `sparsity`、改用 `rounds>1` 的迭代剪枝，
或先用 `sensitivity` 定位敏感层。

**如何确保微调不会把剪掉的权重重新激活？**
非结构化剪枝需两步：在 `finetune_fn` 中传入
`fit(..., post_step_hook=lambda m: ctx.masks.apply(m))`；
`IterativePruner` 每轮结束时也会自动再压一次掩码。工具同时会校验残留零的个数。

**参数量明显下降，为什么延迟没有下降？**
稠密算子不会因为权重中包含零而变快；小模型的耗时由算子启动开销与内存带宽主导。
详见 §5.1 与 §11 第 6、7 条。

**是否支持 Transformer / LLM？**
当前版本对复合模块（如 `MultiheadAttention`）仅作屏障处理。若要裁剪注意力头，
需先将 `q_proj` / `k_proj` / `v_proj` / `out_proj` 展开为显式 `Linear`，
再使用本工具的通道组机制。该场景尚未提供开箱支持。

---

## 13. 贡献

欢迎提交 issue 与 pull request。提交前建议执行：

```bash
python -m unittest discover -s tests -v   # 单元测试（81 个用例）
python tools/verify_models.py             # 全模型自检
```

两个脚本均不需要下载数据或训练模型。若新增功能，请补充对应测试；
若修复缺陷，请在 PR 描述中说明复现步骤。

---

## 14. 许可证

本项目基于 [MIT 许可证](LICENSE) 发布，Copyright (c) 2026 Zhe-g。

可自由使用、修改、分发，包括商业用途，但需保留版权声明与许可声明。
