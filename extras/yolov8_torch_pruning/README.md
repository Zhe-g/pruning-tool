# 用 torch-pruning 剪 YOLOv8（实测可复现）

本目录是**在用户机器上真实跑通**的 YOLOv8 结构化剪枝 recipe，
对应上游官方示例 [examples/yolov8/yolov8_pruning.py](https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py)
（该示例改编自 [issue #147](https://github.com/VainF/Torch-Pruning/issues/147)）。

## 结论先行

| 做法 | 结果 |
| --- | --- |
| 直接 `tp.MagnitudePruner(yolov8n)` | ❌ `IndexError: index 384 is out of bounds for dimension 0 with size 256` |
| 先把 8 个 `C2f` 换成无 `chunk` 的等价实现 `C2fV2`，再剪 | ✅ 成功，见下方实测数据 |

## 实测数据

环境：`C:\Users\GuoZhe\.conda\envs\yolov8`（ultralytics 8.3.133 / torch 2.7.0+cu126），
torch-pruning 1.6.1，`yolov8n.yaml`（随机初始化，只验证结构与计算量，不涉及精度）。

```
替换 C2f → C2fV2: 8 个
改造前后最大数值差: 0.000e+00        ← 权重迁移正确，改造是数学等价的
基线: 参数 3.157 M, MACs 4.40 G
比例 0.2: 参数 2.213 M (-29.9%), MACs 3.25 G (-26.2%), 前向 OK
比例 0.5: 参数 1.175 M (-62.8%), MACs 2.02 G (-54.2%), 前向 OK
```

注意：通道剪 50% 换来参数 -62.8%、MACs -54.2%（通道删除会在层间级联）。

## 为什么必须改造 C2f

`ultralytics` 的 `C2f.forward` 里有一句

```python
y = list(self.cv1(x).chunk(2, 1))     # chunk → 在 DepGraph 里变成 _SplitOp
```

`chunk` 产生的 split 索引空间与后续 `torch.cat` 的空间对不上，导致剪枝组的索引越界
（实测问题组里同时混着 **128 通道的分支**和 **_SplitOp 的 512 通道空间**）。

官方解法是把 `chunk` 换成**两个独立卷积**，并把原 `cv1` 的权重/BN 按通道切成两半迁移过去：

```python
class C2f_v2(nn.Module):
    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]        # 不再 chunk
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```

本目录的 `reproduce.py` 就是这个思路，并且做了两处 8.3.x 适配（见下）。

## 怎么跑

```powershell
# torch-pruning 需要自己装（本目录的脚本我是在临时目录里 --no-deps 装的，你的环境里没有）
& 'C:\Users\GuoZhe\.conda\envs\yolov8\python.exe' -m pip install torch-pruning

cd E:\ItemCodeFlies\pruning
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
& 'C:\Users\GuoZhe\.conda\envs\yolov8\python.exe' extras\yolov8_torch_pruning\reproduce.py
```

## 版本陷阱（重要）

官方示例写于 ultralytics **8.0.x**，直接抄到 8.3.x 会 import 失败——它用的是
`ultralytics.yolo.engine.model`、`ultralytics.yolo.utils` 这些**已被移除**的路径
（上游 [issue #370](https://github.com/VainF/Torch-Pruning/issues/370) 正是在讨论升级问题）。

`reproduce.py` 相对官方示例做了两处适配，其余思路一致：

1. **不重新构造 Bottleneck**，直接复用原始 `c2f.m`（新版 `Bottleneck` 的签名变了，
   官方示例里的 `k=((3, 3), (3, 3))` 传参在 8.3.x 不适用）；
2. **显式拷贝 `f` / `i` / `np` 等元属性**——ultralytics 的 `_predict_once` 依赖它们，
   漏掉会报 `AttributeError: 'C2fV2' object has no attribute 'f'`。

## 只做到"能剪"还不够：完整流程还差什么

本脚本只验证到「剪枝后前向仍然正确」。要真正拿到可用的压缩模型，还需要（官方示例里有）：

1. **剪完必须重新微调**，否则 mAP 会大幅下降（通道被物理删除，信息不可恢复）；
2. **用 mAP 评估**而不是分类准确率——调 ultralytics 的 `model.val()`；
3. **忽略整个 `Detect` 头**（`ignored_layers=[检测头]`），否则会破坏 `4*reg_max + nc` 的输出布局；
4. 迭代式剪枝（多轮「剪一点 → 微调」）+ `target_prune_rate` 控制总比例；
5. `round_to=8` 之类的对齐约束，让通道数是硬件友好的倍数；
6. 最后 `model.export(format='onnx')` 落地产物。

## 与 `pruning_toolkit` 的关系

本仓库自带的 `pruning_toolkit` **不能**剪 YOLOv8，且这不是配置问题：

- 它的依赖图基于 `torch.fx`，而 ultralytics 的 `BaseModel._predict_once` 对 Proxy 做迭代
  （`for i, m in enumerate(self.model)`），fx 直接 `TraceError: Proxy object cannot be iterated`
  ——**即使先做了 C2f 改造也一样**（已实测）；
- 它把 `torch.cat` / `Upsample` 当作屏障，而 YOLOv8 的 PAN-FPN 颈部全靠这两者。

所以两者定位不同：`pruning_toolkit` 面向 **fx 可追踪的常规分类/回归网络**（MLP、VGG 风格 CNN、
残差网络、深度可分离 CNN），对 YOLOv8 这类复合检测网络请用 torch-pruning。
