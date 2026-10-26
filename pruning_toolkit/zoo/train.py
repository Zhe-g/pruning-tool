"""最小训练器：够用的训练 / 评估循环，外加剪枝工具需要的钩子。

这里刻意不实现任何「花哨」的东西（没有 AMP、没有分布式、没有早停），
只保证三件事：

1. :func:`evaluate` 只读地评估模型，**不改变** 模型原有的 ``train`` / ``eval`` 状态；
2. :func:`fit` 在**每个优化器 step 之后**调用 ``post_step_hook`` —— 剪枝工具
   靠这个钩子在每步把掩码重新压回权重，实现「掩码感知微调」
   （先 ``optimizer.step()`` 更新全部参数，再用掩码把被剪权重清零）；
3. 全程 CPU 友好：示例模型 10 个 epoch 远低于 1 分钟。

复现性：除 ``DataLoader`` 自身的固定 ``generator`` 外，:func:`fit` 在开始时还会用
显式 ``seed`` 重置一次全局随机源，保证同 seed 下初始化（若模型尚未初始化）与
dropout 等随机行为一致。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

__all__ = [
    "resolve_device",
    "evaluate",
    "fit",
    "trainable_parameters",
    "format_history",
]


def resolve_device(device: Any = "cpu") -> torch.device:
    """把 ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` / ``torch.device`` 统一成 :class:`torch.device`。

    参数:
        device: 设备字符串或 ``torch.device``。传 ``"cuda"`` 但 CUDA 不可用时退回 CPU。

    返回:
        ``torch.device`` 实例。
    """
    if isinstance(device, torch.device):
        resolved = device
    else:
        resolved = torch.device(str(device))
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def _batch_size_of(inputs: Any) -> int:
    """从模型输入里取批大小（张量取第 0 维，元组/列表/dict 取第一个张量）。"""
    if isinstance(inputs, torch.Tensor):
        return int(inputs.shape[0])
    if isinstance(inputs, (tuple, list)):
        for item in inputs:
            if isinstance(item, torch.Tensor):
                return int(item.shape[0])
        return 0
    if isinstance(inputs, dict):
        for item in inputs.values():
            if isinstance(item, torch.Tensor):
                return int(item.shape[0])
        return 0
    return 0


def _move(inputs: Any, device: torch.device) -> Any:
    """把张量 / 张量容器搬到目标设备。"""
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    if isinstance(inputs, (tuple, list)):
        return type(inputs)(_move(item, device) for item in inputs)
    if isinstance(inputs, dict):
        return {key: _move(value, device) for key, value in inputs.items()}
    return inputs


def evaluate(
    model: nn.Module,
    loader: Any,
    criterion: Optional[Callable[..., torch.Tensor]] = None,
    device: Any = "cpu",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """在 ``loader`` 上评估模型，返回平均损失与准确率。

    行为约定：
        * 内部用 ``torch.no_grad()``，不建计算图、不改梯度；
        * 内部临时切到 ``eval()``，**结束时恢复调用前的 ``training`` 状态**，
          因此可以在训练中途安全调用，不会破坏 BatchNorm 的运行统计；
        * 若模型原本处于 ``train()`` 模式，评估用的前向仍按 ``eval`` 行为执行
          （BatchNorm 用运行统计、Dropout 关闭），这是评估语义所要求的。

    参数:
        model: 待评估模型。
        loader: ``DataLoader``，每个 batch 为 ``(inputs, targets)``。
        criterion: 损失函数，默认 ``nn.CrossEntropyLoss()``。
        device: 计算设备，默认 ``"cpu"``。
        max_batches: 最多评估多少个 batch；``None`` 表示全部。

    返回:
        ``{"loss": float, "acc": float}``，``acc`` 为 top-1 准确率（0~1）。
    """
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    resolved = resolve_device(device)
    model.to(resolved)

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                inputs, targets = batch[0], batch[1]
                inputs = _move(inputs, resolved)
                targets = _move(targets, resolved)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                batch_size = _batch_size_of(targets) or _batch_size_of(inputs)
                total_loss += float(loss.item()) * batch_size
                total_correct += int((outputs.argmax(dim=1) == targets).sum().item())
                total_samples += batch_size
    finally:
        if was_training:
            model.train()

    if total_samples == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": total_loss / total_samples, "acc": total_correct / total_samples}


def _sync_batch_norm(model: nn.Module, loader: Any, device: torch.device) -> None:
    """跑一遍完整数据以刷新 BatchNorm 运行统计（仅在 ``bn_sync=True`` 且模型原本是 train 模式时使用）。"""
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            for batch in loader:
                model(_move(batch[0], device))
    finally:
        model.train(was_training)


def fit(
    model: nn.Module,
    train_loader: Any,
    test_loader: Optional[Any] = None,
    epochs: int = 10,
    lr: float = 1e-2,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    device: Any = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    post_step_hook: Optional[Callable[[nn.Module], None]] = None,
    log: Optional[Callable[[str], None]] = None,
    log_every: int = 0,
    seed: int = 0,
    criterion: Optional[Callable[..., torch.Tensor]] = None,
    max_batches: Optional[int] = None,
    bn_sync: bool = False,
) -> Dict[str, List[float]]:
    """标准训练循环：交叉熵损失 + SGD（或调用方传入的 ``optimizer``）。

    每个优化器 step 之后都会调用一次 ``post_step_hook(model)``。剪枝工具用它在每步
    重新应用掩码（把被剪权重强制清零），从而在微调过程中保持稀疏结构不变。

    参数:
        model: 待训练模型。
        train_loader: 训练 ``DataLoader``，每个 batch 为 ``(inputs, targets)``。
        test_loader: 可选测试 ``DataLoader``；给了才会记录 ``test_*`` 曲线。
        epochs: 训练轮数。
        lr: SGD 学习率（``optimizer`` 已给定时忽略）。
        momentum: SGD 动量（``optimizer`` 已给定时忽略）。
        weight_decay: SGD 权重衰减（``optimizer`` 已给定时忽略）。
        device: 计算设备，默认 ``"cpu"``。
        optimizer: 外部优化器；给了就用它（便于剪枝工具自己配置参数组）。
        scheduler: 可选学习率调度器；每个 epoch 结束时调用一次
            （优先无参 ``step()``，旧式调度器若报 ``TypeError`` 则退回 ``step(epoch)``）。
        post_step_hook: 每个优化器 step 之后调用，签名 ``hook(model) -> None``。
        log: 日志回调，默认 ``print``；传 ``None`` 也用默认 ``print``，
            传一个空函数即可静默。
        log_every: 每多少个 step 打一条日志；``0`` 表示只按 epoch 打。
        seed: 随机种子；训练开始前用 ``torch.manual_seed(seed)`` 固定随机源。
        criterion: 损失函数，默认 ``nn.CrossEntropyLoss()``。
        max_batches: 每个 epoch 最多训练多少个 batch；``None`` 表示全部（用于快速冒烟测试）。
        bn_sync: 是否在训练前用 ``train_loader`` 跑一遍以刷新 BatchNorm 运行统计。
            默认 ``False`` —— 这是**有意为之**：剪枝工具在微调前常把运行统计清零或重置，
            若这里偷偷跑一遍全量数据，会破坏「清零未训练 BN」的语义，让对照实验不可比。
            模型刚从 ``train()`` 状态来、想恢复统计时再显式打开。

    返回:
        历史字典，键固定为 ``"train_loss"`` / ``"train_acc"`` / ``"test_loss"`` / ``"test_acc"``，
        值为每 epoch 一个 float 的列表。未给 ``test_loader`` 时 ``test_*`` 始终为空列表。
    """
    resolved = resolve_device(device)
    torch.manual_seed(int(seed))

    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    logger = log if log is not None else print

    model.to(resolved)
    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay)
        )

    if bn_sync:
        _sync_batch_norm(model, train_loader, resolved)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        started = time.time()

        for step, batch in enumerate(train_loader, start=1):
            if max_batches is not None and step > int(max_batches):
                break
            inputs, targets = batch[0], batch[1]
            inputs = _move(inputs, resolved)
            targets = _move(targets, resolved)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if post_step_hook is not None:
                # 掩码感知微调的关键位置：step 之后立刻把掩码压回权重。
                post_step_hook(model)

            batch_size = _batch_size_of(targets) or _batch_size_of(inputs)
            epoch_loss += float(loss.item()) * batch_size
            epoch_correct += int((outputs.argmax(dim=1) == targets).sum().item())
            epoch_samples += batch_size

            if log_every and logger is not None and step % int(log_every) == 0:
                logger(
                    "  [epoch %d] step %d：loss=%.4f，acc=%.4f"
                    % (step, step, float(loss.item()), float((outputs.argmax(dim=1) == targets).float().mean().item()))
                )

        train_loss = epoch_loss / epoch_samples if epoch_samples else float("nan")
        train_acc = epoch_correct / epoch_samples if epoch_samples else float("nan")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        message = "epoch %d/%d：train_loss=%.4f，train_acc=%.4f" % (
            epoch,
            int(epochs),
            train_loss,
            train_acc,
        )

        if test_loader is not None:
            metrics = evaluate(model, test_loader, criterion=criterion, device=resolved)
            history["test_loss"].append(float(metrics["loss"]))
            history["test_acc"].append(float(metrics["acc"]))
            message += "，test_loss=%.4f，test_acc=%.4f" % (metrics["loss"], metrics["acc"])
        else:
            history["test_loss"].append(float("nan"))
            history["test_acc"].append(float("nan"))

        if scheduler is not None:
            # 优先使用无参 step()（新式链式调度器）；旧式调度器要求传入 epoch 时再退回。
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(epoch)

        message += "（%.1fs）" % (time.time() - started)
        if logger is not None:
            logger(message)

    return history


def trainable_parameters(model: nn.Module) -> int:
    """统计 ``requires_grad=True`` 的参数元素个数。

    参数:
        model: 任意 ``nn.Module``。

    返回:
        可训练参数个数（int）。剪枝前后对比这个数字最直观。
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def format_history(history: Dict[str, List[float]], title: str = "训练历史") -> str:
    """把 :func:`fit` 返回的历史渲染成简短的中文文本表格，供 CLI 直接打印。

    参数:
        history: ``fit`` 的返回值；缺键按「无该列」处理。
        title: 表头上方的一行标题。

    返回:
        多行字符串。``train_loss`` / ``train_acc`` / ``test_loss`` / ``test_acc``
        四列中，若某列全为 ``nan``（例如没给 ``test_loader``）则省略该列。
    """
    train_loss = list(history.get("train_loss") or [])
    train_acc = list(history.get("train_acc") or [])
    test_loss = list(history.get("test_loss") or [])
    test_acc = list(history.get("test_acc") or [])

    def _meaningful(values: List[float]) -> bool:
        if not values:
            return False
        for value in values:
            if value == value:  # 排除 nan
                return True
        return False

    columns: List[str] = ["epoch", "train_loss", "train_acc"]
    if _meaningful(test_loss):
        columns.append("test_loss")
    if _meaningful(test_acc):
        columns.append("test_acc")

    n_epochs = max(len(train_loss), len(train_acc), len(test_loss), len(test_acc))
    rows: List[List[str]] = []
    for index in range(n_epochs):
        row = [str(index + 1)]
        row.append("%.4f" % train_loss[index] if index < len(train_loss) else "-")
        row.append("%.4f" % train_acc[index] if index < len(train_acc) else "-")
        if "test_loss" in columns:
            row.append("%.4f" % test_loss[index] if index < len(test_loss) else "-")
        if "test_acc" in columns:
            row.append("%.4f" % test_acc[index] if index < len(test_acc) else "-")
        rows.append(row)

    widths = [len(column) for column in columns]
    for row in rows:
        for position, cell in enumerate(row):
            widths[position] = max(widths[position], len(cell))

    def _line(cells: List[str]) -> str:
        return "  ".join(cell.ljust(widths[position]) for position, cell in enumerate(cells)).rstrip()

    out = [title, _line(columns), _line(["-" * width for width in widths])]
    for row in rows:
        out.append(_line(row))
    return "\n".join(out)
