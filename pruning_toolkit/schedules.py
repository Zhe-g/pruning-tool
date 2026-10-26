"""稀疏度调度：把「最终稀疏度」拆成多轮迭代的逐轮目标。

迭代式剪枝（Iterative Magnitude Pruning 等）通常不是一步剪到目标稀疏度，
而是分多轮逐步加码，每轮之间做微调。本模块提供三种常用的累积稀疏度曲线，
并把累积目标换算成「本轮相对剩余权重还需剪掉的比例」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

__all__ = [
    "SCHEDULES",
    "cumulative_targets",
    "incremental_ratios",
    "per_round_ratio",
    "Scheduler",
]

#: 支持的调度曲线
SCHEDULES = ("constant", "linear", "cubic", "exponential")


def _validate(final_sparsity: float, rounds: int, initial_sparsity: float) -> None:
    if rounds < 1:
        raise ValueError("rounds 必须 >= 1，实际 %r" % (rounds,))
    if not 0.0 <= initial_sparsity < 1.0:
        raise ValueError("initial_sparsity 需在 [0, 1) 内，实际 %r" % (initial_sparsity,))
    if not initial_sparsity <= final_sparsity < 1.0:
        raise ValueError(
            "final_sparsity 需满足 initial_sparsity <= final < 1，实际 %r" % (final_sparsity,)
        )


def cumulative_targets(
    final_sparsity: float,
    rounds: int,
    schedule: str = "cubic",
    initial_sparsity: float = 0.0,
) -> List[float]:
    """返回每一轮结束时的**累积**稀疏度目标，长度等于 ``rounds``。

    四条曲线的区别（以 ``final=0.9, rounds=4`` 的第一轮目标为例）：

    * ``constant``：第一轮直接到目标（0.9），适合只跑一轮或想尽快压满稀疏度；
    * ``linear``：稀疏度线性增长（第一轮 0.225）；
    * ``cubic``：**前期快、后期缓**（第一轮 0.520 —— 因为后几轮只剩很少的余量可剪），
      对应 Zhu & Gupta (2017) 经典的 cubic 稀疏度调度，是本工具的默认值；
    * ``exponential``：几何增长，起点为 0 时退化为平方曲线。

    注意：这里所有曲线都只描述「累积目标怎么爬升」，真正的每轮动作由
    :func:`incremental_ratios` 换算成「本轮要剪掉剩余部分的比例」。
    """
    _validate(final_sparsity, rounds, initial_sparsity)
    if schedule not in SCHEDULES:
        raise KeyError("未知的调度曲线 %r，可选：%s" % (schedule, list(SCHEDULES)))

    span = final_sparsity - initial_sparsity
    targets: List[float] = []
    for step in range(1, rounds + 1):
        progress = step / float(rounds)
        if schedule == "constant":
            value = final_sparsity if step == rounds else initial_sparsity
        elif schedule == "linear":
            value = initial_sparsity + span * progress
        elif schedule == "cubic":
            value = initial_sparsity + span * (1.0 - (1.0 - progress) ** 3)
        else:  # exponential
            if initial_sparsity <= 0.0:
                # 起点为 0 时退化为平方曲线，避免 log(0)
                value = initial_sparsity + span * progress ** 2
            else:
                value = initial_sparsity * (final_sparsity / initial_sparsity) ** progress
        targets.append(float(min(max(value, 0.0), final_sparsity)))
    targets[-1] = float(final_sparsity)
    return targets


def incremental_ratios(targets: Sequence[float]) -> List[float]:
    """把累积稀疏度序列换算成本轮需要剪掉的**剩余比例**。

    第 t 轮的增量比例 ``r_t`` 满足 ``(1-s_t) = (1-s_{t-1}) * (1-r_t)``，
    即 ``r_t = 1 - (1-s_t)/(1-s_{t-1})``。结构化剪枝按单元计算时必须用这个
    比值，才能在删掉单元后仍然命中累积目标。
    """
    ratios: List[float] = []
    previous = 0.0
    for target in targets:
        if target < previous - 1e-12:
            raise ValueError("累积稀疏度必须单调不减，出现 %r -> %r" % (previous, target))
        remaining_before = max(1.0 - previous, 1e-12)
        remaining_after = max(1.0 - target, 0.0)
        ratio = 1.0 - remaining_after / remaining_before
        ratios.append(float(min(max(ratio, 0.0), 1.0)))
        previous = target
    return ratios


def per_round_ratio(final_sparsity: float, rounds: int) -> float:
    """若每轮剪掉相同的剩余比例，返回该比例。

    满足 ``(1-r)^rounds = 1-final_sparsity``，即 ``r = 1-(1-s)^(1/T)``。
    """
    _validate(final_sparsity, rounds, 0.0)
    return float(1.0 - (1.0 - final_sparsity) ** (1.0 / rounds))


@dataclass
class Scheduler:
    """调度器对象封装，方便写进配置或在训练循环里逐步查询。"""

    final_sparsity: float
    rounds: int
    schedule: str = "cubic"
    initial_sparsity: float = 0.0

    def __post_init__(self) -> None:
        self.targets: List[float] = cumulative_targets(
            self.final_sparsity, self.rounds, self.schedule, self.initial_sparsity
        )
        self.ratios: List[float] = incremental_ratios(self.targets)

    def target(self, step: int) -> float:
        """第 ``step`` 轮（从 1 开始）的累积稀疏度目标。"""
        if not 1 <= step <= self.rounds:
            raise IndexError("step 需在 1..%d 内，实际 %r" % (self.rounds, step))
        return self.targets[step - 1]

    def ratio(self, step: int) -> float:
        """第 ``step`` 轮需要剪掉的剩余比例。"""
        if not 1 <= step <= self.rounds:
            raise IndexError("step 需在 1..%d 内，实际 %r" % (self.rounds, step))
        return self.ratios[step - 1]
