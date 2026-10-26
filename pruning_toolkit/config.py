"""配置系统：一次剪枝实验的完整参数，支持 JSON 与 YAML。

配置文件是「可复现实验」的载体：把模型、数据集、剪枝策略、微调、输出全部写进一个
文件，任何人拿到它都能跑出同样的结果（配合固定 ``seed``）。

格式
----
* ``.json``  —— 无额外依赖，推荐；
* ``.yaml`` / ``.yml`` —— 需要 ``pip install pyyaml``，未安装时会给出明确提示。

未知字段会**报错**而不是被忽略，避免把 ``criterion`` 写成 ``critirion`` 却毫无察觉。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Mapping, Optional

from .criteria import CRITERIA
from .schedules import SCHEDULES

__all__ = [
    "TrainConfig",
    "FinetuneConfig",
    "OutputConfig",
    "PruneConfig",
    "load_config",
]


@dataclass
class TrainConfig:
    """剪枝**之前**的初始训练（可选）。

    如果模型已经有训练好的权重，把 ``enabled`` 设为 ``False`` 即可。
    """

    enabled: bool = True
    epochs: int = 6
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    max_batches: Optional[int] = None

    def validate(self) -> None:
        if self.epochs < 0:
            raise ValueError("train.epochs 不能为负")
        if self.lr <= 0:
            raise ValueError("train.lr 必须为正")


@dataclass
class FinetuneConfig:
    """每轮剪枝之后的微调。"""

    enabled: bool = True
    epochs_per_round: int = 2
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0

    def validate(self) -> None:
        if self.epochs_per_round < 0:
            raise ValueError("finetune.epochs_per_round 不能为负")
        if self.lr <= 0:
            raise ValueError("finetune.lr 必须为正")


@dataclass
class OutputConfig:
    """输出产物设置。"""

    directory: str = "runs"
    tag: str = ""
    write_reports: bool = True
    write_checkpoint: bool = True
    write_masks: bool = True
    export_torchscript: bool = False

    @property
    def run_directory(self) -> str:
        return os.path.join(self.directory, self.tag) if self.tag else self.directory


@dataclass
class PruneConfig:
    """一次剪枝实验的完整配置。"""

    # ---- 模型与数据 ----
    model: str = "small_vgg"
    dataset: str = "tiny_images"
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 64
    device: str = "cpu"
    seed: int = 0

    # ---- 剪枝策略 ----
    mode: str = "structured"
    criterion: str = "l1"
    scope: str = "local"
    schedule: str = "cubic"
    rounds: int = 3
    sparsity: float = 0.5
    per_layer: Dict[str, float] = field(default_factory=dict)
    exclude: List[str] = field(default_factory=list)
    protect_output: bool = True
    protect_first_last: bool = False

    # ---- 自动分配逐层稀疏度（敏感度驱动）----
    allocate: bool = False
    alpha: float = 1.0
    reference: float = 0.5

    # ---- 评估与产物 ----
    benchmark: bool = True
    verbose: bool = True
    train: TrainConfig = field(default_factory=TrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------ 构造
    def __post_init__(self) -> None:
        """允许用字典直接构造嵌套配置，例如 ``PruneConfig(train={"epochs": 3})``。

        dataclass 本身不会做这种转换，而「用字面量字典写配置」是最自然的用法
        （CLI、示例、测试都会这么写），所以在这里统一收敛：
        嵌套字段若是 Mapping 就转成对应的 dataclass，未知键照旧报错。
        """
        nested = {
            "train": TrainConfig,
            "finetune": FinetuneConfig,
            "output": OutputConfig,
        }
        for key, klass in nested.items():
            value = getattr(self, key)
            if isinstance(value, Mapping):
                allowed = {item.name for item in fields(klass)}
                unknown = set(value) - allowed
                if unknown:
                    raise ValueError(
                        "%s 中有未知字段：%s；可接受字段：%s"
                        % (key, ", ".join(sorted(unknown)), ", ".join(sorted(allowed)))
                    )
                setattr(self, key, klass(**value))

    # ------------------------------------------------------------------ 校验
    def validate(self) -> None:
        """检查字段合法性，出错时抛出带字段路径的 ``ValueError``。"""
        if self.mode not in ("structured", "unstructured"):
            raise ValueError("mode 只能是 'structured' 或 'unstructured'，实际 %r" % self.mode)
        if self.criterion not in CRITERIA:
            raise KeyError(
                "criterion %r 未知，可选：%s" % (self.criterion, ", ".join(sorted(CRITERIA)))
            )
        if self.scope not in ("global", "local"):
            raise ValueError("scope 只能是 'global' 或 'local'，实际 %r" % self.scope)
        if self.schedule not in SCHEDULES:
            raise KeyError(
                "schedule %r 未知，可选：%s" % (self.schedule, ", ".join(SCHEDULES))
            )
        if self.rounds < 1:
            raise ValueError("rounds 必须 >= 1")
        if not 0.0 < self.sparsity < 1.0:
            raise ValueError(
                "sparsity 需在 (0, 1) 内（非结构化=目标稀疏度，结构化=通道裁剪比例）"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size 必须为正整数")
        for name, value in self.per_layer.items():
            if not 0.0 <= float(value) < 1.0:
                raise ValueError("per_layer[%r] 需在 [0, 1) 内，实际 %r" % (name, value))
        self.train.validate()
        self.finetune.validate()

    # ------------------------------------------------------------------ 序列化
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PruneConfig":
        """从字典构造；未知字段报错并列出全部可接受字段。"""
        data = dict(payload)
        nested = {
            "train": TrainConfig,
            "finetune": FinetuneConfig,
            "output": OutputConfig,
        }
        for key, klass in nested.items():
            if key in data and isinstance(data[key], Mapping):
                allowed = {item.name for item in fields(klass)}
                unknown = set(data[key]) - allowed
                if unknown:
                    raise ValueError(
                        "%s 中有未知字段：%s；可接受字段：%s"
                        % (key, ", ".join(sorted(unknown)), ", ".join(sorted(allowed)))
                    )
                data[key] = klass(**data[key])
        allowed_top = {item.name for item in fields(cls)}
        unknown_top = set(data) - allowed_top
        if unknown_top:
            raise ValueError(
                "配置中有未知字段：%s；可接受字段：%s"
                % (", ".join(sorted(unknown_top)), ", ".join(sorted(allowed_top)))
            )
        return cls(**data)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path:
            directory = os.path.dirname(os.path.abspath(os.fspath(path)))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(os.fspath(path), "w", encoding="utf-8") as handle:
                handle.write(text)
        return text

    def to_yaml(self, path: Optional[str] = None) -> str:
        yaml = _import_yaml()
        text = yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)
        if path:
            directory = os.path.dirname(os.path.abspath(os.fspath(path)))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(os.fspath(path), "w", encoding="utf-8") as handle:
                handle.write(text)
        return text

    def save(self, path: str) -> str:
        """按扩展名选择格式保存（``.json`` 或 ``.yaml``/``.yml``）。"""
        suffix = os.path.splitext(os.fspath(path))[1].lower()
        if suffix == ".json":
            self.to_json(path)
        elif suffix in (".yaml", ".yml"):
            self.to_yaml(path)
        else:
            raise ValueError("不认识的配置扩展名 %r，请用 .json / .yaml / .yml" % suffix)
        return os.fspath(path)

    def with_overrides(self, **overrides: Any) -> "PruneConfig":
        """返回替换了部分字段的新配置（不修改原对象）。"""
        nested = {"train", "finetune", "output"}
        flat: Dict[str, Any] = {}
        for key, value in overrides.items():
            if value is None:
                continue
            if key in nested and isinstance(value, Mapping):
                current = getattr(self, key)
                flat[key] = replace(current, **value)
            else:
                flat[key] = value
        return replace(self, **flat)

    # ------------------------------------------------------------------ 预设
    @classmethod
    def for_demo(cls) -> "PruneConfig":
        """开箱即用的演示配置：小 VGG + 合成图像 + 结构化剪枝 + 迭代微调。"""
        return cls(
            model="small_vgg",
            dataset="tiny_images",
            dataset_kwargs={"n_samples": 1500, "noise": 0.6},
            batch_size=64,
            train=TrainConfig(enabled=True, epochs=8, lr=0.01),
            mode="structured",
            criterion="l1",
            scope="local",
            schedule="cubic",
            rounds=3,
            sparsity=0.5,
            finetune=FinetuneConfig(enabled=True, epochs_per_round=2, lr=0.005),
            output=OutputConfig(directory="runs", tag="demo"),
        )

    @classmethod
    def for_mlp_demo(cls) -> "PruneConfig":
        """MLP + blobs 的非结构化剪枝演示配置。"""
        return cls(
            model="mlp",
            dataset="blobs",
            dataset_kwargs={"n_samples": 1500, "n_classes": 3},
            batch_size=64,
            train=TrainConfig(enabled=True, epochs=40, lr=0.05),
            mode="unstructured",
            criterion="l1",
            scope="global",
            schedule="cubic",
            rounds=4,
            sparsity=0.9,
            finetune=FinetuneConfig(enabled=True, epochs_per_round=15, lr=0.02),
            output=OutputConfig(directory="runs", tag="mlp-demo"),
        )


def _import_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "读写 YAML 配置需要 pyyaml：pip install pyyaml；"
            "或者改用 JSON 配置（.json）——本工具默认格式就是 JSON。"
        ) from exc
    return yaml


def load_config(path: str) -> PruneConfig:
    """读取配置：按扩展名选择 JSON 或 YAML。"""
    target = os.fspath(path)
    suffix = os.path.splitext(target)[1].lower()
    with open(target, "r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in (".yaml", ".yml"):
            payload = _import_yaml().safe_load(handle)
        else:
            raise ValueError("不认识的配置扩展名 %r，请用 .json / .yaml / .yml" % suffix)
    if not isinstance(payload, Mapping):
        raise ValueError("配置文件顶层必须是一个映射（字典）：%s" % target)
    config = PruneConfig.from_dict(payload)
    config.validate()
    return config
