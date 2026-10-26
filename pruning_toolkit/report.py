"""剪枝报告：结构化数据 + JSON / Markdown 双格式渲染。

所有剪枝、评估、分析的结果都统一表示成 :class:`Report`：内部是一串
*小节（section）*，每节只允许四种形态 —— 表格、键值对、纯文本、原始
Markdown。这样同一份结果既能落成便于程序读取的 JSON，也能直接落成
便于人阅读的 Markdown 报告，不需要维护两套输出逻辑。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "TOOL_NAME",
    "TOOL_VERSION",
    "table",
    "kv",
    "text",
    "markdown",
    "format_value",
    "Report",
    "render_markdown",
]

TOOL_NAME = "pruning-toolkit"
TOOL_VERSION = "1.0.0"

TABLE = "table"
KV = "kv"
TEXT = "text"
MARKDOWN = "markdown"


# ---------------------------------------------------------------------- 小节构造


def table(
    name: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """构造表格小节。``rows`` 中每个元素长度应与 ``columns`` 一致。"""
    normalized = []
    for row in rows:
        values = list(row)
        if len(values) != len(columns):
            raise ValueError(
                "表格 %r 的行长度 %d 与列数 %d 不一致" % (name, len(values), len(columns))
            )
        normalized.append(values)
    return {"name": name, "kind": TABLE, "columns": list(columns), "rows": normalized, "note": note}


def kv(name: str, items: Mapping[str, Any], note: Optional[str] = None) -> Dict[str, Any]:
    """构造键值对小节（已按传入顺序保留）。"""
    return {"name": name, "kind": KV, "items": dict(items), "note": note}


def text(name: str, body: str, note: Optional[str] = None) -> Dict[str, Any]:
    """构造纯文本小节。"""
    return {"name": name, "kind": TEXT, "body": body, "note": note}


def markdown(name: str, body: str) -> Dict[str, Any]:
    """构造原始 Markdown 小节。"""
    return {"name": name, "kind": MARKDOWN, "body": body}


# ---------------------------------------------------------------------- 数值格式化


def format_value(value: Any, precision: int = 4) -> str:
    """把任意值格式化成适合放进表格的短字符串。"""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        magnitude = abs(value)
        if magnitude != 0 and (magnitude < 1e-4 or magnitude >= 1e6):
            return "%.3e" % value
        if magnitude >= 1000 or float(value).is_integer():
            return "%.0f" % value
        return ("%." + str(precision) + "f") % value
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(item, precision) for item in value)
    if isinstance(value, Mapping):
        return ", ".join("%s=%s" % (k, format_value(v, precision)) for k, v in value.items())
    return str(value)


def _escape_cell(value: Any, precision: int = 4) -> str:
    return format_value(value, precision).replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------------- Report


@dataclass
class Report:
    """一份剪枝报告。

    典型用法::

        report = Report("结构化剪枝报告", model_name="demo-cnn")
        report.add_table("逐层结果", ["层", "保留通道"], [["conv1", 32]])
        report.save("report.md")
    """

    title: str
    model_name: Optional[str] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    tool: str = TOOL_NAME
    version: str = TOOL_VERSION

    # ------------------------------------------------------------------ 添加小节
    def add(self, section: Dict[str, Any]) -> "Report":
        if not isinstance(section, dict) or "kind" not in section:
            raise TypeError("小节必须是 table()/kv()/text()/markdown() 的返回值")
        self.sections.append(section)
        return self

    def add_table(self, name, columns, rows, note=None) -> "Report":
        return self.add(table(name, columns, rows, note=note))

    def add_kv(self, name, items, note=None) -> "Report":
        return self.add(kv(name, items, note=note))

    def add_text(self, name, body, note=None) -> "Report":
        return self.add(text(name, body, note=note))

    def add_markdown(self, name, body) -> "Report":
        return self.add(markdown(name, body))

    def has_section(self, name: str) -> bool:
        return any(section.get("name") == name for section in self.sections)

    # ------------------------------------------------------------------ 序列化
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "model_name": self.model_name,
            "tool": self.tool,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "sections": self.sections,
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)
        if path:
            directory = os.path.dirname(os.path.abspath(os.fspath(path)))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(os.fspath(path), "w", encoding="utf-8") as handle:
                handle.write(payload)
        return payload

    def to_markdown(self) -> str:
        return render_markdown(self.to_dict())

    def to_text(self) -> str:
        """把报告渲染成等宽文本，便于直接打印到终端。"""
        lines = [self.title]
        if self.model_name:
            lines.append("模型：%s" % self.model_name)
        lines.append("生成时间：%s    工具：%s v%s" % (self.created_at, self.tool, self.version))
        for key, value in self.metadata.items():
            lines.append("%s：%s" % (key, format_value(value)))
        for section in self.sections:
            lines.append("")
            lines.append("== %s ==" % section.get("name", ""))
            kind = section.get("kind")
            if kind == TABLE:
                for row in section.get("rows", []):
                    lines.append("  " + " | ".join(_escape_cell(v) for v in row))
            elif kind == KV:
                for key, value in section.get("items", {}).items():
                    lines.append("  %s: %s" % (key, format_value(value)))
            elif kind in (TEXT, MARKDOWN):
                lines.append(str(section.get("body", "")))
            if section.get("note"):
                lines.append("  注：%s" % section["note"])
        return "\n".join(lines)

    def save(self, path: str) -> str:
        """按扩展名选择格式：``.json`` 存 JSON，``.md/.markdown`` 存 Markdown。"""
        target = os.fspath(path)
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        suffix = os.path.splitext(target)[1].lower()
        if suffix == ".json":
            self.to_json(target)
        elif suffix in (".md", ".markdown"):
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(self.to_markdown())
        else:
            raise ValueError("不认识报告扩展名 %r，请使用 .json / .md" % suffix)
        return target

    def __len__(self) -> int:
        return len(self.sections)


# ---------------------------------------------------------------------- Markdown 渲染


def render_markdown(payload: Mapping[str, Any]) -> str:
    """把 :meth:`Report.to_dict` 的结果渲染成 Markdown。"""
    lines: List[str] = ["# %s" % payload.get("title", "剪枝报告"), ""]
    info = []
    if payload.get("model_name"):
        info.append("**模型**：%s" % payload["model_name"])
    info.append("**生成时间**：%s" % payload.get("created_at", "-"))
    info.append("**工具**：%s v%s" % (payload.get("tool", TOOL_NAME), payload.get("version", TOOL_VERSION)))
    lines.append("  \n".join(info))
    lines.append("")

    metadata = payload.get("metadata") or {}
    if metadata:
        lines.append("| 项 | 值 |")
        lines.append("| --- | --- |")
        for key, value in metadata.items():
            lines.append("| %s | %s |" % (_escape_cell(key), _escape_cell(value)))
        lines.append("")

    for section in payload.get("sections", []):
        name = section.get("name", "")
        lines.append("## %s" % name)
        lines.append("")
        kind = section.get("kind")
        if kind == TABLE:
            columns = section.get("columns", [])
            lines.append("| " + " | ".join(str(c) for c in columns) + " |")
            lines.append("| " + " | ".join("---" for _ in columns) + " |")
            for row in section.get("rows", []):
                lines.append("| " + " | ".join(_escape_cell(v) for v in row) + " |")
        elif kind == KV:
            lines.append("| 项 | 值 |")
            lines.append("| --- | --- |")
            for key, value in section.get("items", {}).items():
                lines.append("| %s | %s |" % (_escape_cell(key), _escape_cell(value)))
        elif kind in (TEXT, MARKDOWN):
            lines.append(str(section.get("body", "")))
        else:
            lines.append("```json")
            lines.append(json.dumps(section, ensure_ascii=False, indent=2, default=str))
            lines.append("```")
        if section.get("note"):
            lines.append("")
            lines.append("> %s" % section["note"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
