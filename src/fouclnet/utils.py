"""通用工具：随机种子、设备选择、路径与日志。"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Optional

# 仓库根目录（src/fouclnet/utils.py -> src/fouclnet -> src -> 仓库根）
REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """统一设置 Python / NumPy / PyTorch 的随机种子。

    Parameters
    ----------
    seed:
        随机种子。默认主实验使用 42。
    deterministic:
        为 ``True`` 时开启 PyTorch 确定性算法（会牺牲部分性能）。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy 缺失时忽略
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - 未安装 torch 时忽略
        pass


def get_device(spec: str = "auto"):
    """根据配置字符串选择计算设备。

    Parameters
    ----------
    spec:
        ``auto`` / ``cuda`` / ``cpu``，或形如 ``cuda:1`` 的显式设备。

    Returns
    -------
    torch.device
    """
    import torch

    if spec and spec != "auto":
        if spec.startswith("cuda") and not torch.cuda.is_available():
            print("[警告] 指定了 CUDA 设备但当前环境不可用，回退到 CPU。")
            return torch.device("cpu")
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    """创建目录（若不存在）并返回其 :class:`~pathlib.Path`。"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(path: str | Path, base: Optional[str | Path] = None) -> Path:
    """把相对路径解析为基于 ``base``（默认仓库根目录）的绝对路径。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(base) if base is not None else REPO_ROOT
    return (root / candidate).resolve()


def save_json(payload: Any, path: str | Path) -> Path:
    """把结果以 UTF-8 JSON 写入 ``path``（自动创建父目录）。"""
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def load_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def banner(title: str, width: int = 70) -> str:
    """返回统一风格的分节标题，便于日志阅读。"""
    return "\n" + "=" * width + f"\n{title}\n" + "=" * width
