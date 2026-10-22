"""输入时序数据的加载、预处理与严格数据划分。

通用数据链路::

    series.csv (单列物理通道, 采样率 fs)
        │  取前 use_fraction
        │  可选 Savitzky-Golay 平滑
        │  数值求导 (derivative_order 阶)
        ▼
    广义激励力 F(t) = forcing_scale * d^order x / dt^order
        │
        ▼
    数值参考解 x_ref(t)  (见 reference.py)
        │
        ▼
    训练 / 离散验证 / 严格隔离段 / 域外外推 四类索引

划分约定
--------
* 前 ``split_ratio`` 比例的时间域为“训练域”，其中 ``split_ratio`` 比例的随机点
  用于训练，其余随机点用于同分布离散验证；
* 末段完全不参与训练，用于域外外推测试；
* ``cont_val_start``/``cont_val_end`` 给定时，在训练域内额外**硬隔离**一段连续区间，
  该区间任何数据点都不参与损失计算与参数更新（用于检验方法的时序外推能力）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from .config import DataConfig, SystemConfig


@dataclass
class SeriesData:
    """输入通道及其派生量。"""

    #: 时间轴 [s]
    t: np.ndarray
    #: 原始被测通道采样值
    signal: np.ndarray
    #: 平滑（可选）并求导后的通道量
    derivative: np.ndarray
    #: 广义激励力 F [N]
    forcing: np.ndarray
    #: 采样率 [Hz]
    fs: float

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def duration(self) -> float:
        """记录时长 [s]。"""
        return float(self.t[-1]) if self.t.size else 0.0


@dataclass
class SplitResult:
    """数据划分结果（全部为索引数组）。"""

    #: 训练点索引
    train_idx: np.ndarray
    #: 同分布离散验证点索引
    val_idx: np.ndarray
    #: 域外外推点索引
    test_idx: np.ndarray
    #: 严格隔离的连续段索引（未隔离时为空数组）
    cont_val_idx: np.ndarray
    #: 训练域与外推域的分界点索引
    split_point: int

    def describe(self, t: Optional[np.ndarray] = None) -> str:
        """返回可读的划分摘要。"""
        lines = [
            f"训练点: {self.train_idx.size} 个",
            f"同分布离散验证点: {self.val_idx.size} 个",
            f"域外外推点: {self.test_idx.size} 个",
        ]
        if self.cont_val_idx.size:
            lines.append(f"严格隔离连续段: {self.cont_val_idx.size} 个")
        if t is not None:
            lines.append(
                f"训练域: 0 – {t[self.split_point - 1]:.1f} s | "
                f"外推域: {t[self.split_point]:.1f} – {t[-1]:.1f} s"
            )
        return "\n".join(lines)


def load_series(
    csv_path: str | Path,
    system: SystemConfig,
    data: Optional[DataConfig] = None,
    use_fraction: Optional[float] = None,
) -> SeriesData:
    """读取单列 CSV 并完成平滑、求导与激励力构造。

    Parameters
    ----------
    csv_path:
        单列（或多列）CSV 路径，内容为被测物理通道的采样值。
    system:
        系统参数（采样率 ``fs`` 由此提供）。
    data:
        数据预处理设置（平滑方式/窗口/阶数、求导阶数、激励力比例系数）。
    use_fraction:
        使用记录前端的比例；``None`` 时取 ``data.use_fraction``。

    Returns
    -------
    SeriesData
        含时间轴、原始通道、导数通道与广义激励力。
    """
    data = data or DataConfig()
    fraction = data.use_fraction if use_fraction is None else use_fraction

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到数据文件: {path}\n"
            "请参考 data/README.md 准备数据，或用 --data-csv 指定路径。"
        )

    raw = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    raw = np.atleast_1d(raw)
    if raw.ndim > 1:
        if data.column is None:
            raise ValueError(
                f"数据文件含 {raw.shape[1]} 列，请在配置中用 data.column 指定要使用的列（0 起）。"
            )
        raw = raw[:, data.column]
    raw = np.asarray(raw, dtype=float).flatten()

    n_total = int(raw.size)
    n_use = int(n_total * fraction)
    if n_use <= 0:
        raise ValueError(f"use_fraction={fraction} 导致可用数据点数为 0，请检查配置。")
    raw = raw[:n_use]

    t = np.arange(n_use) * system.dt

    # 可选平滑：抑制高频噪声在求导环节的放大
    if str(data.smoothing).lower() == "savgol":
        window = min(int(data.savgol_window), n_use if n_use % 2 else n_use - 1)
        if window <= data.savgol_polyorder:
            raise ValueError("savgol_window 必须大于 savgol_polyorder。")
        smoothed = savgol_filter(raw, window_length=window, polyorder=data.savgol_polyorder)
    else:
        smoothed = raw

    # 数值求导
    derivative = smoothed
    for _ in range(int(data.derivative_order)):
        derivative = np.gradient(derivative, system.dt)

    forcing = float(data.forcing_scale) * derivative

    print(f"[数据] 文件: {path}")
    print(f"[数据] 采样率 {system.fs} Hz，可用点数 {n_use:,}（总 {n_total:,}），时长 {t[-1]:.1f} s")
    print(f"[数据] 激励力范围: [{forcing.min():.3e}, {forcing.max():.3e}] N")
    print(f"[数据] 预期响应量级 std(F)/K ≈ {np.std(forcing) / system.stiffness:.3e}")

    return SeriesData(t=t, signal=raw, derivative=derivative, forcing=forcing, fs=system.fs)


def create_train_val_split(
    t: np.ndarray,
    split_ratio: float = 0.8,
    seed: int = 42,
    cont_val_start: Optional[float] = None,
    cont_val_end: Optional[float] = None,
    fs: Optional[float] = None,
) -> SplitResult:
    """构造严格无泄漏的训练/验证/外推/隔离段索引。

    Parameters
    ----------
    t:
        时间轴 [s]。
    split_ratio:
        训练域在整条记录中的占比。
    seed:
        随机种子，决定训练/验证点的随机划分。
    cont_val_start, cont_val_end:
        严格隔离连续段的起止时间 [s]；``None`` 或 ``start >= end`` 表示不隔离。
    fs:
        采样率 [Hz]；``None`` 时由 ``t`` 的相邻间隔推断。

    Returns
    -------
    SplitResult
    """
    np.random.seed(seed)
    n_total = int(len(t))
    split_point = int(n_total * split_ratio)

    if fs is None:
        fs = 1.0 / (t[1] - t[0]) if n_total > 1 else 1.0

    # 1) 严格隔离的连续段（可选）
    if cont_val_start is not None and cont_val_end is not None and cont_val_start < cont_val_end:
        c_start = int(cont_val_start * fs)
        c_end = int(cont_val_end * fs)
        cont_val_idx = np.arange(c_start, min(c_end, split_point))
    else:
        cont_val_idx = np.array([], dtype=int)

    # 2) 训练域内剩余点随机划分为训练集与离散验证集
    pre_indices = np.arange(split_point)
    mask_excl = ~np.isin(pre_indices, cont_val_idx)
    pre_indices_excl = pre_indices[mask_excl]

    n_train = int(len(pre_indices_excl) * split_ratio)
    shuffled = np.random.permutation(pre_indices_excl)
    train_idx = np.sort(shuffled[:n_train])
    val_idx = np.sort(shuffled[n_train:])

    # 3) 域外外推测试域（末段），完全不参与训练
    test_idx = np.arange(split_point, n_total)

    print("\n========== 数据划分 ==========")
    print(f"训练域: 0–{t[split_point - 1]:.1f} s | 训练集 {train_idx.size} 点 | 离散验证集 {val_idx.size} 点")
    if cont_val_idx.size:
        print(
            f"严格隔离连续段: {cont_val_start:.0f}–{cont_val_end:.0f} s, "
            f"{cont_val_idx.size} 点（不参与任何参数更新）"
        )
    else:
        print("严格隔离连续段: 无（默认模式）")
    print(f"域外外推域: {t[split_point]:.1f}–{t[-1]:.1f} s, {test_idx.size} 点")

    return SplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cont_val_idx=cont_val_idx,
        split_point=split_point,
    )
