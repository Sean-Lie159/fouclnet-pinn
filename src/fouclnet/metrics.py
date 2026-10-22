"""评价指标模块（相关系数、相对误差、振幅匹配、PSD、高频能量占比）。

所有指标定义统一实现，可直接用于不同系统之间的横向对比。
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


def correlation(pred, ref) -> float:
    """皮尔逊相关系数 R（主指标）。"""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pred.size < 2 or np.std(pred) == 0 or np.std(ref) == 0:
        return float("nan")
    return float(np.corrcoef(pred, ref)[0, 1])


def mean_relative_error(pred, ref, eps: float = 1e-10) -> float:
    """平均相对误差（归一化累计误差）。

    该指标在参考解过零处会出现数值尖峰，
    因此同时给出 R 与振幅匹配度作为互补指标。
    """
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return float(np.mean(np.abs((pred - ref) / (np.abs(ref) + eps))))


def amplitude_match(pred, ref) -> float:
    """振幅匹配度 [%]：std(预测) / std(参考) x 100。"""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return float(np.std(pred) / (np.std(ref) + 1e-10) * 100.0)


def full_scale(ref) -> float:
    """满量程（peak-to-peak）[单位同参考解]。"""
    ref = np.asarray(ref, dtype=float)
    return float(ref.max() - ref.min())


def peak_error_fs_percent(pred, ref) -> float:
    """峰值误差 [% F.S.]：max|预测-参考| / 满量程 x 100。

    满量程归一化可避免参考解过零处相对误差的数值尖峰，
    是与"峰值误差 < 0.4% F.S."这类工程指标对齐的口径。
    """
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    fs = full_scale(ref)
    if fs <= 0:
        return float("nan")
    return float(np.max(np.abs(pred - ref)) / fs * 100.0)


def relative_error_fs_percent(pred, ref) -> float:
    """归一化相对误差 [% F.S.]：mean|预测-参考| / 满量程 x 100。"""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    fs = full_scale(ref)
    if fs <= 0:
        return float("nan")
    return float(np.mean(np.abs(pred - ref)) / fs * 100.0)


def relative_error_above_threshold(pred, ref, threshold: float = 0.1) -> float:
    """有效信号段相对误差 [%]：仅在 |参考| > threshold x 满量程 的点上统计。"""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    fs = full_scale(ref)
    mask = np.abs(ref) > threshold * fs
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - ref[mask]) / np.abs(ref[mask])) * 100.0)


def equivalent_phase_shift(pred, ref, fs: float, freq: float) -> float:
    """等效相位偏移 [rad]：互相关时延换算到关注频率上的相位差。

    正值表示预测波形相对参考解滞后，负值表示超前。
    """
    from scipy.signal import correlate

    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pred.size < 3:
        return float("nan")
    corr = correlate(pred - pred.mean(), ref - ref.mean(), mode="full")
    lag = int(np.argmax(corr) - (pred.size - 1))
    dt = lag / float(fs)
    return float(2 * np.pi * freq * dt)


def summarize(pred, ref) -> Dict[str, float]:
    """汇总常用误差指标（同时给出严格口径与满量程归一化口径）。"""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return {
        "R": correlation(pred, ref),
        "MSE": float(np.mean((pred - ref) ** 2)),
        "relative_error": mean_relative_error(pred, ref),
        "amplitude_match": amplitude_match(pred, ref),
        "max_abs_error": float(np.max(np.abs(pred - ref))),
        "peak_error_fs_percent": peak_error_fs_percent(pred, ref),
        "relative_error_fs_percent": relative_error_fs_percent(pred, ref),
        "relative_error_above_10pct_fs": relative_error_above_threshold(pred, ref, 0.1),
        "full_scale": full_scale(ref),
    }


def welch_psd(x, fs: float, nperseg: int = 131072, noverlap: Optional[int] = None):
    """Welch 功率谱密度估计。

    默认采用：段长 131072 点、50% 重叠、汉宁窗、线性去趋势、
    scaling='density'（单位 rad^2/Hz）。
    """
    x = np.asarray(x, dtype=float)
    if noverlap is None:
        noverlap = nperseg // 2
    return welch(
        x,
        fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend="linear",
        scaling="density",
    )


def hf_energy_ratio(
    x,
    fs: float,
    f_cutoff: float = 0.02,
    nperseg: int = 131072,
) -> Dict[str, float]:
    """计算 PSD 高频段（> f_cutoff）能量占比。

    积分方法为梯形法（``scipy.integrate.trapezoid``），
    Welch 参数完全显式给出，便于复现与横向对比。
    """
    f, psd = welch_psd(x, fs, nperseg=nperseg)
    total_energy = trapezoid(psd, f)
    hf_mask = f > f_cutoff
    hf_energy = trapezoid(psd[hf_mask], f[hf_mask])
    ratio = hf_energy / total_energy * 100.0 if total_energy else float("nan")
    return {
        "total_energy": float(total_energy),
        "hf_energy": float(hf_energy),
        "hf_ratio_percent": float(ratio),
        "f_cutoff": float(f_cutoff),
        "nperseg": int(nperseg),
        "df": float(f[1] - f[0]),
        "f_max": float(f[-1]),
        "n_segments": int(np.ceil((len(x) - nperseg // 2) / (nperseg - nperseg // 2))),
    }


def benchmark_scaling(model, system, device, scales: Optional[Sequence[int]] = None) -> list:
    """多尺度计算效率基准。

    对比神经网络批量推理与数值逐步积分在 10^4 ~ 10^7 点规模下的
    耗时、加速比、单点延迟与显存占用。
    """
    import torch
    from scipy.integrate import odeint

    scales = list(scales or [10_000, 100_000, 1_000_000, 10_000_000])
    results = []

    for n_points in scales:
        t_test = np.linspace(0, n_points / system.fs, n_points)
        forcing_test = 1e-12 * np.random.randn(n_points)
        t_norm = torch.tensor(t_test / system.t_scale, dtype=torch.float32, device=device).reshape(-1, 1)

        start = time.time()
        with torch.no_grad():
            batch_size = 65536 if n_points >= 100_000 else 2048
            for i in range(0, len(t_norm), batch_size):
                model(t_norm[i : i + batch_size])
        pinn_time = time.time() - start

        if n_points <= 1_000_000:

            def equations(state, t_val):
                x, omega = state
                f_val = np.interp(t_val, t_test, forcing_test)
                return [omega, (f_val - system.damping * omega - system.stiffness * x) / system.mass]

            start = time.time()
            odeint(equations, [0, 0], t_test)
            rk4_time = time.time() - start
        else:
            rk4_time = float("inf")

        results.append(
            {
                "n_points": n_points,
                "pinn_time_s": pinn_time,
                "rk4_time_s": rk4_time,
                "speedup": (rk4_time / pinn_time) if pinn_time else float("nan"),
            }
        )
        print(
            f"  {n_points:>10,} 点 | PINN {pinn_time:8.3f} s | "
            f"RK4 {rk4_time:8.3f} s | 加速比 {results[-1]['speedup']:.1f}x"
        )
    return results
