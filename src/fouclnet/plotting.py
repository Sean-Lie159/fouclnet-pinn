"""图表绘制实现。

函数与图形的对应关系：

* :func:`plot_training_set_overview`   训练集时域/误差/散点/历程/PSD 综合图（a-f 六个子图）
* :func:`plot_zoomed_comparison`       局部时间窗放大对比（含等效相位偏移量化）
* :func:`plot_in_distribution`         同分布离散验证
* :func:`plot_isolated_segment`        严格隔离连续段
* :func:`plot_extrapolation`           域外外推测试
* :func:`plot_figure4_revised`         隔离段 vs 外推段的失效模式对比
* :func:`plot_ablation_scatter`        消融实验离散验证散点
* :func:`plot_ablation_continuous`     消融实验连续时段对比
* :func:`plot_psd_hf_detail`           高频段能量占比补充图

所有函数使用非交互后端（Agg），并把图片写入指定目录。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import correlate  # noqa: E402

from .config import Config  # noqa: E402
from .metrics import welch_psd  # noqa: E402


def _save(fig, path: str | Path) -> Path:
    """保存图片并返回路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 已保存: {target}")
    return target


def focus_frequency(cfg: Config) -> float:
    """返回关注的系统特征频率 [Hz]（默认取系统固有频率 ``system.f0``）。"""
    freq = cfg.evaluation.focus_freq
    if freq is None:
        freq = cfg.system.f0
    return float(freq)


def plot_training_set_overview(
    cfg: Config,
    t: np.ndarray,
    reference: np.ndarray,
    theta_pred: np.ndarray,
    history: Dict[str, list],
    output_path: str | Path = "results/figures/overview.png",
    split_point: Optional[int] = None,
) -> Path:
    """训练集总览图：时域对比与频域功率谱密度（a–f 六个子图）。"""
    split_point = split_point or int(len(t) * cfg.train.split_ratio)
    t_train = t[:split_point]
    ref = reference[:split_point]
    pred = theta_pred[:split_point]

    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t_train, ref, "b-", label="RK4", alpha=0.8, linewidth=1.2)
    ax1.plot(t_train, pred, "r--", label="FouCLNet", alpha=0.6, linewidth=1.2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (rad)")
    ax1.set_title("(a) Time Domain Comparison (Train Domain)")
    ax1.set_xlim([0, t_train[-1]])
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    r_train = float(np.corrcoef(ref, pred)[0, 1])
    amp_match = float(np.std(pred) / (np.std(ref) + 1e-10) * 100)
    ax1.text(
        0.02, 0.98, f"R = {r_train:.6f}\nAmp Match: {amp_match:.1f}%",
        transform=ax1.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2 = plt.subplot(2, 3, 2)
    rel_error = np.abs((pred - ref) / (np.abs(ref) + 1e-10)) * 100
    ax2.semilogy(t_train, rel_error, "g-", alpha=0.7, linewidth=0.8)
    ax2.axhline(y=float(np.mean(rel_error)), color="r", linestyle="--", label=f"Mean: {np.mean(rel_error):.2f}%")
    ax2.axvline(x=t_train[-1], color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Relative Error (%)")
    ax2.set_title("(b) Prediction Relative Error (Train Domain)")
    ax2.set_xlim([0, t_train[-1]])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(ref, pred, alpha=0.5, s=1.5)
    ax3.plot([ref.min(), ref.max()], [ref.min(), ref.max()], "r--", label="Perfect Match", linewidth=2)
    ax3.set_xlabel("Target (rad)")
    ax3.set_ylabel("Prediction (rad)")
    ax3.set_title(f"(c) Correlation (R = {r_train:.6f})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    ax4.semilogy(history["epoch"], history["total"], "k-", label="Total", linewidth=2)
    ax4.semilogy(history["epoch"], history["data"], "b-", label="Data", alpha=0.7)
    ax4.semilogy(history["epoch"], history["corr"], "r-", label="Corr", alpha=0.7)
    ax4.semilogy(history["epoch"], history["freq"], "g-", label="Freq", alpha=0.7)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.set_title("(d) Training History")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(history["epoch"], history["r_value"], "m-", linewidth=2)
    ax5.axhline(y=0.9, color="r", linestyle="--", label="Target R = 0.9")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Pearson R")
    ax5.set_title("(e) Correlation Evolution")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    f_ref, psd_ref = welch_psd(reference, cfg.system.fs, cfg.evaluation.welch_nperseg)
    f_pred, psd_pred = welch_psd(theta_pred, cfg.system.fs, cfg.evaluation.welch_nperseg)
    ax6.loglog(f_ref, psd_ref, "b-", label="Target", linewidth=2)
    ax6.loglog(f_pred, psd_pred, "r--", label="FouCLNet", linewidth=2)
    ax6.axvline(x=cfg.system.f0, color="purple", linestyle=":", alpha=0.7, label=f"f0 = {cfg.system.f0:.3f} Hz")
    ax6.set_xlabel("Frequency (Hz)")
    ax6.set_ylabel("PSD (rad^2/Hz)")
    ax6.set_title("(f) Power Spectral Density")
    ax6.legend()
    ax6.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    return _save(fig, output_path)


def plot_zoomed_comparison(
    cfg: Config,
    t: np.ndarray,
    reference: np.ndarray,
    theta_pred: np.ndarray,
    output_path: str | Path = "results/figures/zoomed_comparison.png",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Path:
    """局部放大时域对比（含等效相位偏移量化）。"""
    t_start = cfg.evaluation.zoom_start if t_start is None else t_start
    t_end = cfg.evaluation.zoom_end if t_end is None else t_end

    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        print(f"[跳过] 局部放大窗口 {t_start:.0f}-{t_end:.0f} s 超出数据范围 "
              f"[{t[0]:.0f}, {t[-1]:.0f}] s。")
        return None
    t_seg = t[mask]
    ref = reference[mask]
    pred = theta_pred[mask]


    r_local = float(np.corrcoef(ref, pred)[0, 1])
    abs_err = np.abs(pred - ref)
    rel_err = abs_err / (np.abs(ref) + 1e-12) * 100

    # 互相关求时延 -> 等效相位偏移
    corr = correlate(pred - pred.mean(), ref - ref.mean(), mode="full")
    lag = int(np.argmax(corr) - (len(ref) - 1))
    dt = lag / cfg.system.fs
    phase_rad = 2 * np.pi * focus_frequency(cfg) * dt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1, 1]})

    ax = axes[0]
    ax.plot(t_seg, ref, "b-", label="RK4", linewidth=1.5, alpha=0.9)
    ax.plot(t_seg, pred, "r--", label="FouCLNet", linewidth=1.5, alpha=0.8)
    ax.set_title(f"Zoomed Comparison ({t_start:.0f}-{t_end:.0f} s) | R = {r_local:.4f}")
    ax.set_ylabel("Angle (rad)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(t_seg, abs_err * 1e6, "g-", alpha=0.7, linewidth=0.8)
    ax.axhline(y=float(np.mean(abs_err)) * 1e6, color="orange", linestyle="--",
               label=f"Mean |Error| = {np.mean(abs_err) * 1e6:.2f} urad")
    ax.set_ylabel("Abs Error (urad)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t_seg, rel_err, "purple", alpha=0.6, linewidth=0.8)
    ax.axhline(y=float(np.mean(rel_err)), color="r", linestyle="--",
               label=f"Mean RelErr = {np.mean(rel_err):.1f}%")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Error (%)")
    ax.set_ylim([0, min(200, float(np.percentile(rel_err, 99)))])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = _save(fig, output_path)
    print(f"[局部放大 {t_start:.0f}-{t_end:.0f}s] R = {r_local:.4f}，"
          f"平均绝对误差 {np.mean(abs_err):.3e} rad，等效相位偏移 {phase_rad:.3e} rad")
    return path


def plot_psd_hf_detail(
    cfg: Config,
    reference: np.ndarray,
    theta_pred: np.ndarray,
    output_path: str | Path = "results/figures/psd_hf_detail.png",
) -> Path:
    """高频段能量占比补充图。"""
    fs = cfg.system.fs
    f_cut = cfg.evaluation.hf_cutoff
    f_ref, psd_ref = welch_psd(reference, fs, cfg.evaluation.welch_nperseg)
    f_pred, psd_pred = welch_psd(theta_pred, fs, cfg.evaluation.welch_nperseg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.loglog(f_ref, psd_ref, "b-", linewidth=1.5, label="RK4 Reference", alpha=0.9)
    ax.loglog(f_pred, psd_pred, "r--", linewidth=1.5, label="FouCLNet Prediction", alpha=0.9)
    ax.axvline(x=focus_frequency(cfg), color="purple", linestyle=":", linewidth=2,
               label=f"f0 = {focus_frequency(cfg):.3f} Hz")
    ax.axvline(x=f_cut, color="gray", linestyle="--", linewidth=2, label=f"HF cutoff = {f_cut} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (rad^2/Hz)")
    ax.set_title("Power Spectral Density (Full Band)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    mask = f_ref > f_cut
    ax.semilogy(f_ref[mask], psd_ref[mask], "b-", linewidth=1.5, label="RK4")
    ax.semilogy(f_pred[mask], psd_pred[mask], "r--", linewidth=1.5, label="FouCLNet")
    ax.axvline(x=f_cut, color="gray", linestyle="--", linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (rad^2/Hz)")
    ax.set_title(f"High-Frequency Detail (>{f_cut} Hz)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    return _save(fig, output_path)


def plot_in_distribution(
    cfg: Config,
    t: np.ndarray,
    reference: np.ndarray,
    theta_pred: np.ndarray,
    indices: np.ndarray,
    output_path: str | Path = "results/figures/in_distribution.png",
) -> Path:
    """同分布离散验证（训练域内未参与训练的离散点）。"""
    t_sel = t[indices]
    ref = reference[indices]
    pred = theta_pred[indices]
    r_val = float(np.corrcoef(ref, pred)[0, 1])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    skip = max(1, len(indices) // 800)
    axes[0].scatter(t_sel[::skip], ref[::skip], c="blue", s=25, alpha=0.7,
                     edgecolors="navy", linewidth=0.5, label="RK4 (unseen validation points)")
    axes[0].scatter(t_sel[::skip], pred[::skip], c="red", s=20, alpha=0.7, marker="x",
                     linewidth=1.2, label="FouCLNet (prediction)")
    axes[0].text(0.02, 0.95, f"Unseen points: {len(indices):,}\nR = {r_val:.6f}",
                 transform=axes[0].transAxes, fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].set_title("(a) In-Distribution Validation (Unseen Discrete Points on Train Domain)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(t_sel, np.abs(pred - ref), "g-", alpha=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Absolute Error (rad)")
    axes[1].set_title("In-Distribution Error")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return _save(fig, output_path)


def plot_isolated_segment(
    cfg: Config,
    t: np.ndarray,
    reference: np.ndarray,
    theta_pred: np.ndarray,
    indices: np.ndarray,
    output_path: str | Path = "results/figures/isolated_segment.png",
) -> Path:
    """严格隔离连续段（训练阶段被完全屏蔽）。"""
    t_seg = t[indices]
    ref = reference[indices]
    pred = theta_pred[indices]
    if t_seg.size == 0:
        print("[跳过] 严格隔离段为空（数据范围不足或未启用隔离划分）。")
        return None
    r_cont = float(np.corrcoef(ref, pred)[0, 1])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(t_seg, ref, "b-", label="RK4", linewidth=1.2, alpha=0.9)
    axes[0].plot(t_seg, pred, "r--", label="FouCLNet (Isolated)", linewidth=1.2, alpha=0.8)
    axes[0].set_title(f"(b) Isolated Segment ({t_seg[0]:.0f}-{t_seg[-1]:.0f} s), R = {r_cont:.4f}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(t_seg, np.abs(pred - ref), "g-", alpha=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Absolute Error (rad)")
    axes[1].set_title("Absolute Error (Isolated Segment)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return _save(fig, output_path)


def plot_extrapolation(
    cfg: Config,
    t: np.ndarray,
    reference: np.ndarray,
    theta_pred: np.ndarray,
    indices: np.ndarray,
    output_path: str | Path = "results/figures/extrapolation.png",
) -> Path:
    """训练域外的外推测试。"""
    t_seg = t[indices]
    ref = reference[indices]
    pred = theta_pred[indices]
    r_test = float(np.corrcoef(ref, pred)[0, 1])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(t_seg, ref, "b-", label="RK4", alpha=0.7, linewidth=1)
    axes[0].plot(t_seg, pred, "r--", label="FouCLNet", alpha=0.7, linewidth=1)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].set_title(f"(c) Extrapolation Test (Out-of-Domain), R = {r_test:.4f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(t_seg, np.abs(pred - ref), "g-", alpha=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Absolute Error (rad)")
    axes[1].set_title("Extrapolation Error")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return _save(fig, output_path)


def plot_figure4_revised(
    cfg: Config,
    t: np.ndarray,
    reference: np.ndarray,
    iso_pred: np.ndarray,
    extrap_pred: np.ndarray,
    split,
    output_path: str | Path = "results/figures/isolated_vs_extrapolation.png",
) -> Path:
    """严格隔离段（上）与域外外推段（下）的失效模式对比。"""
    iso_idx = split.cont_val_idx
    test_idx = split.test_idx

    t_iso = t[iso_idx]
    ref_iso = reference[iso_idx]
    if t_iso.size == 0:
        print("[跳过] 严格隔离段为空，无法生成图 4 汇总版。")
        return None
    r_iso = float(np.corrcoef(ref_iso, iso_pred[iso_idx])[0, 1])

    t_ext = t[test_idx]
    ref_ext = reference[test_idx]
    r_ext = float(np.corrcoef(ref_ext, extrap_pred[test_idx])[0, 1])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(t_iso, ref_iso, "b-", label="RK4", linewidth=1.2, alpha=0.9)
    ax.plot(t_iso, iso_pred[iso_idx], "r--", label="FouCLNet (Isolated)", linewidth=1.2, alpha=0.8)
    ax.set_title(f"(a) Isolated Segment ({t_iso[0]:.0f}-{t_iso[-1]:.0f} s), R = {r_iso:.4f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(t_iso, np.abs(iso_pred[iso_idx] - ref_iso), "g-", alpha=0.7, linewidth=0.8)
    ax.set_title("Absolute Error (Isolated Segment)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Error| (rad)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_ext, ref_ext, "b-", label="RK4", linewidth=1.2, alpha=0.9)
    ax.plot(t_ext, extrap_pred[test_idx], "r--", label="FouCLNet (Main)", linewidth=1.2, alpha=0.8)
    ax.set_title(f"(b) Extrapolation Test ({t_ext[0]:.0f}-{t_ext[-1]:.0f} s), R = {r_ext:.4f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(t_ext, np.abs(extrap_pred[test_idx] - ref_ext), "g-", alpha=0.7, linewidth=0.8)
    ax.set_title("Absolute Error (Extrapolation)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Error| (rad)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = _save(fig, output_path)
    print(f"[图4] 隔离段 R = {r_iso:.4f}，域外外推 R = {r_ext:.4f}")
    return path


def plot_ablation_scatter(
    cfg: Config,
    reference: np.ndarray,
    predictions: Dict[str, np.ndarray],
    val_indices: np.ndarray,
    output_path: str | Path = "results/figures/ablation_scatter.png",
) -> Path:
    """消融实验在离散验证集上的散点图。"""
    titles = {"A": "Standard MLP", "B": "FFN without HAC", "E": "FFN+HAC, no PDE",
              "D": "SIREN", "C": "Ours (FouCLNet)"}
    ref = reference[val_indices]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for idx, (key, title) in enumerate(titles.items()):
        if key not in predictions:
            continue
        row, col = positions[idx]
        ax = axes[row, col]
        pred = predictions[key]
        ax.scatter(ref[::50], pred[::50], alpha=0.4, s=2, c="purple")
        lim_min = min(ref.min(), pred.min())
        lim_max = max(ref.max(), pred.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5, label="y = x")
        ax.set_title(f"{title}\nR = {np.corrcoef(ref, pred)[0, 1]:.3f}")
        ax.set_xlabel("RK4 (rad)")
        ax.set_ylabel("Pred (rad)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[1, 2].axis("off")
    plt.tight_layout()
    return _save(fig, output_path)


def plot_ablation_continuous(
    cfg: Config,
    t: np.ndarray,
    reference: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: str | Path = "results/figures/ablation_continuous.png",
) -> Path:
    """消融实验在训练域内连续时段的对比。"""
    titles = {"A": "Standard MLP", "B": "FFN without HAC", "E": "FFN+HAC, no PDE",
              "D": "SIREN", "C": "Ours (FouCLNet)"}
    mask = (t >= cfg.evaluation.vis_start) & (t <= cfg.evaluation.vis_end)
    if not mask.any():
        print(f"[跳过] 连续时段窗口 {cfg.evaluation.vis_start:.0f}-{cfg.evaluation.vis_end:.0f} s "
              f"超出数据范围 [{t[0]:.0f}, {t[-1]:.0f}] s。")
        return None
    t_seg = t[mask]
    ref = reference[mask]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for idx, (key, title) in enumerate(titles.items()):
        if key not in predictions:
            continue
        row, col = positions[idx]
        ax = axes[row, col]
        pred = predictions[key]
        ax.plot(t_seg, ref, "b-", label="RK4", alpha=0.8, linewidth=1.2)
        ax.plot(t_seg, pred, "r--", label="Pred", alpha=0.7, linewidth=1.2)
        r_seg = float(np.corrcoef(ref, pred)[0, 1])
        amp = float(np.std(pred) / (np.std(ref) + 1e-10) * 100)
        ax.set_title(f"{title}\nR = {r_seg:.3f}, Amp = {amp:.1f}%")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[1, 2].axis("off")
    plt.tight_layout()
    return _save(fig, output_path)
