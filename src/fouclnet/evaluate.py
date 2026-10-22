"""评估流程：同分布离散验证、域外外推测试与严格隔离连续段验证。

三类评估分别回答不同问题：

* 同分布离散验证 —— 训练域内随机抽取但未参与训练的离散时间点
  （衡量“插值式重建”精度，也是早停监控口径）；
* 严格隔离连续段 —— 在训练阶段被**整段屏蔽**的时间区间
  （衡量方法在未观测连续时段的表现，预期显著下降）；
* 域外外推 —— 训练域之后的时间段（衡量时序外推能力，预期显著下降）。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .config import Config
from .metrics import summarize
from .train import predict_series


def predict(model, t: np.ndarray, cfg: Config, device: Optional[torch.device] = None) -> np.ndarray:
    """对给定时间轴做批量推理，返回角位移 [rad]。"""
    if device is None:
        from .utils import get_device

        device = get_device(cfg.train.device)
    t_norm = torch.tensor(
        t / cfg.system.t_scale, dtype=torch.float32, device=device
    ).reshape(-1, 1)
    return predict_series(model, t_norm, cfg.train.batch_size)


def evaluate_indices(
    model,
    t: np.ndarray,
    reference: np.ndarray,
    indices: np.ndarray,
    cfg: Config,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """在给定索引集合上评估并返回指标字典。"""
    t_sel = t[indices]
    ref = reference[indices]
    pred = predict(model, t_sel, cfg, device)
    return summarize(pred, ref)


def evaluate_in_distribution(model, t, reference, split, cfg, device=None) -> Dict[str, float]:
    """同分布离散验证。"""
    print("\n========== 同分布离散验证 ==========")
    print(f"验证点数: {split.val_idx.size:,}，时间范围 {t[split.val_idx].min():.1f}–{t[split.val_idx].max():.1f} s")
    print("说明：这些离散点在训练阶段未参与损失计算与参数更新，仅用于早停监控与精度评估。")
    metrics = evaluate_indices(model, t, reference, split.val_idx, cfg, device)
    _print_metrics(metrics)
    return metrics


def evaluate_extrapolation(model, t, reference, split, cfg, device=None) -> Dict[str, float]:
    """域外外推测试。"""
    print("\n========== 域外外推测试 ==========")
    print(f"测试点数: {split.test_idx.size:,}，时间范围 {t[split.test_idx].min():.1f}–{t[split.test_idx].max():.1f} s")
    print("说明：该时段在训练/验证阶段完全未参与。")
    metrics = evaluate_indices(model, t, reference, split.test_idx, cfg, device)
    _print_metrics(metrics)
    return metrics


def evaluate_isolated_segment(model, t, reference, split, cfg, device=None) -> Dict[str, float]:
    """严格隔离连续段验证。"""
    if not split.cont_val_idx.size:
        print("\n[跳过] 当前划分未包含严格隔离连续段（默认模式）。")
        return {}
    print("\n========== 严格隔离连续段验证 ==========")
    print(f"段点数: {split.cont_val_idx.size:,}，时间范围 "
          f"{t[split.cont_val_idx].min():.1f}–{t[split.cont_val_idx].max():.1f} s")
    metrics = evaluate_indices(model, t, reference, split.cont_val_idx, cfg, device)
    _print_metrics(metrics)
    return metrics


def _print_metrics(metrics: Dict[str, float]) -> None:
    """打印评估指标。

    同时给出两种相对误差口径：

    * 严格口径 ``relative_error``：``mean|pred-ref| / (|ref|+eps)``，
      在参考解过零处会出现数值尖峰，适合看"相对量级"；
    * 满量程口径 ``relative_error_fs_percent`` 与 ``peak_error_fs_percent``：
      以参考解峰峰值为基准，适合与工程指标（% F.S.）直接对照。
    """
    if not metrics:
        return
    print(f"  相关系数 R: {metrics['R']:.6f}")
    print(f"  MSE: {metrics['MSE']:.4e}")
    print(f"  平均相对误差（严格口径）: {metrics['relative_error']:.2%}")
    print(f"  振幅匹配度: {metrics['amplitude_match']:.1f}%")
    if "peak_error_fs_percent" in metrics:
        print(f"  峰值误差: {metrics['peak_error_fs_percent']:.3f}% F.S.")
        print(f"  归一化相对误差: {metrics['relative_error_fs_percent']:.3f}% F.S.")
        print(f"  有效信号段相对误差(|ref|>10% F.S.): {metrics['relative_error_above_10pct_fs']:.2f}%")
