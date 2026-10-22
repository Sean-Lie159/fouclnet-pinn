"""训练主循环（FouCLNet）。

训练策略
--------------------
* 优化器 AdamW（lr=1e-4, weight_decay=1e-5）+ 余弦退火学习率（T_max=3000, eta_min=1e-6）；
* 每轮按"重点时间窗加权"从训练集中有放回采样 2048 个点；
* 每 500 轮在离散验证集上评估一次，以平均相对误差为早停判据（patience=30, min_delta=1e-7）；
* 训练后期若验证相关系数超过 0.999，则按 1.2 倍自适应提升 PDE 权重上限（硬上限 1e-3）；
* 训练完成后加载最佳权重（``best_pinn_model*.pth``）并返回。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config
from .data import SeriesData, SplitResult
from .loss import HybridLoss
from .metrics import correlation
from .model import FouCLNet
from .utils import ensure_dir, get_device, set_seed


def _sample_weights(t: np.ndarray, train_idx: np.ndarray, cfg: Config) -> np.ndarray:
    """构造训练采样的重要性权重（重点时间窗加权）。"""
    t_train = t[train_idx]
    weights = np.ones_like(t_train, dtype=np.float32)
    focus_mask = (t_train >= cfg.train.focus_start) & (t_train <= cfg.train.focus_end)
    weights[focus_mask] = cfg.train.focus_weight
    weights /= weights.sum()
    return weights


def predict_series(model, t_norm: torch.Tensor, batch_size: int) -> np.ndarray:
    """分块推理，返回物理量纲下的位移序列。"""
    outputs = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(0, len(t_norm), batch_size):
            batch = t_norm[i : i + batch_size]
            out = model(batch)
            x = out[0] if isinstance(out, (tuple, list)) else out
            outputs.append(x.cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(outputs).flatten()


def train_model(
    cfg: Config,
    data: SeriesData,
    split: SplitResult,
    reference: np.ndarray,
    model: Optional[nn.Module] = None,
    loss_fn: Optional[object] = None,
    model_save_path: Optional[str | Path] = "checkpoints/model_full.pth",
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, list]]:
    """训练 FouCLNet。

    Parameters
    ----------
    cfg:
        完整配置。
    data:
        输入时序数据（含广义激励力 ``forcing``）。
    split:
        数据划分结果。
    reference:
        数值参考解（作为监督标签）。

    model:
        预置模型；``None`` 时按配置新建 :class:`FouCLNet`。
    loss_fn:
        预置损失；``None`` 时使用 :class:`HybridLoss`。
    model_save_path:
        最佳权重保存路径；``None`` 表示不落盘（仅在内存中返回最佳权重）。
    verbose:
        是否打印训练日志。

    Returns
    -------
    tuple
        ``(model, history)``；``history`` 含逐次验证的损失与指标。
    """
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    system = cfg.system
    train_cfg = cfg.train

    if model is None:
        model = FouCLNet(system, train_cfg).to(device)
    if loss_fn is None:
        loss_fn = HybridLoss(system, train_cfg)

    save_path = Path(model_save_path) if model_save_path else None
    if save_path is not None:
        ensure_dir(save_path.parent)

    t = data.t
    forcing = data.forcing

    t_norm = torch.tensor(t / system.t_scale, dtype=torch.float32).reshape(-1, 1).to(device)
    forcing_tensor = torch.tensor(forcing, dtype=torch.float32).reshape(-1, 1).to(device)
    ref_tensor = torch.tensor(reference, dtype=torch.float32).reshape(-1, 1).to(device)

    t_norm_train = t_norm[split.train_idx]
    forcing_train = forcing_tensor[split.train_idx]
    ref_train = ref_tensor[split.train_idx]
    t_norm_val = t_norm[split.val_idx]
    ref_val = ref_tensor[split.val_idx]

    weights = _sample_weights(t, split.train_idx, cfg)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.scheduler_t_max, eta_min=train_cfg.scheduler_eta_min
    )

    history: Dict[str, list] = {
        "epoch": [],
        "total": [],
        "data": [],
        "corr": [],
        "freq": [],
        "pde": [],
        "w_pde": [],
        "val_error": [],
        "r_value": [],
    }

    best_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    if verbose:
        print(f"\n开始训练（最大 {train_cfg.max_epochs} 轮，设备 {device}）")
        print(f"模型保存路径: {save_path if save_path else '(不落盘)'}")
        print("格式: [Epoch] Total | Data | Corr | Freq | PDE | w_PDE | R_val | ValErr | Amp")

    model.train()
    for epoch in range(train_cfg.max_epochs):
        optimizer.zero_grad()

        if len(t_norm_train) > train_cfg.batch_size:
            idx = torch.multinomial(weights_tensor, train_cfg.batch_size, replacement=True)
            t_batch = t_norm_train[idx]
            forcing_batch = forcing_train[idx]
            ref_batch = ref_train[idx]
        else:
            t_batch = t_norm_train
            forcing_batch = forcing_train
            ref_batch = ref_train

        losses, _ = loss_fn.compute(model, t_batch, forcing_batch, ref_batch, epoch)

        # 自适应提升 PDE 权重（训练充分收敛后加强物理正则化）
        if epoch > train_cfg.pde_full_epoch and epoch % train_cfg.pde_adaptive_interval == 0:
            current_r = history["r_value"][-1] if history["r_value"] else 0.0
            if current_r > train_cfg.pde_adaptive_r_threshold:
                old_max = train_cfg.w_pde_max
                train_cfg.w_pde_max = min(old_max * train_cfg.pde_adaptive_multiplier, 1e-3)
                if train_cfg.w_pde_max != old_max:
                    print(f"  [自适应 PDE] R={current_r:.4f}>{train_cfg.pde_adaptive_r_threshold}，"
                          f"w_pde_max: {old_max:.2e} -> {train_cfg.w_pde_max:.2e}")

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch % train_cfg.validation_interval == 0 or epoch == train_cfg.max_epochs - 1:
            pred_val = predict_series(model, t_norm_val, train_cfg.batch_size)
            target_val = ref_val.cpu().numpy().flatten()
            val_error = float(np.mean(np.abs(pred_val - target_val) / (np.abs(target_val) + 1e-10)))
            r_value = correlation(pred_val, target_val)
            amp_match = float(np.std(pred_val) / (np.std(target_val) + 1e-10) * 100)

            history["epoch"].append(epoch)
            history["total"].append(float(losses["total"].item()))
            history["data"].append(float(losses["data"].item()))
            history["corr"].append(float(losses["corr"].item()))
            history["freq"].append(float(losses["freq"].item()) if hasattr(losses["freq"], "item") else 0.0)
            history["pde"].append(float(losses["pde"].item()) if hasattr(losses["pde"], "item") else 0.0)
            history["w_pde"].append(float(losses["w_pde"]))
            history["val_error"].append(val_error)
            history["r_value"].append(r_value)

            if verbose:
                print(
                    f"[{epoch:5d}] {losses['total'].item():.2e} | "
                    f"D:{losses['data'].item():.2e} | "
                    f"C:{losses['corr'].item():.2e} | "
                    f"F:{losses['freq'].item():.2e} | "
                    f"P:{losses['pde'].item():.2e} | "
                    f"wP:{float(losses['w_pde']):.2e} | "
                    f"R_val:{r_value:.4f} | "
                    f"ValErr:{val_error:.2%} | "
                    f"Amp:{amp_match:.1f}%"
                )

            if val_error < best_loss - train_cfg.min_delta:
                best_loss = val_error
                patience_counter = 0
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= train_cfg.patience:
                    print(f"早停触发于 epoch {epoch}（最佳验证相对误差 {best_loss:.2%}）")
                    break

    if save_path is not None and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    print(f"\n训练完成，耗时 {(time.time() - start_time) / 60:.1f} min，"
          f"最佳验证相对误差 {best_loss:.2%}，最佳验证 R {max(history['r_value']):.4f}")
    return model, history
