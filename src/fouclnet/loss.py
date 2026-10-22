"""损失函数：硬振幅约束混合损失（HAC）与标准物理约束损失。

总损失::

    L = w_data * L_data + w_corr * L_HAC + w_freq * L_freq + w_pde * L_pde + w_ic * L_ic

其中

* L_data  = MSE + 0.1 * MAE + 0.01 * 相对误差；
* L_HAC   = 波形相似项（Z-score 标准化后的 MSE）+ 振幅匹配项（相对标准差误差）；
* L_freq  = 对数幅度谱 MSE + 2 倍相位失配项 1 - cos(dphi)；
* L_pde   = 归一化物理残差平方均值，权重按课程学习三阶段退火至 1e-8 量级；
* L_ic    = 初始条件约束（默认关闭，权重为 0）。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .config import SystemConfig, TrainConfig


class HybridLoss:
    """FouCLNet 的混合损失（数据 + HAC + 频域 + 弱物理残差 + 初始条件）。

    实现细节
    --------
    * 物理残差权重按 ``pde_start_epoch`` / ``pde_full_epoch`` 线性退火到 ``w_pde_max``；
    * 残差按样本归一化（除以 |F / M|），避免激励量级差异淹没残差信号；
    * 频域项每 100 轮计算一次，且要求批内样本数不少于 4096；
    * 频域项关注系统固有频率附近的频带（默认 ``[0.7 f0, 1.4 f0]``，可配置）。
    """

    def __init__(self, system: SystemConfig, train: TrainConfig):
        self.system = system
        self.train = train

    # ------------------------------------------------------------------
    def pde_weight(self, epoch: int) -> float:
        """按课程学习策略返回当前轮次的物理残差权重。"""
        if epoch < self.train.pde_start_epoch:
            return 0.0
        if epoch < self.train.pde_full_epoch:
            progress = (epoch - self.train.pde_start_epoch) / (
                self.train.pde_full_epoch - self.train.pde_start_epoch
            )
            return self.train.w_pde_max * progress
        return self.train.w_pde_max

    # ------------------------------------------------------------------
    def freq_band(self) -> Tuple[float, float]:
        """返回频域项关注的频带 [Hz]。"""
        low = self.train.freq_band_min
        high = self.train.freq_band_max
        if low is None:
            low = 0.7 * self.system.f0
        if high is None:
            high = 1.4 * self.system.f0
        return float(low), float(high)

    # ------------------------------------------------------------------
    def compute(
        self,
        model,
        t_norm: torch.Tensor,
        forcing: torch.Tensor,
        x_target: torch.Tensor,
        epoch: int,
    ) -> Tuple[Dict[str, object], torch.Tensor]:
        """计算各项损失。

        Parameters
        ----------
        model:
            待训练的模型（需返回 ``(x, dx_dt, d2x_dt2, x_norm)``）。
        t_norm:
            归一化时间批，形状 ``(B, 1)``。
        forcing:
            广义激励力批，形状 ``(B, 1)``。
        x_target:
            数值参考解批，形状 ``(B, 1)``。
        epoch:
            当前训练轮次（用于课程学习权重与频域项调度）。

        Returns
        -------
        tuple
            ``(losses, x)``，其中 ``losses`` 含各分项与 ``total``。
        """
        x, dx_dt, d2x_dt2, _ = model(t_norm)
        losses: Dict[str, object] = {}

        # ---------------- 数据拟合项 ----------------
        if x_target is not None:
            mse = torch.mean((x - x_target) ** 2)
            mae = torch.mean(torch.abs(x - x_target))
            rel_error = torch.mean(torch.abs((x - x_target) / (torch.abs(x_target) + 1e-8)))
            losses["data"] = mse + 0.1 * mae + 0.01 * rel_error
        else:
            losses["data"] = torch.tensor(0.0, device=t_norm.device)

        # ---------------- 弱物理残差项 ----------------
        w_pde = self.pde_weight(epoch)
        residual = (
            self.system.mass * d2x_dt2
            + self.system.damping * dx_dt
            + self.system.stiffness * x
            - forcing
        )
        # 按激励力量级归一化，使残差项无量纲，且在不同系统之间具有一致的权重含义
        residual_norm = residual / (torch.abs(forcing) + 1e-10)
        losses["pde"] = torch.mean(residual_norm ** 2)
        losses["w_pde"] = w_pde

        # ---------------- 硬振幅约束（HAC）项 ----------------
        if x_target is not None and self.train.w_corr > 0:
            pred_norm = (x - x.mean()) / (x.std() + 1e-10)
            target_norm = (x_target - x_target.mean()) / (x_target.std() + 1e-10)
            norm_mse = torch.mean((pred_norm - target_norm) ** 2)
            amp_error = torch.abs(x.std() - x_target.std()) / (x_target.std() + 1e-10)
            losses["corr"] = norm_mse + amp_error

            pred_centered = x - torch.mean(x)
            target_centered = x_target - torch.mean(x_target)
            corr_num = torch.sum(pred_centered * target_centered)
            corr_den = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2)) + 1e-10
            corr = corr_num / corr_den
        else:
            losses["corr"] = torch.tensor(0.0, device=t_norm.device)
            corr = torch.tensor(0.0)

        # ---------------- 频域匹配项 ----------------
        losses["freq"] = self._frequency_loss(x, x_target, epoch)

        # ---------------- 初始条件项 ----------------
        if self.train.w_ic > 0:
            t0 = torch.tensor([[0.0]], device=t_norm.device, dtype=t_norm.dtype)
            x0, v0, _, _ = model(t0)
            losses["ic"] = x0 ** 2 + (v0 - self.system.v0) ** 2
        else:
            losses["ic"] = torch.tensor(0.0, device=t_norm.device)

        w_data = self.train.w_data
        w_corr = self.train.w_corr
        w_freq = self.train.w_freq if epoch > 500 else 0.0
        w_ic = self.train.w_ic

        total = (
            w_data * losses["data"]
            + w_corr * losses["corr"]
            + w_freq * losses["freq"]
            + w_pde * losses["pde"]
            + w_ic * losses["ic"]
        )

        losses["total"] = total
        losses["corr_value"] = corr.item()
        return losses, x

    # ------------------------------------------------------------------
    def _frequency_loss(self, x, x_target, epoch: int) -> torch.Tensor:
        """频域匹配项：对数幅度谱误差 + 相位误差（每 100 轮计算一次）。"""
        zero = torch.tensor(0.0, device=x.device)
        if x_target is None or epoch <= 500:
            return zero
        if epoch % 100 != 0 and epoch != self.train.max_epochs - 1:
            return zero
        if len(x) < 4096:
            return zero

        mid = len(x) // 2
        x_seg = x[mid - 2048 : mid + 2048]
        target_seg = x_target[mid - 2048 : mid + 2048]

        x_detrend = x_seg - x_seg.mean()
        target_detrend = target_seg - target_seg.mean()

        fft_pred = torch.fft.rfft(x_detrend)
        fft_target = torch.fft.rfft(target_detrend)
        freqs = torch.fft.rfftfreq(len(x_seg), d=1.0 / self.system.fs).to(x.device)

        # 关注系统固有频率附近频带
        low, high = self.freq_band()
        mask = (freqs >= low) & (freqs <= high)
        if not bool(mask.any()) or int(mask.sum()) < 3:
            return zero

        mag_pred = torch.log(torch.abs(fft_pred[mask]) + 1e-12)
        mag_target = torch.log(torch.abs(fft_target[mask]) + 1e-12)
        mag_loss = torch.mean((mag_pred - mag_target) ** 2)

        phase_pred = torch.angle(fft_pred[mask])
        phase_target = torch.angle(fft_target[mask])
        phase_loss = torch.mean(1 - torch.cos(phase_pred - phase_target))

        return mag_loss + 2.0 * phase_loss


class StandardLoss:
    """标准物理约束损失（数据 MSE + 残差项），用于对照实验。"""

    def __init__(self, system: SystemConfig, w_data: float = 1.0, w_pde: float = 1e-8):
        self.system = system
        self.w_data = w_data
        self.w_pde = w_pde

    def compute(self, model, t_norm, forcing, x_target, epoch: int):
        """计算标准物理约束损失。"""
        x, dx_dt, d2x_dt2, _ = model(t_norm)
        loss_data = torch.mean((x - x_target) ** 2)
        residual = (
            self.system.mass * d2x_dt2
            + self.system.damping * dx_dt
            + self.system.stiffness * x
            - forcing
        )
        loss_pde = torch.mean(residual ** 2)
        total = self.w_data * loss_data + self.w_pde * loss_pde
        return {
            "total": total,
            "data": loss_data.item(),
            "pde": loss_pde.item(),
            "corr": 0.0,
            "amp_match": float(torch.std(x) / (torch.std(x_target) + 1e-10)),
        }, x


def build_loss(kind: str, system: SystemConfig, train: TrainConfig):
    """按名称构造损失函数（``hybrid`` / ``standard``）。"""
    key = kind.strip().lower()
    if key in {"hybrid", "hac", "ours"}:
        return HybridLoss(system, train)
    if key == "standard":
        return StandardLoss(system, w_data=train.w_data, w_pde=train.w_pde_max)
    raise ValueError(f"未知损失类型: {kind!r}（可选: hybrid / standard）")
