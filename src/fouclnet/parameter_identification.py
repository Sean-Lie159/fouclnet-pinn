"""物理参数反演（辅助实验）。

利用 Fourier 特征网络端到端的可微性，通过梯度下降反推系统物理参数
``M``、``C``、``K``，用于考察模型与物理模型的一致性潜力
（PINN 反问题思路，参考 Karniadakis et al., 2021）。

说明：该模块为可选的补充性实验，不参与主流程与结论；
原实现保留于历史脚本中，此处做最小改写并去除硬编码路径。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config
from .utils import get_device


def inverse_parameter_identification(
    cfg: Config,
    model,
    data,
    split,
    reference: np.ndarray,
    n_samples: int = 5000,
    n_epochs: int = 2000,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, float], Dict[str, list]]:
    """反演物理参数 ``I``/``gamma``/``kappa``。

    Returns
    -------
    tuple
        ``(参数字典, 训练历史)``。
    """
    device = device or get_device(cfg.train.device)
    system = cfg.system

    sample_size = min(n_samples, split.train_idx.size)
    sample_idx = np.random.choice(split.train_idx, sample_size, replace=False)

    t_sample = torch.tensor(
        data.t[sample_idx] / system.t_scale, dtype=torch.float32, device=device
    ).reshape(-1, 1)
    theta_ref = torch.tensor(
        reference[sample_idx], dtype=torch.float32, device=device
    ).reshape(-1, 1)
    Ns_sample = torch.tensor(
        data.forcing[sample_idx], dtype=torch.float32, device=device
    ).reshape(-1, 1)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 对数参数化保证正值；初始值带 ±10~20% 偏差以检验反演能力
    log_I = nn.Parameter(torch.tensor(np.log(system.mass * 1.2), dtype=torch.float32, device=device))
    log_gamma = nn.Parameter(torch.tensor(np.log(system.damping * 0.8), dtype=torch.float32, device=device))
    log_kappa = nn.Parameter(torch.tensor(np.log(system.stiffness * 1.1), dtype=torch.float32, device=device))

    optimizer = optim.Adam([log_I, log_gamma, log_kappa], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    history: Dict[str, list] = {"loss": [], "I_error": [], "gamma_error": [], "kappa_error": []}

    print("\n========== 物理参数反演 ==========")
    print(f"  初始猜测: I = {torch.exp(log_I).item():.3e}, "
          f"gamma = {torch.exp(log_gamma).item():.3e}, kappa = {torch.exp(log_kappa).item():.3e}")
    print(f"  真实值  : I = {system.mass:.3e}, gamma = {system.damping:.3e}, kappa = {system.stiffness:.3e}")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        I_est = torch.exp(log_I)
        gamma_est = torch.exp(log_gamma)
        kappa_est = torch.exp(log_kappa)

        theta_pred, dtheta_dt, d2theta_dt2, _ = model(t_sample)
        omega0_est = torch.sqrt(kappa_est / I_est)
        zeta_est = gamma_est / (2 * torch.sqrt(I_est * kappa_est))
        residual = (
            d2theta_dt2
            + 2 * zeta_est * omega0_est * dtheta_dt
            + omega0_est ** 2 * theta_pred
            - Ns_sample / I_est
        )
        loss_data = torch.mean((theta_pred - theta_ref) ** 2)
        loss_pde = torch.mean(residual ** 2)
        loss = loss_data + 1e-6 * loss_pde

        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())

        with torch.no_grad():
            I_err = float(torch.abs(I_est - system.mass) / system.mass * 100)
            gamma_err = float(torch.abs(gamma_est - system.damping) / system.damping * 100)
            kappa_err = float(torch.abs(kappa_est - system.stiffness) / system.stiffness * 100)
        history["loss"].append(float(loss.item()))
        history["I_error"].append(I_err)
        history["gamma_error"].append(gamma_err)
        history["kappa_error"].append(kappa_err)

        if epoch % 500 == 0:
            print(f"  epoch {epoch:5d}: loss = {loss.item():.2e}, "
                  f"I 误差 = {I_err:.1f}%, gamma 误差 = {gamma_err:.1f}%, kappa 误差 = {kappa_err:.1f}%")

    for param in model.parameters():
        param.requires_grad = True

    result = {
        "I": float(torch.exp(log_I).item()),
        "gamma": float(torch.exp(log_gamma).item()),
        "kappa": float(torch.exp(log_kappa).item()),
        "I_error_percent": history["I_error"][-1],
        "gamma_error_percent": history["gamma_error"][-1],
        "kappa_error_percent": history["kappa_error"][-1],
    }
    print(f"反演结果: I = {result['I']:.3e} ({result['I_error_percent']:.1f}%), "
          f"gamma = {result['gamma']:.3e} ({result['gamma_error_percent']:.1f}%), "
          f"kappa = {result['kappa']:.3e} ({result['kappa_error_percent']:.1f}%)")
    return result, history
