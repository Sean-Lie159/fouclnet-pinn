"""神经网络结构：傅里叶特征映射、FouCLNet 主干与消融基线。

结构::

    t_norm (1)
      -> FourierFeature: gamma(t) = [sin(2*pi*B t), cos(2*pi*B t)]
      -> Linear(1024 -> 512) -> Tanh -> Linear(512 -> 512) -> Tanh
      -> Linear(512 -> 256) -> Tanh -> Linear(256 -> 128) -> Tanh
      -> Linear(128 -> 1)  (无量纲位移)

傅里叶频率 B 按对数间距均匀分布于 ``freq_min`` ~ ``freq_max``，
使网络的可学习频率范围精确锁定在系统特征频率附近（弱阻尼、高 Q 值系统尤其有效）。

兼容性说明
----------
模块内参数名固定为 ``fourier.B``、``net.<i>.weight`` 等，
因此不同系统、不同数据上训练得到的权重可互相替换加载。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import SystemConfig, TrainConfig


class FourierFeature(nn.Module):
    """对数间距傅里叶特征映射。

    Parameters
    ----------
    input_dim:
        输入维度（时间轴为 1 维）。
    mapping_size:
        频率个数。
    f_min, f_max:
        频率范围 [Hz]。
    """

    def __init__(self, input_dim: int = 1, mapping_size: int = 512, f_min: float = 0.003, f_max: float = 0.02):
        super().__init__()
        # 对数间距：保证特征频率附近有足够的频率采样密度
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), mapping_size)
        self.B = nn.Parameter(
            torch.tensor(freqs, dtype=torch.float32).reshape(1, -1),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把输入坐标映射为 [sin, cos] 拼接的高维特征。"""
        scale = torch.tensor(2 * np.pi, dtype=self.B.dtype, device=self.B.device)
        x_proj = scale * (x.to(self.B.dtype) @ self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FouCLNet(nn.Module):
    """FouCLNet 主干：傅里叶特征映射 + 多层全连接网络。

    前向返回 ``(x, dx_dt, d2x_dt2, x_norm)``：
    前三项为物理量纲下的位移及其一、二阶时间导数（用于物理残差项），
    最后一项为无量纲输出（用于损失函数与评估）。

    在 ``torch.no_grad()`` 上下文（推理模式）下只做一次前向，
    不构造计算图，以降低大批量推理的开销。
    """

    def __init__(self, system: SystemConfig, train: TrainConfig):
        super().__init__()
        self.system = system
        self.train_cfg = train

        self.fourier = FourierFeature(1, train.fourier_dim, train.freq_min, train.freq_max)

        layers = [2 * train.fourier_dim] + list(train.hidden_layers) + [1]
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

        # Xavier 正态初始化
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t_norm: torch.Tensor):
        """前向计算。

        Parameters
        ----------
        t_norm:
            归一化时间 ``t / t_scale``，形状 ``(N, 1)``。

        Returns
        -------
        tuple
            ``(x, dx_dt, d2x_dt2, x_norm)``；
            推理模式下后三项为 ``None``（除 ``x_norm`` 外）。
        """
        if not torch.is_grad_enabled():
            features = self.fourier(t_norm)
            x_norm = self.net(features)
            x = x_norm * self.system.x_scale
            return x, None, None, x_norm

        t_norm = t_norm.requires_grad_(True)
        features = self.fourier(t_norm)
        x_norm = self.net(features)
        x = x_norm * self.system.x_scale

        # 自动微分求一阶、二阶导数，并还原到物理时间尺度
        dx_dt = torch.autograd.grad(
            x,
            t_norm,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0] / self.system.t_scale

        d2x_dt2 = torch.autograd.grad(
            dx_dt,
            t_norm,
            grad_outputs=torch.ones_like(dx_dt),
            create_graph=True,
            retain_graph=True,
        )[0] / self.system.t_scale

        return x, dx_dt, d2x_dt2, x_norm


class NaiveMLP(nn.Module):
    """消融基线 A：标准多层感知机（无傅里叶特征、无物理约束）。

    用于验证谱偏置现象：普通 MLP 难以捕捉系统的低频共振成分。
    """

    def __init__(self, system: SystemConfig, train: TrainConfig):
        super().__init__()
        self.system = system
        self.train_cfg = train

        layers = [1] + list(train.hidden_layers) + [1]
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t_norm: torch.Tensor):
        """前向计算，返回格式与 :class:`FouCLNet` 一致。"""
        if not torch.is_grad_enabled():
            x_norm = self.net(t_norm)
            return x_norm * self.system.x_scale, None, None, x_norm

        t_norm = t_norm.requires_grad_(True)
        x_norm = self.net(t_norm)
        x = x_norm * self.system.x_scale
        dx_dt = torch.autograd.grad(
            x, t_norm, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True
        )[0] / self.system.t_scale
        d2x_dt2 = torch.autograd.grad(
            dx_dt, t_norm, grad_outputs=torch.ones_like(dx_dt), create_graph=True, retain_graph=True
        )[0] / self.system.t_scale
        return x, dx_dt, d2x_dt2, x_norm


class SIREN(nn.Module):
    """消融基线 D：正弦表示网络（sinusoidal representation network）。

    参考 Sitzmann 等人的实现，用于验证正弦激活在高 Q 值弱阻尼系统中的适用性。
    """

    def __init__(self, system: SystemConfig, hidden_dim: int = 256, num_layers: int = 4, omega_0: float = 30.0):
        super().__init__()
        self.system = system
        self.omega_0 = omega_0

        layers = [nn.Linear(1, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.ModuleList(layers)

        with torch.no_grad():
            self.net[0].weight.uniform_(-1.0, 1.0)
            for i in range(1, len(self.net)):
                if i < len(self.net) - 1:
                    self.net[i].weight.uniform_(
                        -np.sqrt(6.0 / hidden_dim) / omega_0,
                        np.sqrt(6.0 / hidden_dim) / omega_0,
                    )
                else:
                    self.net[i].weight.uniform_(-np.sqrt(6.0 / hidden_dim), np.sqrt(6.0 / hidden_dim))

    def forward(self, t_norm: torch.Tensor):
        """前向计算，返回格式与 :class:`FouCLNet` 一致。"""
        if not torch.is_grad_enabled():
            x = torch.sin(self.omega_0 * self.net[0](t_norm))
            for i in range(1, len(self.net) - 1):
                x = torch.sin(self.net[i](x))
            x_norm = self.net[-1](x)
            return x_norm * self.system.x_scale, None, None, x_norm

        t_norm = t_norm.requires_grad_(True)
        x = torch.sin(self.omega_0 * self.net[0](t_norm))
        for i in range(1, len(self.net) - 1):
            x = torch.sin(self.net[i](x))
        x_norm = self.net[-1](x)
        x = x_norm * self.system.x_scale
        dx_dt = torch.autograd.grad(
            x, t_norm, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True
        )[0] / self.system.t_scale
        d2x_dt2 = torch.autograd.grad(
            dx_dt, t_norm, grad_outputs=torch.ones_like(dx_dt), create_graph=True, retain_graph=True
        )[0] / self.system.t_scale
        return x, dx_dt, d2x_dt2, x_norm


def build_model(name: str, system: SystemConfig, train: TrainConfig) -> nn.Module:
    """按名称构造模型。

    Parameters
    ----------
    name:
        ``fouclnet`` / ``mlp`` / ``siren``。

    Returns
    -------
    torch.nn.Module
    """
    key = name.strip().lower()
    if key in {"fouclnet", "ffn", "improved", "ours"}:
        return FouCLNet(system, train)
    if key in {"mlp", "naive", "naive_mlp"}:
        return NaiveMLP(system, train)
    if key == "siren":
        return SIREN(system)
    raise ValueError(f"未知模型名称: {name!r}（可选: fouclnet / mlp / siren）")


#: 兼容别名：早期版本中的类名
ImprovedTorsionPINN = FouCLNet
