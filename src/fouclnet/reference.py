"""数值参考解（二阶线性受迫振动的积分求解）。

方法把数值积分结果作为“真值标签”：一方面用于监督训练（数据项），
另一方面作为重建精度的评价基准。

方程::

    M * x'' + C * x' + K * x = F(t)
    x(0) = x0,  x'(0) = v0

实现说明
--------
使用 ``scipy.integrate.odeint``（lsoda 求解器），长时程下数值轨迹稳定；
若改用 ``solve_ivp`` 等其它求解器，细微差异会传导到相关系数与 PSD 指标上。
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from .config import SystemConfig


def solve_reference(t, forcing, system: SystemConfig) -> np.ndarray:
    """计算广义二阶系统的位移参考解。

    Parameters
    ----------
    t:
        时间轴 [s]，等间隔。
    forcing:
        与 ``t`` 等长的广义激励力序列 F(t)。
    system:
        系统参数（M、C、K、初始条件、采样率）。

    Returns
    -------
    numpy.ndarray
        位移参考解，形状与 ``t`` 相同。
    """
    t = np.asarray(t, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    if t.shape != forcing.shape:
        raise ValueError(f"t 与 forcing 长度不一致: {t.shape} vs {forcing.shape}")

    def equations(state, t_val):
        x, v = state
        # 线性插值取当前时刻的激励力
        f_val = np.interp(t_val, t, forcing)
        dx = v
        dv = (f_val - system.damping * v - system.stiffness * x) / system.mass
        return [dx, dv]

    initial_state = [system.x0, system.v0]
    solution = odeint(equations, initial_state, t)
    reference = solution[:, 0]

    print(f"[参考解] 积分完成，位移 std = {np.std(reference):.4e}")
    return reference
