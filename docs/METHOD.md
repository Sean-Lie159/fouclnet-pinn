# 方法说明

本文给出 FouCLNet 的数学形式与实现约定。所有参数均可在配置文件中找到对应项。

## 1. 物理层：广义二阶线性系统

```
M · x'' + C · x' + K · x = F(t)
```

| 符号 | 含义 | 派生量 |
|---|---|---|
| M | 广义质量 | `omega0 = sqrt(K/M)` |
| C | 广义阻尼 | `zeta = C / (2 sqrt(M K))` |
| K | 广义刚度 | `f0 = omega0 / (2 pi)` |
| F(t) | 广义激励力 | 由被测通道经平滑/求导得到 |

对扭转摆取 `M = I`、`C = gamma`、`K = kappa`、`F = Ns`；
对质量-弹簧-阻尼系统取 `M = m`、`C = c`、`K = k`、`F = F(t)`。

## 2. 激励构造

```
原始通道 x(t)  --平滑-->  x_s(t)  --求导 n 阶-->  d^n x_s / dt^n
F(t) = forcing_scale · d^n x_s / dt^n
```

* 平滑：Savitzky-Golay（窗口 51、多项式阶数 3 为默认），用于抑制求导对高频噪声的放大；
* 求导：中心差分 `numpy.gradient`；
* `forcing_scale` 承担量纲换算与符号约定。

## 3. 参考解

以 `scipy.integrate.odeint`（lsoda）积分上式，步长取采样间隔，
所得 `x_ref` 作为监督标签与评价真值。

## 4. 网络结构

```
t_norm = t / t_scale
gamma(t) = [sin(2 pi B t), cos(2 pi B t)]   # B: 对数间距频率, 默认 512 个
MLP: 1024 -> 512 -> 512 -> 256 -> 128 -> 1   (Tanh, Xavier 正态初始化)
x = net(gamma(t)) · x_scale
```

频率区间 `freq_min ~ freq_max` 应当覆盖系统的特征频率 `f0`；
过宽会稀释特征，过窄则限制泛化能力。

## 5. 混合损失

```
L = w_data·L_data + w_corr·L_HAC + w_freq·L_freq + w_pde·L_pde + w_ic·L_ic
```

| 项 | 定义 | 默认权重 |
|---|---|---|
| `L_data` | MSE + 0.1·MAE + 0.01·相对误差 | 1.0 |
| `L_HAC` | Z-score 标准化波形 MSE + 相对标准差误差 | 10.0 |
| `L_freq` | 对数幅度谱 MSE + 2×(1 − cos Δφ)，每 100 轮一次，批内样本 ≥ 4096 | 10.0 |
| `L_pde` | 归一化物理残差平方均值 | 0 → 1e-8（课程学习） |
| `L_ic` | 初始位移/速度约束 | 0（关闭） |

**HAC 的作用**：普通 MSE 允许“形状对、幅值小”的解；把波形相关性与幅值匹配
拆成两个子项后，弱信号训练中的幅度衰减被显著抑制。

**课程学习（残差权重退火）**：`epoch < pde_start_epoch` 时权重为 0；
`pde_start_epoch ≤ epoch < pde_full_epoch` 时线性增长到 `w_pde_max = 1e-8`；之后保持。
若验证相关系数超过 `pde_adaptive_r_threshold`，则每 `pde_adaptive_interval` 轮
把上限乘以 `pde_adaptive_multiplier`（硬上限 1e-3）。

**频域关注频带**：默认 `[0.7 f0, 1.4 f0]`，可用 `train.freq_band_min/max` 覆盖。

## 6. 训练与采样

* 优化器 AdamW（lr 1e-4、weight_decay 1e-5）+ 余弦退火（T_max 3000、eta_min 1e-6）；
* 每轮按“重点时间窗加权”从训练集有放回采样 `batch_size` 个点；
* 每 `validation_interval` 轮在离散验证集上评估一次，以平均相对误差为早停判据；
* 训练结束后加载最佳权重返回。

## 7. 已知能力边界

* **分布内离散点插值**：精度很高（R 可达 0.99+）；
* **未参与训练的连续时段**：显著失效（案例中 R ≈ 0.029）；
* **训练域之外的外推**：显著失效（案例中 R ≈ 0.019）；
* 弱物理残差的主要作用是**降低训练方差**（案例中 E/C 组方差比约 22.6 倍），
  对离散点平均精度的提升有限。

因此本方法的定位是“分布内插值式重建 + 单点常数级延迟访问”，
不适用于连续波形生成或长期预测。
