# 数据格式说明

## 1. 输入文件

单列 CSV（也支持多列 + `data.column` 指定列），每行一个浮点采样值，无表头：

```
-4.25625126e-07
-4.27252140e-07
...
```

* 采样率由配置 `system.fs` 给出；
* 单位由数据本身决定（m/s、m/s^2、rad、µV……），只影响 `forcing_scale` 的取值；
* 文件大小没有硬性限制（超过 100 MB 时建议用 Git LFS 或 Release 附件分发）。

## 2. 数据链路

```
原始通道 x(t)
   │  取前 use_fraction 比例
   │  可选 Savitzky-Golay 平滑（smoothing: savgol，window/polyorder）
   │  数值求导 derivative_order 阶（0 / 1 / 2）
   ▼
广义激励力 F(t) = forcing_scale · d^order x / dt^order
   │
   ▼
数值参考解 x_ref(t)（src/fouclnet/reference.py）
   │
   ▼
训练 / 离散验证 / 严格隔离段 / 域外外推 四类索引
```

常用组合：

| 原始通道 | derivative_order | forcing_scale | 含义 |
|---|---|---|---|
| 位移 | 0 | 1.0 | 直接作为广义力 |
| 速度 | 1 | 1.0 | 一次求导得到加速度 |
| 加速度 | 0 | 1.0 | 直接作为广义力 |
| 速度 | 1 | −0.023 | 案例：等效力矩 `Ns = −Cs·a_g`（`Cs = m·arm`） |

## 3. 划分约定

* 前 `split_ratio` 比例的时间域为训练域：其中 `split_ratio` 比例的随机点用于训练，
  其余随机点作为同分布离散验证集；
* 末段完全不参与训练，用于域外外推测试；
* `evaluation.isolated_start/end` 给定时，训练域内该连续区间被**硬隔离**，
  任何点都不参与损失与参数更新，用于检验时序外推能力。

## 4. 快速自检

仓库内没有随附的通用数据集；请用案例目录下的抽样文件做流程自检：

```bash
python scripts/train.py --config examples/torsion_pendulum_seismic/config.yaml \
    --data-csv examples/torsion_pendulum_seismic/data/sample/ZA_velocity_sample.csv \
    --set data.use_fraction=1.0 --set train.max_epochs=20
```

抽样文件只覆盖记录的前 1%，**仅用于验证流程可跑通**，指标不能与完整数据对照。
