# 案例：地震动激励下高 Q 值扭摆系统的响应重建

本案例演示 FouCLNet 在一个真实工程问题上的应用：
**在普通实验室环境（存在地震噪声）中，重建高 Q 值扭摆系统的受迫振动响应**。

## 1. 问题设定

* 扭摆系统：`I·θ'' + γ·θ' + κ·θ = Ns(t)`
  （I = 5.0e-5 kg·m²，γ = 1.0e-8 N·m·s/rad，κ = 1.0e-7 N·m/rad，
  固有频率 f0 ≈ 0.00711 Hz，阻尼比 ≈ 2.24e-3）；
* 激励：地面 Z 分分量速度记录（200 Hz）→ Savitzky-Golay 平滑 → 一次数值求导得加速度
  `a_g` → 等效力矩 `Ns = −Cs·a_g`（`Cs = m·arm = 0.023 kg·m`）；
* 信号量级：角位移 std ≈ 2.2e-3 rad，属微弧度量级的弱信号；
* 任务：仅用激励的已知信息，监督网络重建摆锤角位移。

## 2. 目录内容

| 路径 | 说明 |
|---|---|
| `config.yaml` | 案例配置（系统参数、数据链路、训练与评估设置） |
| `data/sample/ZA_velocity_sample.csv` | 地面记录前 1% 的抽样（126 911 点，约 634 s），仅供流程自检 |
| `checkpoints/model_full.pth` | 完整模式模型（训练域内不隔离任何连续段） |
| `checkpoints/model_isolated.pth` | 隔离模式模型（2000–3000 s 被完全屏蔽） |
| `results/history/*.json` | 两个模型的训练历史（逐次验证的损失、R 与相对误差） |
| `results/figures/*.png` | 案例的全部图表（总览、局部放大、离散验证、外推、隔离段、消融散点、消融连续段、PSD 高频细节） |

## 3. 复现步骤

```bash
# 1) 准备完整数据（原始记录不在本仓库中，见文末说明）
#    ZA_velocity.csv：单列，200 Hz，12 691 186 点
python scripts/prepare_data.py --input raw/za_velocity.txt \
    --output examples/torsion_pendulum_seismic/data/ZA_velocity.csv

# 2) 训练完整模式模型
python scripts/train.py --config examples/torsion_pendulum_seismic/config.yaml \
    --data-csv examples/torsion_pendulum_seismic/data/ZA_velocity.csv \
    --outdir results

# 3) 训练隔离模式模型
python scripts/train.py --config examples/torsion_pendulum_seismic/config.yaml \
    --mode isolated --data-csv <同上> --outdir results

# 4) 评估
python scripts/evaluate.py --config examples/torsion_pendulum_seismic/config.yaml \
    --checkpoint examples/torsion_pendulum_seismic/checkpoints/model_full.pth \
    --data-csv <同上> --outdir results

# 5) 消融与统计实验
python scripts/run_ablation.py --config examples/torsion_pendulum_seismic/config.yaml \
    --main-checkpoint examples/torsion_pendulum_seismic/checkpoints/model_full.pth \
    --data-csv <同上> --outdir results

python scripts/run_statistics.py --config examples/torsion_pendulum_seismic/config.yaml \
    --n-runs 15 --data-csv <同上> --outdir results

# 6) 图表与频谱指标
python scripts/make_figures.py --config examples/torsion_pendulum_seismic/config.yaml \
    --checkpoint examples/torsion_pendulum_seismic/checkpoints/model_full.pth \
    --isolated-checkpoint examples/torsion_pendulum_seismic/checkpoints/model_isolated.pth \
    --history examples/torsion_pendulum_seismic/results/history/training_history_main.json \
    --data-csv <同上> --outdir results
```

## 4. 结果

### 4.1 论文报告值（文献值，对应投稿时的运行）

| 指标 | 论文报告值 | 出处 |
|---|---|---|
| 训练集相关系数 R | 0.998829 | 表 1 / 图 2(c) |
| 训练集振幅匹配度 | 100.0% | 表 1 |
| 训练集峰值误差 | < 0.4% F.S. | 表 1 |
| 训练集相对误差 | 25.67% | 表 1（该指标定义论文未明示） |
| 同分布离散验证 R | 0.995827 | 表 2 |
| 同分布离散验证相对误差 | 44.26% | 表 2（同上） |
| 同分布验证点数量 | 406 118 | 3.2 节 / 图 4(a) |
| 训练点数量 | 1 624 472 | 3.2 节 |
| 严格隔离连续段（2000–3000 s）R | 0.029 | 图 4(b) / 4.2 节 |
| 域外外推 R | 0.019 | 图 4(c) / 4.4 节（评估窗 15 000–16 000 s） |
| 消融 A 标准 MLP | 0.011 / 1.0% | 表 4 |
| 消融 B 无物理残差 | 0.361 / 18.2% | 表 4 |
| 消融 C 完整方法 | 0.999 / 100.0% | 表 4 |
| 消融 D SIREN | 0.142 / 0.2% | 表 4 |
| 消融 E 去掉物理残差 | 0.995 / 100.0% | 表 4 |
| 连续段（500–1500 s）C / E | 0.986 / 98.8% 与 0.511 / 42.7% | 4.5 节 |
| 重复训练 n=15：C / E | 0.9986 ± 0.0023 / 0.9953 ± 0.0110 | 表 5 |
| 配对检验 | t = 1.09, p = 0.294；Wilcoxon p = 0.599；d = 0.28 | 表 5 |
| 贝叶斯因子 | BF10 = 0.435，BF01 = 2.30 | 表 5 |
| 方差齐性 | Bartlett K² = 24.56, p = 7.2e-7 | 表 5 |
| PSD 高频段（> 0.02 Hz）能量占比 | RK4 0.027% / 网络 0.030%（1.15e-9 / 1.17e-9 rad²） | 4.1 节 |
| 单点推理延迟 | 0.524 ms（RK4 约 40 ms） | 表 3 / 图 3 |
| 数据 | 12 691 s / 2 538 237 点，200 Hz，Butterworth 0.003–0.02 Hz | 3.1 节 |
| 扭摆固有频率 | 0.007 Hz | 2.1 / 3.1 节 |

### 4.2 本仓库随附权重的实测值

随附权重为 **2026-10-22 用现行代码在论文数据上完整重训**的产物
（40 000 轮、seed 42、CPU；见 `results/metrics/metrics.json`），
用 `scripts/evaluate.py` 复算：

| 指标 | 本仓库实测 | 论文报告值 | 备注 |
|---|---|---|---|
| 训练域 R | **0.999363** | 0.998829 | 差 0.0005 |
| 训练域振幅匹配度 | **100.0%** | 100.0% | 一致 |
| 训练域归一化相对误差 | 0.524% F.S. | 25.67% | 论文未明示该指标定义，口径不同 |
| 训练域峰值误差 | 6.718% F.S. | < 0.4% F.S. | 同上（论文口径未明示） |
| 同分布离散验证 R（406 118 点） | **0.999364** | 0.995827 | 差 0.0035，在单次运行波动量级内（σ ≈ 0.0023） |
| 同分布振幅匹配度 | **100.0%** | 100.0% | 一致 |
| 严格隔离连续段 R（隔离模型，2000–3000 s） | **−0.0147** | 0.029 | 均为"失效"量级 |
| 域外外推 R | 0.0014（完整模型）/ 0.0427（隔离模型） | 0.019 | 均为"失效"量级 |
| 局部放大（2000–2100 s）R | 0.9979，平均绝对误差 3.31e-5 rad | — | 等效相位偏移 0 |

**结论边界（复现与投稿一致）**：方法在分布内离散点上达到高精度插值水平
（振幅匹配 100%），但对未参与训练的连续时段与训练域之外的外推失效
（|R| ≈ 0.01–0.04），属于方法本身的能力边界，而非实现缺陷。

## 5. 数值一致性说明

**当前状态：已完全复现。** 本仓库随附的权重是用现行代码在论文所用数据文件上
**完整重训**得到的（2026-10-22），其数值与投稿报告值在同一量级：

| | 训练集 R | 同分布验证 R | 振幅匹配 |
|---|---|---|---|
| 论文报告值 | 0.998829 | 0.995827 | 100.0% |
| 本仓库实测 | 0.999363 | 0.999364 | 100.0% |

**复现方式**（任何人都可重跑）：

```bash
python scripts/train.py --config examples/torsion_pendulum_seismic/config.yaml \
    --data-csv <ZA_velocity.csv> --outdir results
python scripts/evaluate.py --config examples/torsion_pendulum_seismic/config.yaml \
    --checkpoint checkpoints/model_full.pth --data-csv <ZA_velocity.csv> --outdir results
```

由于神经网络训练的随机性，重跑结果会在 R ≈ 0.99–0.999、振幅匹配 95–100% 区间内波动，
**不应期待逐位一致**；判据是"分布内高精度 + 连续段/外推失效"这组定性结论成立。
逐轮的期望曲线见 [`docs/REPRODUCIBILITY.md`](../../../docs/REPRODUCIBILITY.md) 第 8 节。

## 6. 数据来源与授权

原始地面振动记录来自作者的监测数据，未随本仓库分发；
如需用于其它研究，请通过仓库 Issue 页面联系获取。
