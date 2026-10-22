# FouCLNet：面向弱信号受迫振动响应的傅里叶特征网络重建

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.3724%2FCO.2026--0083-informational.svg)](https://doi.org/10.3724/CO.2026-0083)

> 一个可复现的神经算子实现：**对数间距傅里叶特征网络 + 硬振幅约束损失 + 弱物理残差课程学习**，
> 用于在强噪声背景下重建**弱信号**（微位移 / 微弧度量级）受迫振动的响应。
>
> 仓库地址：<https://github.com/Sean-Lie159/fouclnet-pinn>

本仓库是**方法库 + 案例**的组织方式：核心包 `src/fouclnet/` 与具体问题无关，
只依赖一个广义二阶线性系统 `M x'' + C x' + K x = F(t)` 和一份单列时序数据；
`examples/` 下给出一个完整的应用实例（地震动激励下高 Q 值扭摆系统的响应重建），
包含配置、训练好的权重、训练历史与全部图表。

## 1. 方法概要

| 组件 | 解决的问题 | 实现位置 |
|---|---|---|
| 对数间距傅里叶特征映射 | 标准 MLP 的**谱偏置**——难以拟合低频、窄带、强共振的响应 | `src/fouclnet/model.py::FourierFeature` |
| 硬振幅约束损失（HAC） | 弱信号训练中的**幅度衰减**（形状对、幅值小） | `src/fouclnet/loss.py::HybridLoss` |
| 频域匹配项 | 波形指标合格但**频谱相位/幅度**不匹配 | `HybridLoss._frequency_loss` |
| 弱物理残差课程学习 | 物理约束与数据拟合的**梯度冲突**（残差权重退火至 1e-8 量级） | `HybridLoss.pde_weight` |
| 数值参考解 | 缺少真实标签时，用积分结果提供监督信号与评价真值 | `src/fouclnet/reference.py` |

**网络结构**：`t → t/t_scale → [sin(2πBt), cos(2πBt)] → MLP(Tanh) → x/t_scale还原`，
其中 `B` 为 `freq_min ~ freq_max` 之间的对数间距频率（默认 512 个）。

**总损失**：`L = w_data·L_data + w_corr·L_HAC + w_freq·L_freq + w_pde·L_pde + w_ic·L_ic`。

## 2. 目录结构

```
fouclnet-pinn/
├── README.md                  本文件
├── LICENSE                    MIT 许可证
├── requirements.txt           依赖清单
├── pyproject.toml             可编辑安装配置
├── configs/default.yaml       方法级默认配置（与数据集无关）
├── src/fouclnet/              核心方法包
│   ├── config.py              系统/训练/数据/评估配置
│   ├── model.py               傅里叶特征网络与基线（MLP / SIREN）
│   ├── loss.py                HAC 混合损失与标准物理约束损失
│   ├── data.py                时序数据加载、预处理与严格划分
│   ├── reference.py           二阶系统数值参考解
│   ├── metrics.py             指标、PSD、高频能量占比、效率基准
│   ├── train.py               训练主循环
│   ├── evaluate.py            同分布 / 外推 / 隔离段评估
│   ├── ablation.py            A/B/C/D/E 五组消融
│   ├── statistics.py          重复训练统计检验与贝叶斯因子
│   ├── parameter_identification.py  物理参数反演（可选）
│   ├── plotting.py            图表绘制
│   └── utils.py               种子、设备、路径与日志
├── scripts/                   命令行入口
│   ├── train.py               训练（完整模式 / 隔离模式）
│   ├── evaluate.py            评估
│   ├── run_ablation.py        消融实验
│   ├── run_statistics.py      统计稳健性实验
│   ├── run_spectral_metrics.py  频谱指标（高频能量占比）
│   ├── make_figures.py        复现图表
│   └── prepare_data.py        原始记录 → 规范 CSV
├── examples/                  应用案例
│   └── torsion_pendulum_seismic/
│       ├── README.md          案例说明与复现步骤
│       ├── config.yaml        案例配置
│       ├── data/sample/       1% 抽样数据（冒烟测试用）
│       ├── checkpoints/       训练好的权重
│       └── results/           图表与训练历史
├── data/README.md             数据格式说明（通用）
├── docs/                      方法、复现与整理说明
└── tests/                     自检测试（pytest）
```

## 3. 环境与安装

* Python >= 3.9（开发验证环境：Python 3.13 / 3.14 + CUDA 或 CPU）
* numpy、scipy、torch（>= 2.0）、matplotlib、PyYAML

```bash
pip install -r requirements.txt
pip install -e .          # 可选：以可编辑方式安装核心包
# 或直接从 GitHub 安装
pip install git+https://github.com/Sean-Lie159/fouclnet-pinn.git
```

CPU 可跑通全流程；完整训练（40 000 轮）建议使用单卡 GPU（显存 8 GB 以上即可）。

## 4. 快速开始

### 4.1 用自己的数据

数据只需满足：**单列（或指定列）浮点时序** + 已知采样率 + 已知系统参数 `M / C / K`。

```bash
# 1) 把数据整理为单列 CSV（也可直接用原始文件）
python scripts/prepare_data.py --input raw/record.txt --output data/series.csv

# 2) 训练（把系统参数与数据选项写进配置）
python scripts/train.py --config configs/default.yaml --data-csv data/series.csv \
    --set system.mass=1.0 --set system.damping=0.05 --set system.stiffness=20.0 \
    --set data.forcing_scale=1.0 --outdir results

# 3) 评估
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth \
    --data-csv data/series.csv --outdir results
```

任何脚本都支持 `--set section.field=value` 覆盖配置。

### 4.2 复现示例案例

```bash
python scripts/train.py --config examples/torsion_pendulum_seismic/config.yaml \
    --data-csv <ZA_velocity.csv> --outdir results

python scripts/make_figures.py --config examples/torsion_pendulum_seismic/config.yaml \
    --checkpoint examples/torsion_pendulum_seismic/checkpoints/model_full.pth \
    --isolated-checkpoint examples/torsion_pendulum_seismic/checkpoints/model_isolated.pth \
    --data-csv <ZA_velocity.csv> --outdir results
```

### 4.3 自检

```bash
python -m pytest tests -q
```

## 5. 数据格式

单列 CSV，每行一个浮点采样值，无表头；多列文件用 `data.column` 指定列号。
数据链路为：**原始通道 → 可选 Savitzky-Golay 平滑 → 数值求导 → 广义激励力**，
其中 `F = forcing_scale · d^order x / dt^order`（`order` 可取 0/1/2）。
详见 [`data/README.md`](data/README.md)。

## 6. 方法边界（请在引用前确认）

* 方法在**训练分布内的离散点**上精度很高（相关系数可达 0.99+）；
* **时序外推**（训练域之外）与**未参与训练的连续时段**精度显著下降
  （示例案例中 |R| 均在 0.02–0.03 量级）；
* 因此适用定位是"**分布内插值式重建 + 单点常数级延迟访问**"，
  不适用于连续波形生成或长期预测。详见 [`docs/METHOD.md`](docs/METHOD.md)。

> **关于示例案例的数值**：仓库随附的权重是用现行代码在论文所用数据上
> **完整重训**得到的（2026-10-22，40 000 轮、seed 42），
> 实测训练域 R = 0.9994、同分布离散验证 R = 0.9994、振幅匹配 **100.0%**，
> 与论文报告值（0.9988 / 0.9958 / 100.0%）一致；
> 隔离连续段与域外外推的 \|R\| ≈ 0.01–0.04，与论文的失效结论一致。
> 重跑会因训练随机性在 R ≈ 0.99–0.999 区间内波动，判据见
> [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)；
> 逐项数值与历史核查记录见
> [`examples/torsion_pendulum_seismic/README.md`](examples/torsion_pendulum_seismic/README.md) 第 4–5 节。

## 7. 相关文献

本方法的一个完整应用实例（地震动激励下高 Q 值扭摆系统的响应重建）已正式发表：

> 李俊祥，潘玟琦，齐克奇，王少鑫，董鹏. 扭摆地面振动响应的傅里叶特征网络重建.
> 中国光学（中英文）, 2026, 19(5): 1150–1165. DOI: 10.3724/CO.2026-0083.
> <https://chineseoptics.net.cn/cn/article/doi/10.3724/CO.2026-0083>
>
> Li J., Pan W., Qi K., Wang S., Dong P. Reconstruction of torsion pendulum ground
> vibration response via Fourier feature network. *Chinese Optics*, 2026, 19(5): 1150–1165.

案例说明见 [`examples/torsion_pendulum_seismic/README.md`](examples/torsion_pendulum_seismic/README.md)。

**BibTeX**

```bibtex
@article{li2026torsion,
  title   = {Reconstruction of torsion pendulum ground vibration response via Fourier feature network},
  author  = {Li, Jun-xiang and Pan, Wen-qi and Qi, Ke-qi and Wang, Shao-xin and Dong, Peng},
  journal = {Chinese Optics},
  year    = {2026},
  volume  = {19},
  number  = {5},
  pages   = {1150--1165},
  doi     = {10.3724/CO.2026-0083}
}
```


## 8. 许可证

代码以 MIT 许可证发布，见 [`LICENSE`](LICENSE)。
`examples/` 下的权重与图表为随案例一并发布的复现产物；
数据集本身不在本仓库中分发。
