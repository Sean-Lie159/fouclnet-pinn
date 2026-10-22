# 应用案例

本目录收录 FouCLNet 的完整应用实例，每个案例包含：配置、数据抽样、
训练好的权重、训练历史与全部图表。

| 案例 | 说明 | 状态 |
|---|---|---|
| [`torsion_pendulum_seismic`](torsion_pendulum_seismic/README.md) | 地震动激励下高 Q 值扭摆系统的响应重建 | 完整（配置 + 权重 + 图表 + 训练历史） |

新增案例时，请遵循同一目录结构：

```
<case_name>/
├── README.md          问题设定、复现步骤、结果与结论边界
├── config.yaml        案例配置（相对仓库根目录的路径）
├── data/sample/       小规模抽样数据（冒烟测试用）
├── checkpoints/       训练好的权重
└── results/           图表与训练历史
```
