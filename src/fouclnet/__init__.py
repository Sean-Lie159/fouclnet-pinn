"""FouCLNet —— 面向弱信号受迫振动响应的傅里叶特征网络重建方法（开源实现）。

方法概述
--------
FouCLNet 把三件事组合成一个可复现的重建流程：

1. **对数间距傅里叶特征映射**：把可学习频率锁定在系统特征频率附近，
   缓解标准 MLP 的谱偏置问题；
2. **硬振幅约束损失（HAC）**：把波形相似性与幅值匹配解耦，
   抑制弱信号（微弧度/微位移量级）训练中的幅度衰减；
3. **弱物理残差课程学习**：残差权重退火到极小量级，
   在不破坏数据拟合的前提下引入物理一致性。

物理层采用广义二阶线性系统 ``M x'' + C x' + K x = F(t)``，
具体系统（扭摆、质量-弹簧-阻尼、结构振动……）只需替换配置与数据即可。

模块划分
--------
``config``                 系统/训练/数据/评估参数（dataclass + YAML）
``model``                  FourierFeature、FouCLNet 及消融基线（MLP / SIREN）
``loss``                   硬振幅约束混合损失（HAC + 频域项 + 弱物理残差项）
``data``                   时序数据加载、预处理与严格的数据划分（含隔离段）
``reference``              数值参考解（二阶系统积分）
``metrics``                相关性/相对误差/振幅匹配/PSD/高频能量占比/效率基准
``train``                  训练主循环
``evaluate``               同分布、外推、隔离连续段等评估流程
``ablation``               A/B/C/D/E 五组消融实验
``statistics``             n 次独立训练统计检验与 JZS 贝叶斯因子
``parameter_identification``  物理参数反演（辅助实验）
``plotting``               图表绘制实现
``utils``                  随机种子、设备、路径与日志等通用工具

为便于按需导入，本包对子模块采用惰性导入（PEP 562），
因此仅使用 numpy/scipy 的模块（如 ``data``/``reference``/``metrics``）
在没有安装 PyTorch 的环境中也可以正常 import。
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "__version__",
    "config",
    "model",
    "loss",
    "data",
    "reference",
    "metrics",
    "train",
    "evaluate",
    "ablation",
    "statistics",
    "parameter_identification",
    "plotting",
    "utils",
]

_LAZY = {
    "config": ("fouclnet.config", ("SystemConfig", "TrainConfig", "DataConfig", "EvalConfig", "Config", "load_config")),
    "model": ("fouclnet.model", ("FourierFeature", "FouCLNet", "NaiveMLP", "SIREN", "build_model")),
    "loss": ("fouclnet.loss", ("HybridLoss", "StandardLoss", "build_loss")),
    "data": ("fouclnet.data", ("SeriesData", "SplitResult", "load_series", "create_train_val_split")),
    "reference": ("fouclnet.reference", ("solve_reference",)),
    "metrics": ("fouclnet.metrics", ("correlation", "mean_relative_error", "amplitude_match", "welch_psd", "hf_energy_ratio", "summarize")),
    "train": ("fouclnet.train", ("train_model",)),
    "evaluate": ("fouclnet.evaluate", ("predict", "evaluate_indices", "evaluate_in_distribution", "evaluate_extrapolation", "evaluate_isolated_segment")),
    "ablation": ("fouclnet.ablation", ("run_ablation",)),
    "statistics": ("fouclnet.statistics", ("run_repeated_trials", "bayes_factor_ttest", "paired_statistics")),
    "parameter_identification": ("fouclnet.parameter_identification", ("inverse_parameter_identification",)),
    "plotting": ("fouclnet.plotting", ("plot_training_set_overview", "plot_zoomed_comparison", "plot_psd_hf_detail", "plot_figure4_revised")),
    "utils": ("fouclnet.utils", ("set_seed", "get_device", "ensure_dir", "save_json")),
}


def __getattr__(name: str):  # pragma: no cover - 惰性导入实现
    if name not in _LAZY:
        raise AttributeError(f"module 'fouclnet' has no attribute {name!r}")
    import importlib

    module_name, attrs = _LAZY[name]
    module = importlib.import_module(module_name)
    for attr in attrs:
        globals()[attr] = getattr(module, attr)
    return module
