"""集中配置：系统物理参数、训练超参数、数据处理与评估设置。

设计原则
--------
1. 配置以 ``dataclass`` 承载，可通过 YAML/JSON 文件或
   ``--set train.max_epochs=1000`` 形式覆盖单项；
2. 物理层采用**广义二阶线性系统**形式::

       M * x'' + C * x' + K * x = F(t)

   其中 ``M``/``C``/``K`` 为广义质量、阻尼与刚度，``F(t)`` 为广义激励力。
   扭转摆（转动惯量 I、阻尼 gamma、刚度 kappa、等效力矩 Ns）只是该形式的一个特例，
   具体取值见 ``examples/`` 下的案例配置；
3. 默认值均为**方法级默认值**（不绑定任何具体数据集），
   案例相关取值一律放在 ``examples/<案例>/config.yaml`` 中。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass
class SystemConfig:
    """广义二阶线性系统的物理参数。

    运动方程::

        M * x'' + C * x' + K * x = F(t)

    对扭转摆取 ``M = I``（转动惯量）、``C = gamma``（阻尼系数）、
    ``K = kappa``（扭转刚度）、``F = Ns``（等效力矩）即可。
    """

    #: 广义质量 M [kg]
    mass: float = 1.0
    #: 广义阻尼 C
    damping: float = 0.1
    #: 广义刚度 K
    stiffness: float = 1.0
    #: 采样率 [Hz]
    fs: float = 200.0
    #: 初始位移
    x0: float = 0.0
    #: 初始速度
    v0: float = 0.0
    #: 时间归一化尺度（无量纲化用）
    t_scale: float = 50.0
    #: 输出量纲还原尺度（网络输出 × 该值 = 物理量）
    x_scale: float = 1.0e-3

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        """采样时间步长 [s]。"""
        return 1.0 / self.fs

    @property
    def omega0(self) -> float:
        """无阻尼固有角频率 sqrt(K / M) [rad/s]。"""
        return math.sqrt(self.stiffness / self.mass)

    @property
    def zeta(self) -> float:
        """阻尼比 C / (2 sqrt(M K))。"""
        return self.damping / (2.0 * math.sqrt(self.mass * self.stiffness))

    @property
    def f0(self) -> float:
        """固有频率 [Hz]。"""
        return self.omega0 / (2.0 * math.pi)


@dataclass
class TrainConfig:
    """网络结构与训练超参数。"""

    # ---------------- 随机性 ----------------
    #: 训练/验证随机划分比例（训练域占整个记录的 80%）
    split_ratio: float = 0.8
    #: 全局随机种子
    seed: int = 42

    # ---------------- 网络结构 ----------------
    #: 傅里叶特征频率个数（对数间距分布于 freq_min ~ freq_max）
    fourier_dim: int = 512
    #: 傅里叶特征频率下界 [Hz]
    freq_min: float = 0.003
    #: 傅里叶特征频率上界 [Hz]
    freq_max: float = 0.02
    #: 主干网络隐藏层宽度
    hidden_layers: Sequence[int] = (512, 512, 256, 128)

    # ---------------- 损失权重 ----------------
    #: 数据拟合项权重
    w_data: float = 1.0
    #: 硬振幅约束（HAC）项权重
    w_corr: float = 10.0
    #: 频域匹配项权重
    w_freq: float = 10.0
    #: 初始条件项权重（默认关闭）
    w_ic: float = 0.0

    # ---------------- 频域项关注频带 ----------------
    #: 频域项关注频带下界 [Hz]；``None`` 时取 ``0.7 * f0``
    freq_band_min: Optional[float] = None
    #: 频域项关注频带上界 [Hz]；``None`` 时取 ``1.4 * f0``
    freq_band_max: Optional[float] = None

    # ---------------- 弱物理残差课程学习 ----------------
    #: 开始引入物理残差项的训练轮次
    pde_start_epoch: int = 10000
    #: 物理残差权重达到最大值的训练轮次
    pde_full_epoch: int = 20000
    #: 物理残差权重上限（1e-8 量级弱正则化）
    w_pde_max: float = 1e-8
    #: 自适应提升残差权重的相关系数阈值
    pde_adaptive_r_threshold: float = 0.999
    #: 自适应提升残差权重的倍率
    pde_adaptive_multiplier: float = 1.2
    #: 自适应提升残差权重的检查间隔（轮）
    pde_adaptive_interval: int = 2000

    # ---------------- 优化与调度 ----------------
    #: 最大训练轮数
    max_epochs: int = 40000
    #: 批大小
    batch_size: int = 2048
    #: 验证间隔（轮）
    validation_interval: int = 500
    #: 学习率
    lr: float = 1e-4
    #: 权重衰减
    weight_decay: float = 1e-5
    #: 余弦退火调度器 T_max
    scheduler_t_max: int = 3000
    #: 余弦退火调度器最小学习率
    scheduler_eta_min: float = 1e-6
    #: 梯度裁剪范数
    grad_clip: float = 1.0
    #: 早停耐心值
    patience: int = 30
    #: 早停最小改进量
    min_delta: float = 1e-7

    # ---------------- 采样加权 ----------------
    #: 重点关注时间窗起点 [s]
    focus_start: float = 1500.0
    #: 重点关注时间窗终点 [s]
    focus_end: float = 2000.0
    #: 重点时间窗样本权重
    focus_weight: float = 5.0

    # ---------------- 设备 ----------------
    #: 训练设备：``auto`` / ``cuda`` / ``cpu``
    device: str = "auto"


@dataclass
class DataConfig:
    """输入时序数据（单列 CSV）的读取与预处理设置。

    通用约定：CSV 为单列（或指定列）浮点采样值，表示某个**被测物理通道**
    （位移、速度、加速度、应变……）。经平滑与求导后得到广义激励力::

        F(t) = forcing_scale * d^{derivative_order} x / dt^{derivative_order}
    """

    #: 数据 CSV 路径（相对仓库根目录，或绝对路径）
    csv_path: str = "data/series.csv"
    #: 使用的列序号（0 起）；``None`` 表示单列文件
    column: Optional[int] = None
    #: 使用记录前端的比例
    use_fraction: float = 0.20
    #: 平滑方式：``savgol`` / ``none``
    smoothing: str = "savgol"
    #: Savitzky-Golay 平滑窗口长度（点）
    savgol_window: int = 51
    #: Savitzky-Golay 多项式阶数
    savgol_polyorder: int = 3
    #: 对平滑后通道求导的阶数（0 / 1 / 2）
    derivative_order: int = 1
    #: 激励力比例系数：``F = forcing_scale * d^order x / dt^order``
    forcing_scale: float = 1.0


@dataclass
class EvalConfig:
    """评估、频谱分析与消融实验的时间窗/频段设置。"""

    # 连续段验证
    #: 严格隔离段起点 [s]
    isolated_start: float = 2000.0
    #: 严格隔离段终点 [s]
    isolated_end: float = 3000.0
    #: 消融实验连续时段可视化窗口起点 [s]
    vis_start: float = 500.0
    #: 消融实验连续时段可视化窗口终点 [s]
    vis_end: float = 1500.0
    #: 局部放大图时间窗起点 [s]
    zoom_start: float = 2000.0
    #: 局部放大图时间窗终点 [s]
    zoom_end: float = 2100.0

    # 外推实验
    #: 外推测试起点 [s]
    extrapolation_start: float = 15000.0
    #: 外推测试终点 [s]
    extrapolation_end: float = 16000.0

    # 频谱分析
    #: Welch 估计段长（点，131072 点 ≈ 655 s @200 Hz）
    welch_nperseg: int = 131072
    #: 高频段截止频率 [Hz]
    hf_cutoff: float = 0.02
    #: 关注的系统特征频率 [Hz]；``None`` 时取 ``system.f0``
    focus_freq: Optional[float] = None


@dataclass
class Config:
    """四大配置分组的聚合体。"""

    system: SystemConfig = field(default_factory=SystemConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转为嵌套字典（可直接写入 JSON/YAML）。"""
        return {
            "system": asdict(self.system),
            "train": asdict(self.train),
            "data": asdict(self.data),
            "evaluation": asdict(self.evaluation),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "Config":
        """由嵌套字典构造配置，未出现的键保持默认值。"""
        payload = dict(payload or {})
        groups = {
            "system": SystemConfig,
            "train": TrainConfig,
            "data": DataConfig,
            "evaluation": EvalConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, klass in groups.items():
            values = dict(payload.get(key) or {})
            valid = {f.name for f in fields(klass)}
            unknown = set(values) - valid
            if unknown:
                raise ValueError(f"配置节 '{key}' 含有未知字段: {sorted(unknown)}")
            kwargs[key] = klass(**values)
        return cls(**kwargs)

    def copy(self) -> "Config":
        """深拷贝一份配置（消融/统计实验需要改写单项参数）。"""
        return Config.from_dict(self.to_dict())


# ----------------------------------------------------------------------
# 配置文件读写与覆盖
# ----------------------------------------------------------------------
def load_config(path: Optional[str | Path] = None, overrides: Optional[Sequence[str]] = None) -> Config:
    """读取配置文件并应用命令行覆盖项。

    Parameters
    ----------
    path:
        YAML/JSON 配置文件路径；``None`` 表示全部使用默认值。
    overrides:
        形如 ``"train.max_epochs=1000"`` 的覆盖列表（字符串类型自动转换）。

    Returns
    -------
    Config
        合并后的配置对象。
    """
    payload: Dict[str, Any] = {}
    if path is not None:
        payload = _read_structured_file(Path(path))

    config = Config.from_dict(payload)
    if overrides:
        apply_overrides(config, overrides)
    return config


def apply_overrides(config: Config, overrides: Sequence[str]) -> Config:
    """就地应用 ``section.field=value`` 形式的覆盖项，返回同一对象。"""
    groups = {
        "system": config.system,
        "train": config.train,
        "data": config.data,
        "evaluation": config.evaluation,
    }
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖项格式应为 section.field=value，收到: {item!r}")
        key, raw_value = item.split("=", 1)
        parts = key.strip().split(".")
        if len(parts) != 2:
            raise ValueError(f"覆盖项键应为 section.field，收到: {key!r}")
        section_name, field_name = parts
        if section_name not in groups:
            raise ValueError(f"未知配置节 {section_name!r}，可选: {sorted(groups)}")
        section = groups[section_name]
        if not hasattr(section, field_name):
            raise ValueError(f"配置节 {section_name!r} 中不存在字段 {field_name!r}")
        setattr(section, field_name, _parse_value(raw_value))
    return config


def _parse_value(raw: str) -> Any:
    """把命令行字符串转换为 Python 基础类型。"""
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _read_structured_file(path: Path) -> Dict[str, Any]:
    """读取 YAML 或 JSON 配置文件。"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的提示
            raise ImportError(
                "读取 YAML 配置需要 PyYAML，请执行 `pip install pyyaml`，或改用 JSON 配置。"
            ) from exc
        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(f"不支持的配置文件格式: {path.suffix}（支持 .yaml/.yml/.json）")
