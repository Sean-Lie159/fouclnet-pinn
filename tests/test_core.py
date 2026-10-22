"""轻量自检测试：配置、数据划分、参考解与指标。

运行::

    python -m pytest tests -q

测试不依赖真实地面运动数据，可在没有 PyTorch 的环境中运行；
涉及网络结构的测试会自动跳过（见 ``test_model.py``）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fouclnet.config import Config, load_config  # noqa: E402
from fouclnet.data import create_train_val_split  # noqa: E402
from fouclnet.metrics import (  # noqa: E402
    amplitude_match,
    correlation,
    hf_energy_ratio,
    mean_relative_error,
    summarize,
)
from fouclnet.reference import solve_reference  # noqa: E402


def test_default_config_is_method_level():
    """默认配置应为方法级默认值（与具体数据集无关）。"""
    cfg = load_config()
    assert cfg.system.mass == pytest.approx(1.0)
    assert cfg.system.damping == pytest.approx(0.1)
    assert cfg.system.stiffness == pytest.approx(1.0)
    assert cfg.system.fs == pytest.approx(200.0)
    assert cfg.train.fourier_dim == 512
    assert cfg.train.max_epochs == 40000
    assert cfg.train.seed == 42
    assert cfg.evaluation.hf_cutoff == pytest.approx(0.02)


def test_example_config_matches_case_values():
    """案例配置应给出该案例的物理参数与数据链路。"""
    yaml = pytest.importorskip("yaml")
    path = ROOT / "examples" / "torsion_pendulum_seismic" / "config.yaml"
    cfg = load_config(path)
    assert cfg.system.mass == pytest.approx(5e-5)
    assert cfg.system.damping == pytest.approx(1e-8)
    assert cfg.system.stiffness == pytest.approx(1e-7)
    assert cfg.system.f0 == pytest.approx(0.00711, abs=1e-5)
    assert cfg.data.derivative_order == 1
    assert cfg.data.forcing_scale == pytest.approx(-0.023)
    assert cfg.train.freq_band_min == pytest.approx(0.005)
    assert cfg.train.freq_band_max == pytest.approx(0.01)


def test_config_yaml_roundtrip():
    """YAML 配置应能读入并与默认值一致（需要 PyYAML）。"""
    yaml = pytest.importorskip("yaml")
    path = ROOT / "configs" / "default.yaml"
    cfg = load_config(path)
    assert cfg.train.fourier_dim == 512
    assert cfg.system.t_scale == pytest.approx(50.0)


def test_config_overrides():
    """命令行覆盖项应正确生效。"""
    cfg = load_config(None, ["train.max_epochs=123", "train.seed=7", "data.csv_path=/tmp/x.csv"])
    assert cfg.train.max_epochs == 123
    assert cfg.train.seed == 7
    assert cfg.data.csv_path == "/tmp/x.csv"


def test_split_has_no_leakage():
    """训练/验证/外推/隔离段之间不允许出现索引重叠。"""
    t = np.arange(0, 10_000) / 200.0
    split = create_train_val_split(t, split_ratio=0.8, seed=42, fs=200.0)

    train = set(split.train_idx.tolist())
    val = set(split.val_idx.tolist())
    test = set(split.test_idx.tolist())
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert max(train) < split.split_point <= min(test)


def test_split_isolates_segment():
    """严格隔离段不应出现在训练/验证集中。"""
    t = np.arange(0, 2_000_000) / 200.0  # 10 000 s，足以覆盖 2000–3000 s
    split = create_train_val_split(t, split_ratio=0.8, seed=42,
                                   cont_val_start=2000.0, cont_val_end=3000.0, fs=200.0)
    assert split.cont_val_idx.size == 200_000
    assert not (set(split.cont_val_idx.tolist()) & set(split.train_idx.tolist()))
    assert not (set(split.cont_val_idx.tolist()) & set(split.val_idx.tolist()))



def test_split_is_deterministic():
    """相同种子应给出相同划分。"""
    t = np.arange(0, 5000) / 200.0
    a = create_train_val_split(t, seed=42, fs=200.0)
    b = create_train_val_split(t, seed=42, fs=200.0)
    assert np.array_equal(a.train_idx, b.train_idx)
    assert np.array_equal(a.val_idx, b.val_idx)


def test_reference_solution_is_physical():
    """无激励时解应保持为零；给阶跃激励时应产生非零响应。"""
    cfg = Config()
    t = np.arange(0, 2000) / cfg.system.fs
    zeros = solve_reference(t, np.zeros_like(t), cfg.system)
    assert np.allclose(zeros, 0.0, atol=1e-18)

    impulse = np.zeros_like(t)
    impulse[0] = 1e-9
    response = solve_reference(t, impulse, cfg.system)
    assert np.max(np.abs(response)) > 0.0


def test_metrics_on_known_signals():
    """在已知信号上验证指标定义。"""
    x = np.sin(np.linspace(0, 10, 1000))
    assert correlation(x, x) == pytest.approx(1.0)
    assert mean_relative_error(x, x) == pytest.approx(0.0, abs=1e-9)
    assert amplitude_match(x, x) == pytest.approx(100.0)
    assert amplitude_match(0.5 * x, x) == pytest.approx(50.0)

    summary = summarize(x + 0.1 * np.cos(np.linspace(0, 3, 1000)), x)
    assert summary["R"] < 1.0
    assert summary["MSE"] > 0.0


def test_hf_energy_ratio_on_synthetic_signal():
    """低频正弦的高频能量占比应接近 0。"""
    fs = 200.0
    t = np.arange(0, 400_000) / fs
    x = np.sin(2 * np.pi * 0.007 * t)
    stats = hf_energy_ratio(x, fs, f_cutoff=0.02, nperseg=131072)
    assert stats["hf_ratio_percent"] < 1.0
    assert stats["total_energy"] > 0.0
    assert stats["df"] == pytest.approx(fs / 131072, rel=1e-6)
