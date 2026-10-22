"""网络结构与权重兼容性测试（需要 PyTorch）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")

from fouclnet.config import load_config  # noqa: E402
from fouclnet.loss import HybridLoss  # noqa: E402
from fouclnet.model import FouCLNet, NaiveMLP, SIREN  # noqa: E402


def test_fourier_feature_frequencies_are_log_spaced():
    """傅里叶频率应覆盖 [f_min, f_max] 且按对数间距分布。"""
    cfg = load_config()
    model = FouCLNet(cfg.system, cfg.train)
    freqs = model.fourier.B.detach().numpy().flatten()
    assert freqs.size == cfg.train.fourier_dim
    assert freqs.min() == pytest.approx(cfg.train.freq_min, rel=1e-5)
    assert freqs.max() == pytest.approx(cfg.train.freq_max, rel=1e-5)
    ratios = freqs[1:] / freqs[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-4)


def test_forward_shapes_and_derivatives():
    """前向应返回角位移与一、二阶导数，且导数数量级合理。"""
    cfg = load_config()
    model = FouCLNet(cfg.system, cfg.train)
    t = torch.linspace(0, 1, 32, dtype=torch.float32).reshape(-1, 1) / cfg.system.t_scale
    theta, d1, d2, theta_norm = model(t)
    assert theta.shape == (32, 1)
    assert d1.shape == (32, 1)
    assert d2.shape == (32, 1)
    assert torch.all(torch.isfinite(theta))
    assert torch.all(torch.isfinite(d1))


def test_hybrid_loss_is_finite_and_weight_schedule():
    """混合损失应有限，且 PDE 权重按课程学习退火。"""
    cfg = load_config()
    model = FouCLNet(cfg.system, cfg.train)
    loss_fn = HybridLoss(cfg.system, cfg.train)

    t = torch.rand(64, 1) * 10.0
    Ns = torch.full((64, 1), 1e-9)
    theta_target = torch.sin(t)

    losses, _ = loss_fn.compute(model, t, Ns, theta_target, epoch=0)
    assert np.isfinite(float(losses["total"].item()))
    assert loss_fn.pde_weight(0) == 0.0
    assert loss_fn.pde_weight(cfg.train.pde_start_epoch) == 0.0
    assert loss_fn.pde_weight(cfg.train.pde_full_epoch) == pytest.approx(cfg.train.w_pde_max)
    assert loss_fn.pde_weight(cfg.train.max_epochs) == pytest.approx(cfg.train.w_pde_max)


def test_published_checkpoint_loads():
    """随仓库发布的权重应能加载进当前模型定义。"""
    cfg = load_config()
    checkpoint = ROOT / "examples" / "torsion_pendulum_seismic" / "checkpoints" / "model_full.pth"
    if not checkpoint.exists():
        pytest.skip("未提供权重文件")
    model = FouCLNet(cfg.system, cfg.train)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    t = torch.linspace(0, 1, 16, dtype=torch.float32).reshape(-1, 1) / cfg.system.t_scale
    with torch.no_grad():
        theta = model(t)[0]
    assert torch.all(torch.isfinite(theta))


def test_baseline_models_forward():
    """消融基线（MLP / SIREN）应能正常前向。"""
    cfg = load_config()
    t = torch.rand(8, 1)
    assert NaiveMLP(cfg.system, cfg.train)(t)[0].shape == (8, 1)
    assert SIREN(cfg.system)(t)[0].shape == (8, 1)
