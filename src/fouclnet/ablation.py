"""消融实验：A/B/C/D/E 五组对比。

组别定义
--------
======  ==========================================  ==========================
组别    配置                                        验证目的
======  ==========================================  ==========================
A       标准 MLP（无傅里叶特征、无物理约束）          谱偏置的影响
B       FFN 无 HAC（仅数据 MSE）                     硬振幅约束的必要性
C       完整方法 FouCLNet（FFN + HAC + 弱 PDE）       本文方法
D       SIREN（正弦激活）                             正弦激活的适用性
E       FFN + HAC，lambda_PDE = 0（纯数据驱动基准）    弱 PDE 正则化的作用
======  ==========================================  ==========================

公平性约定
----------
除 C 组直接加载已训练好的主模型权重外，其余各组统一训练 40 000 轮、
统一批大小/学习率/早停策略与统一评估口径。
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import Config
from .loss import HybridLoss
from .model import FouCLNet, NaiveMLP, SIREN
from .train import train_model
from .utils import get_device, set_seed


def _make_group(cfg: Config, group: str, device):
    """按组别构造模型、损失函数与该组生效的配置副本。"""
    key = group.upper()
    if key in {"A", "MLP"}:
        model = NaiveMLP(cfg.system, cfg.train).to(device)
        return model, None, copy.deepcopy(cfg)
    if key in {"B", "FFN_NO_HAC"}:
        model = FouCLNet(cfg.system, cfg.train).to(device)
        return model, "data_only", copy.deepcopy(cfg)
    if key in {"C", "OURS", "FOUCLNET"}:
        model = FouCLNet(cfg.system, cfg.train).to(device)
        return model, HybridLoss(cfg.system, cfg.train), copy.deepcopy(cfg)
    if key in {"D", "SIREN"}:
        model = SIREN(cfg.system).to(device)
        return model, "data_only", copy.deepcopy(cfg)
    if key in {"E", "NO_PDE"}:
        cfg_e = copy.deepcopy(cfg)
        cfg_e.train.w_pde_max = 0.0
        cfg_e.train.pde_start_epoch = cfg.train.max_epochs + 1
        cfg_e.train.pde_full_epoch = cfg.train.max_epochs + 2
        model = FouCLNet(cfg.system, cfg_e.train).to(device)
        return model, HybridLoss(cfg.system, cfg_e.train), cfg_e
    raise ValueError(f"未知消融组: {group!r}")



def _data_only_loss(physics, train_cfg):
    """仅数据 MSE 的损失（B/D 组使用）。"""

    class _DataOnlyLoss:
        def compute(self, model, t_norm, Ns, theta_target, epoch):
            import torch

            theta = model(t_norm)[0]
            loss = torch.mean((theta - theta_target) ** 2)
            return {"total": loss, "data": loss, "corr": loss * 0, "freq": loss * 0, "pde": loss * 0, "w_pde": 0.0}, theta

    return _DataOnlyLoss()


def run_ablation(
    cfg: Config,
    data,
    split,
    reference,
    groups: Optional[List[str]] = None,
    main_checkpoint: Optional[str] = None,
    output_dir: str = "results",
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """执行消融实验并返回各组指标。

    Parameters
    ----------
    cfg:
        完整配置。
    data, split, reference:
        地面运动数据、划分与 RK4 参考解。
    groups:
        需要运行的组别，默认 ``["A", "B", "E", "D", "C"]``。
    main_checkpoint:
        C 组使用的模型权重路径。
    output_dir:
        结果输出目录（用于保存权重与中间结果）。
    device:
        计算设备。

    Returns
    -------
    dict
        ``{组别: {"R": ..., "amplitude_match": ..., "relative_error": ...}}``。
    """
    from pathlib import Path

    from .evaluate import evaluate_indices

    device = device or get_device(cfg.train.device)
    groups = groups or ["A", "B", "E", "D", "C"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, np.ndarray] = {}

    for group in groups:
        key = group.upper()
        print(f"\n--- 消融组 {key} ---")
        set_seed(cfg.train.seed)
        model, loss, group_cfg = _make_group(cfg, key, device)

        if key in {"C", "OURS", "FOUCLNET"} and main_checkpoint:
            state = torch.load(main_checkpoint, map_location=device)
            model.load_state_dict(state)
            model.eval()
            print(f"  已加载主模型权重: {main_checkpoint}")
        else:
            if loss == "data_only":
                loss = _data_only_loss(group_cfg.system, group_cfg.train)
            model, _ = train_model(
                group_cfg,
                data,
                split,
                reference,
                model=model,
                loss_fn=loss,
                model_save_path=out_dir / f"ablation_{key}.pth",
                verbose=False,
            )

        metrics = evaluate_indices(model, data.t, reference, split.val_idx, group_cfg, device)
        results[key] = metrics
        print(f"  R = {metrics['R']:.6f} | 振幅匹配 = {metrics['amplitude_match']:.1f}%")

        # 连续时段（500–1500 s）重建质量
        mask = (data.t >= cfg.evaluation.vis_start) & (data.t <= cfg.evaluation.vis_end)
        from .evaluate import predict

        seg_pred = predict(model, data.t[mask], group_cfg, device)
        seg_ref = reference[mask]
        from .metrics import amplitude_match, correlation

        results[key]["continuous_R"] = correlation(seg_pred, seg_ref)
        results[key]["continuous_amplitude_match"] = amplitude_match(seg_pred, seg_ref)
        predictions[key] = seg_pred

    print("\n========== 消融实验结果汇总 ==========")
    print(f"{'组别':<6}{'R(离散验证)':<16}{'振幅匹配(%)':<14}{'连续段R':<12}{'连续段振幅(%)'}")
    for key, metrics in results.items():
        print(
            f"{key:<6}{metrics['R']:<16.6f}{metrics['amplitude_match']:<14.1f}"
            f"{metrics.get('continuous_R', float('nan')):<12.4f}"
            f"{metrics.get('continuous_amplitude_match', float('nan')):.1f}"
        )
    return results
