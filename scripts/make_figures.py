"""用已训练模型复现全套图表（总览、局部放大、验证、隔离段、外推、消融）。

示例::

    python scripts/make_figures.py --checkpoint checkpoints/best_pinn_model_main.pth \
        --isolated-checkpoint checkpoints/best_pinn_model_isolated.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import add_common_arguments, build_config, prepare_dataset

from fouclnet.data import create_train_val_split
from fouclnet.evaluate import predict
from fouclnet.model import FouCLNet
from fouclnet.plotting import (
    plot_extrapolation,
    plot_figure4_revised,
    plot_in_distribution,
    plot_isolated_segment,
    plot_zoomed_comparison,
)
from fouclnet.utils import ensure_dir, get_device


def _load_model(cfg, checkpoint: str, device):
    """加载模型权重。"""
    model = FouCLNet(cfg.system, cfg.train).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="复现图表")
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", required=True, help="主模型权重路径")
    parser.add_argument("--isolated-checkpoint", default=None, help="隔离段对照模型权重路径")
    parser.add_argument("--history", default=None, help="训练历史 JSON（用于图 2(d)(e)）")
    args = parser.parse_args()

    cfg = build_config(args)
    data, reference, split = prepare_dataset(cfg)
    device = get_device(cfg.train.device)
    outdir = ensure_dir(Path(args.outdir))
    figdir = ensure_dir(outdir / "figures")

    model = _load_model(cfg, args.checkpoint, device)
    theta_pred = predict(model, data.t, cfg, device)

    # 图 2：训练集总览（需要训练历史）
    history = {"epoch": [], "total": [], "data": [], "corr": [], "freq": [], "r_value": []}
    if args.history and Path(args.history).exists():
        history = json.loads(Path(args.history).read_text(encoding="utf-8"))
    from fouclnet.plotting import plot_training_set_overview

    plot_training_set_overview(
        cfg, data.t, reference, theta_pred, history,
        output_path=figdir / "overview.png",
        split_point=split.split_point,
    )

    # 图 3：局部放大
    plot_zoomed_comparison(
        cfg, data.t, reference, theta_pred,
        output_path=figdir / "zoomed_comparison.png",
    )

    # 图 4(a)：同分布离散验证
    plot_in_distribution(
        cfg, data.t, reference, theta_pred, split.val_idx,
        output_path=figdir / "in_distribution.png",
    )

    # 图 4(c)：域外外推
    plot_extrapolation(
        cfg, data.t, reference, theta_pred, split.test_idx,
        output_path=figdir / "extrapolation.png",
    )

    # 图 4(b) 与汇总版：需要隔离段对照模型
    if args.isolated_checkpoint:
        isolated_split = create_train_val_split(
            data.t,
            split_ratio=cfg.train.split_ratio,
            seed=cfg.train.seed,
            cont_val_start=cfg.evaluation.isolated_start,
            cont_val_end=cfg.evaluation.isolated_end,
            fs=data.fs,
        )
        iso_model = _load_model(cfg, args.isolated_checkpoint, device)
        iso_pred = predict(iso_model, data.t, cfg, device)
        plot_isolated_segment(
            cfg, data.t, reference, iso_pred, isolated_split.cont_val_idx,
            output_path=figdir / "isolated_segment.png",
        )
        plot_figure4_revised(
            cfg, data.t, reference, iso_pred, theta_pred, isolated_split,
            output_path=figdir / "isolated_vs_extrapolation.png",
        )
    else:
        print("[提示] 未提供 --isolated-checkpoint，跳过图 4(b) 与汇总版。")

    print(f"[完成] 图表已写入 {figdir}")


if __name__ == "__main__":
    main()
