"""评估已训练模型：同分布离散验证、域外外推、严格隔离连续段。

示例::

    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint checkpoints/model_full.pth --data-csv data/series.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import add_common_arguments, build_config, prepare_dataset

from fouclnet.data import create_train_val_split
from fouclnet.evaluate import (
    evaluate_extrapolation,
    evaluate_in_distribution,
    evaluate_isolated_segment,
)
from fouclnet.model import FouCLNet
from fouclnet.utils import ensure_dir, get_device, save_json


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="评估 FouCLNet 模型")
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    args = parser.parse_args()

    cfg = build_config(args)
    data, reference, split = prepare_dataset(cfg)

    device = get_device(cfg.train.device)
    model = FouCLNet(cfg.system, cfg.train).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    results = {
        "in_distribution": evaluate_in_distribution(model, data.t, reference, split, cfg, device),
        "extrapolation": evaluate_extrapolation(model, data.t, reference, split, cfg, device),
    }

    # 若加载的是隔离对照模型，额外评估被硬屏蔽的连续段
    if "isolated" in Path(args.checkpoint).name.lower():
        isolated_split = create_train_val_split(
            data.t,
            split_ratio=cfg.train.split_ratio,
            seed=cfg.train.seed,
            cont_val_start=cfg.evaluation.isolated_start,
            cont_val_end=cfg.evaluation.isolated_end,
            fs=data.fs,
        )
        results["isolated_segment"] = evaluate_isolated_segment(
            model, data.t, reference, isolated_split, cfg, device
        )
    else:
        print("\n[提示] 严格隔离段指标需使用隔离模式权重评估："
              "python scripts/evaluate.py --checkpoint checkpoints/model_isolated.pth")
        results["isolated_note"] = "请用隔离模式权重评估隔离段（见 README 复现步骤）"

    outdir = ensure_dir(Path(args.outdir))
    save_json(results, outdir / "evaluation_metrics.json")
    print(f"[完成] 指标已写入 {outdir / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
