"""训练模型：完整模式（默认）或严格隔离对照模式。

示例::

    python scripts/train.py --config configs/default.yaml --data-csv data/series.csv
    python scripts/train.py --config configs/default.yaml --data-csv data/series.csv \
        --mode isolated
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import REPO_ROOT, add_common_arguments, build_config, prepare_dataset

from fouclnet.train import train_model
from fouclnet.utils import ensure_dir, save_json


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="训练 FouCLNet")
    add_common_arguments(parser)
    parser.add_argument(
        "--mode",
        choices=["full", "isolated"],
        default="full",
        help="full = 训练域内不隔离任何连续段；isolated = 严格隔离 evaluation.isolated_start~end 段",
    )
    parser.add_argument("--checkpoint", default=None, help="权重保存路径")
    args = parser.parse_args()

    cfg = build_config(args)
    data, reference, split = prepare_dataset(cfg)

    if args.mode == "isolated":
        from fouclnet.data import create_train_val_split

        split = create_train_val_split(
            data.t,
            split_ratio=cfg.train.split_ratio,
            seed=cfg.train.seed,
            cont_val_start=cfg.evaluation.isolated_start,
            cont_val_end=cfg.evaluation.isolated_end,
            fs=data.fs,
        )

    outdir = ensure_dir(Path(args.outdir))
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        REPO_ROOT / "checkpoints" / f"model_{args.mode}.pth"
    )

    model, history = train_model(
        cfg,
        data,
        split,
        reference,
        model_save_path=checkpoint,
    )

    history_path = outdir / f"training_history_{args.mode}.json"
    save_json(history, history_path)
    print(f"[完成] 权重: {checkpoint}")
    print(f"[完成] 训练历史: {history_path}")


if __name__ == "__main__":
    main()
