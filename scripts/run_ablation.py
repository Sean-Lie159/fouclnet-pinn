"""消融实验（A/B/C/D/E 五组），用于分离各组件的贡献。

示例::

    python scripts/run_ablation.py --main-checkpoint checkpoints/best_pinn_model_main.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_common_arguments, build_config, prepare_dataset

from fouclnet.ablation import run_ablation
from fouclnet.utils import ensure_dir, save_json


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="FouCLNet 消融实验")
    add_common_arguments(parser)
    parser.add_argument("--main-checkpoint", default=None, help="C 组使用的主模型权重路径")
    parser.add_argument("--groups", nargs="*", default=None, help="需要运行的组别，默认 A B E D C")
    args = parser.parse_args()

    cfg = build_config(args)
    data, reference, split = prepare_dataset(cfg)

    outdir = ensure_dir(Path(args.outdir))
    results = run_ablation(
        cfg,
        data,
        split,
        reference,
        groups=args.groups,
        main_checkpoint=args.main_checkpoint,
        output_dir=str(outdir),
    )
    save_json(results, outdir / "ablation_results.json")
    print(f"[完成] 消融结果已写入 {outdir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
