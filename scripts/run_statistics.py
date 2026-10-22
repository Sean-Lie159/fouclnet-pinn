"""统计稳健性实验：C 组与 E 组各 n 次独立训练 + 配对检验 + 贝叶斯因子。

用于检验“某组件是否真的带来稳定提升”，避免用单次运行的结论下判断。

示例::

    python scripts/run_statistics.py --n-runs 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_common_arguments, build_config, prepare_dataset

from fouclnet.statistics import format_statistics_table, run_repeated_trials
from fouclnet.utils import ensure_dir, save_json


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="C/E 组统计稳健性实验")
    add_common_arguments(parser)
    parser.add_argument("--n-runs", type=int, default=15, help="每组独立训练次数（示例取 15）")
    args = parser.parse_args()

    cfg = build_config(args)
    data, reference, split = prepare_dataset(cfg)

    outdir = ensure_dir(Path(args.outdir))
    result = run_repeated_trials(
        cfg,
        data,
        split,
        reference,
        n_runs=args.n_runs,
        checkpoint_dir=str(outdir / "checkpoints"),
    )

    print("\n" + format_statistics_table(result["statistics"]))
    save_json(result, outdir / "statistics_results.json")
    print(f"\n[完成] 统计结果已写入 {outdir / 'statistics_results.json'}")


if __name__ == "__main__":
    main()
