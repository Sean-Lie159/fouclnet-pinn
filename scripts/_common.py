"""命令行脚本的公共工具：路径引导、配置解析与数据装载。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

# 允许在未安装包的情况下直接运行脚本（把 src/ 加入模块搜索路径）
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fouclnet.config import Config, load_config  # noqa: E402
from fouclnet.data import SeriesData, create_train_val_split, load_series  # noqa: E402
from fouclnet.reference import solve_reference  # noqa: E402
from fouclnet.utils import resolve_path  # noqa: E402


def add_common_arguments(parser) -> None:
    """为解析器添加公共命令行参数。"""
    parser.add_argument("--config", default=None, help="配置文件路径（YAML/JSON）")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="覆盖配置项，可重复指定，例如 --set train.max_epochs=1000",
    )
    parser.add_argument("--data-csv", default=None, help="地面运动 CSV 路径（覆盖配置）")
    parser.add_argument("--outdir", default="results", help="结果输出目录")


def build_config(args) -> Config:
    """根据命令行参数构造配置。"""
    cfg = load_config(args.config, args.overrides)
    if getattr(args, "data_csv", None):
        cfg.data.csv_path = args.data_csv
    return cfg


def prepare_dataset(cfg: Config, with_reference: bool = True):
    """装载地面运动数据、（可选）RK4 参考解与数据划分。

    Returns
    -------
    tuple
        ``(data, reference, split)``；``with_reference=False`` 时
        ``reference`` 为 ``None``。
    """
    csv_path = resolve_path(cfg.data.csv_path, REPO_ROOT)
    data = load_series(csv_path, cfg.system, cfg.data, use_fraction=cfg.data.use_fraction)


    reference = solve_reference(data.t, data.forcing, cfg.system) if with_reference else None

    split = create_train_val_split(
        data.t,
        split_ratio=cfg.train.split_ratio,
        seed=cfg.train.seed,
        fs=data.fs,
    )
    return data, reference, split
