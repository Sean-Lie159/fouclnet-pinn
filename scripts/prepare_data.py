"""把原始记录整理为规范的单列 CSV 数据文件。

输入
----
单列（或多列）文本/CSV 形式的地面 Z 分量速度记录，采样率需为 200 Hz。
若使用 MiniSEED 等二进制格式，请先用 obspy/simplemseed 导出为文本。

处理步骤
--------
1. 读取原始序列（可指定列号）；
2. 可选：去直流、Butterworth 带通（默认关闭，仅在需要时启用）；
3. 可选：按 ``--decimate`` 抽取（保持 200 Hz 时不需要）；
4. 写出单列 CSV，每行一个浮点数（科学计数法）。

示例::

    python scripts/prepare_data.py --input raw/record.txt --output data/series.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from _common import REPO_ROOT  # noqa: F401  (用于路径引导)


def read_series(path: Path, column: int | None = None) -> np.ndarray:
    """读取原始序列（自动识别分隔符与单列/多列布局）。"""
    data = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    data = np.atleast_1d(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.shape[0] == 1 and data.shape[1] > 1:
        data = data.T
    if column is None:
        if data.shape[1] > 1:
            raise ValueError("输入含多列，请用 --column 指定要使用的列（0 起）。")
        column = 0
    return np.asarray(data[:, column], dtype=float)



def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="准备地面运动数据文件")
    parser.add_argument("--input", required=True, help="原始记录路径")
    parser.add_argument("--output", default="data/series.csv", help="输出 CSV 路径")
    parser.add_argument("--column", type=int, default=None, help="使用的列序号（0 起）")
    parser.add_argument("--fs", type=float, default=200.0, help="原始采样率 [Hz]")
    parser.add_argument("--decimate", type=int, default=1, help="抽取因子（1 表示不抽取）")
    parser.add_argument("--bandpass", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
                        help="可选 Butterworth 带通范围 [Hz]")
    parser.add_argument("--order", type=int, default=4, help="Butterworth 阶数")
    parser.add_argument("--demean", action="store_true", help="去直流")
    args = parser.parse_args()

    series = read_series(Path(args.input), args.column)
    print(f"[数据] 读入 {series.size:,} 点")

    if args.demean:
        series = series - series.mean()

    if args.bandpass:
        low, high = args.bandpass
        sos = butter(args.order, [low, high], btype="bandpass", fs=args.fs, output="sos")
        series = sosfiltfilt(sos, series)
        print(f"[数据] 已施加 Butterworth 带通 {low}-{high} Hz（{args.order} 阶）")

    if args.decimate > 1:
        series = series[:: args.decimate]
        print(f"[数据] 抽取因子 {args.decimate}，剩余 {series.size:,} 点")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, series, fmt="%.9e")
    print(f"[数据] 已写出 {output}（{series.size:,} 点）")


if __name__ == "__main__":
    main()
