"""PSD 高频段能量占比计算（频谱分析入口）。

示例::

    python scripts/run_hf_energy.py --checkpoint checkpoints/best_pinn_model_main.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import add_common_arguments, build_config, prepare_dataset

from fouclnet.evaluate import predict
from fouclnet.metrics import hf_energy_ratio
from fouclnet.model import FouCLNet
from fouclnet.plotting import plot_psd_hf_detail
from fouclnet.utils import ensure_dir, get_device, save_json


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="高频段能量占比计算")
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    args = parser.parse_args()

    cfg = build_config(args)
    data, reference, split = prepare_dataset(cfg)

    device = get_device(cfg.train.device)
    model = FouCLNet(cfg.system, cfg.train).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    x_pred = predict(model, data.t, cfg, device)

    ref_stats = hf_energy_ratio(reference, cfg.system.fs, cfg.evaluation.hf_cutoff, cfg.evaluation.welch_nperseg)
    pred_stats = hf_energy_ratio(x_pred, cfg.system.fs, cfg.evaluation.hf_cutoff, cfg.evaluation.welch_nperseg)

    print("\n========== PSD 高频段能量占比（显式可复现） ==========")
    print(f"Welch: nperseg = {ref_stats['nperseg']}, noverlap = {ref_stats['nperseg'] // 2}, "
          f"window = hann, detrend = linear, scaling = density")
    print(f"频率分辨率 df = {ref_stats['df']:.6f} Hz，有效范围 [{0:.6f}, {ref_stats['f_max']:.3f}] Hz，"
          f"分段数 = {ref_stats['n_segments']}")
    print(f"高频截止频率 f_cutoff = {cfg.evaluation.hf_cutoff} Hz")
    print(f"RK4 参考解 : 高频能量 {ref_stats['hf_energy']:.6e} rad^2，占比 {ref_stats['hf_ratio_percent']:.4f}%")
    print(f"FouCLNet   : 高频能量 {pred_stats['hf_energy']:.6e} rad^2，占比 {pred_stats['hf_ratio_percent']:.4f}%")
    print(f"占比差异   : {abs(ref_stats['hf_ratio_percent'] - pred_stats['hf_ratio_percent']):.4f}%")

    outdir = ensure_dir(Path(args.outdir))
    plot_psd_hf_detail(cfg, reference, x_pred, outdir / "figures" / "psd_hf_detail.png")
    save_json({"reference": ref_stats, "prediction": pred_stats}, outdir / "hf_energy_ratio.json")
    print(f"[完成] 结果已写入 {outdir / 'hf_energy_ratio.json'}")


if __name__ == "__main__":
    main()
