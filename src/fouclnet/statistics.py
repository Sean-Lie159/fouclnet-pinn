"""统计稳健性分析：n 次独立训练的配对检验与 JZS 贝叶斯因子。

用于检验结论的统计稳健性：

* 对 C 组（完整方法）与 E 组（FFN+HAC、lambda_PDE = 0）各做 n 次独立训练；
* 报告均值 ± 标准差、配对 t 检验、Wilcoxon 符号秩检验、Cohen's d、
  Levene 与 Bartlett 方差齐性检验，以及 JZS 先验下的贝叶斯因子 BF10/BF01。

贝叶斯因子实现说明
------------------
原始工作脚本 ``Bayers.py`` 用数值积分在辅助变量 g 上计算 H1 边际似然
（Rouder et al., 2009 的 JZS 先验，Cauchy 尺度 r）。本模块把该逻辑
内联实现，从而去除对 ``pingouin`` 的可选依赖，结果与原脚本一致。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import integrate, stats


def bayes_factor_ttest(t_stat: float, n: int, r: float = 0.707):
    """配对 t 检验的 JZS 贝叶斯因子。

    Parameters
    ----------
    t_stat:
        配对 t 统计量。
    n:
        配对样本量。
    r:
        Cauchy 先验尺度（0.707 对应中等效应量）。

    Returns
    -------
    tuple
        ``(bf10, bf01, integration_error)``。
    """
    nu = n - 1
    if not np.isfinite(t_stat) or n < 2:
        return float("nan"), float("nan"), float("nan")

    def integrand(g: float) -> float:

        if g <= 0:
            return 0.0
        # JZS 先验密度 p(g) = (2*pi)^{-1/2} * g^{-3/2} * exp(-1/(2g))
        prior = (2 * np.pi) ** (-0.5) * g ** (-1.5) * np.exp(-0.5 / g)
        v = n * g * (r ** 2)
        term = (1 + v) ** (-0.5) * (1 + t_stat ** 2 / (nu * (1 + v))) ** (-(nu + 1) / 2)
        return term * prior

    marginal_h1, err = integrate.quad(integrand, 0, np.inf, limit=200)
    h0_kernel = (1 + t_stat ** 2 / nu) ** (-(nu + 1) / 2)
    bf10 = marginal_h1 / h0_kernel
    return bf10, 1.0 / bf10, err


def paired_statistics(group_c: Sequence[float], group_e: Sequence[float]) -> Dict[str, float]:
    """C 组与 E 组的配对统计检验汇总。"""
    c_arr = np.asarray(group_c, dtype=float)
    e_arr = np.asarray(group_e, dtype=float)
    if c_arr.size != e_arr.size:
        raise ValueError("配对检验要求两组样本量一致。")

    t_stat, p_value = stats.ttest_rel(c_arr, e_arr)
    _, p_wilcoxon = stats.wilcoxon(c_arr, e_arr)
    diff = c_arr - e_arr
    cohens_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))
    stat_levene, p_levene = stats.levene(c_arr, e_arr)
    stat_bartlett, p_bartlett = stats.bartlett(c_arr, e_arr)
    bf10, bf01, bf_err = bayes_factor_ttest(float(t_stat), int(c_arr.size))

    return {
        "n": int(c_arr.size),
        "c_mean": float(c_arr.mean()),
        "c_std": float(c_arr.std(ddof=1)),
        "e_mean": float(e_arr.mean()),
        "e_std": float(e_arr.std(ddof=1)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "p_wilcoxon": float(p_wilcoxon),
        "cohens_d": cohens_d,
        "levene_W": float(stat_levene),
        "levene_p": float(p_levene),
        "bartlett_K2": float(stat_bartlett),
        "bartlett_p": float(p_bartlett),
        "variance_ratio_E_over_C": float(np.var(e_arr, ddof=1) / (np.var(c_arr, ddof=1) + 1e-12)),
        "BF10": float(bf10),
        "BF01": float(bf01),
        "bf_integration_error": float(bf_err),
    }


def format_statistics_table(stats_dict: Dict[str, float]) -> str:
    """把统计结果格式化为文本表格。"""
    lines = [
        "=" * 60,
        f"统计结果汇总（n = {stats_dict['n']}）",
        "=" * 60,
        f"C 组 (完整方法)   : {stats_dict['c_mean']:.6f} ± {stats_dict['c_std']:.6f}",
        f"E 组 (无 PDE)     : {stats_dict['e_mean']:.6f} ± {stats_dict['e_std']:.6f}",
        f"配对 t 检验      : t = {stats_dict['t_stat']:.4f}, p = {stats_dict['p_value']:.4e}",
        f"Wilcoxon 检验    : p = {stats_dict['p_wilcoxon']:.4e}",
        f"Cohen's d (paired): {stats_dict['cohens_d']:.4f}",
        f"Levene 方差检验  : W = {stats_dict['levene_W']:.4f}, p = {stats_dict['levene_p']:.4e}",
        f"Bartlett 方差检验: K² = {stats_dict['bartlett_K2']:.4f}, p = {stats_dict['bartlett_p']:.4e}",
        f"方差比 (E/C)     : {stats_dict['variance_ratio_E_over_C']:.1f}x",
        f"贝叶斯因子 BF10  : {stats_dict['BF10']:.4f}  (BF01 = {stats_dict['BF01']:.4f})",
    ]
    return "\n".join(lines)


def run_repeated_trials(
    cfg,
    data,
    split,
    reference,
    n_runs: int = 15,
    seeds: Optional[Sequence[int]] = None,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, object]:
    """对 C 组（完整方法）与 E 组（无 PDE）各做 n 次独立训练。

    Parameters
    ----------
    cfg:
        完整配置；``cfg.train.seed`` 会被各次试验的种子覆盖。
    data, split, reference:
        地面运动数据、划分与 RK4 参考解。
    n_runs:
        每组的独立训练次数（示例取 15）。
    seeds:
        随机种子列表；``None`` 时使用内置的 20 个种子取前 n 个。
    checkpoint_dir:
        若给出，则每次试验的权重会保存到该目录。

    Returns
    -------
    dict
        含 ``group_c``、``group_e`` 分数列表与统计结果。
    """
    import copy

    import torch

    from .loss import HybridLoss
    from .model import FouCLNet
    from .train import train_model
    from .utils import get_device, set_seed

    default_seeds = [42, 123, 456, 789, 2024, 55, 66, 77, 88, 99,
                     111, 222, 333, 444, 555, 666, 777, 888, 999, 1010]
    seeds = list(seeds or default_seeds[:n_runs])
    device = get_device(cfg.train.device)

    group_c: list = []
    group_e: list = []

    for i, seed in enumerate(seeds):
        print(f"\n--- 重复试验 {i + 1}/{len(seeds)}，seed = {seed} ---")

        # C 组：完整方法
        cfg_c = copy.deepcopy(cfg)
        cfg_c.train.seed = seed
        set_seed(seed)
        model_c = FouCLNet(cfg.system, cfg_c.train).to(device)
        loss_c = HybridLoss(cfg.system, cfg_c.train)
        path_c = f"{checkpoint_dir}/group_c_seed{seed}.pth" if checkpoint_dir else "checkpoints/group_c_tmp.pth"
        model_c, _ = train_model(cfg_c, data, split, reference, model_c, loss_c, path_c, verbose=False)
        r_c = _validation_correlation(model_c, data, split, reference, cfg, device)
        group_c.append(r_c)
        del model_c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # E 组：FFN + HAC，关闭 PDE 残差项
        cfg_e = copy.deepcopy(cfg)
        cfg_e.train.seed = seed
        cfg_e.train.w_pde_max = 0.0
        cfg_e.train.pde_start_epoch = cfg.train.max_epochs + 1
        cfg_e.train.pde_full_epoch = cfg.train.max_epochs + 2
        set_seed(seed)
        model_e = FouCLNet(cfg.system, cfg_e.train).to(device)
        loss_e = HybridLoss(cfg.system, cfg_e.train)
        path_e = f"{checkpoint_dir}/group_e_seed{seed}.pth" if checkpoint_dir else "checkpoints/group_e_tmp.pth"
        model_e, _ = train_model(cfg_e, data, split, reference, model_e, loss_e, path_e, verbose=False)
        r_e = _validation_correlation(model_e, data, split, reference, cfg, device)
        group_e.append(r_e)
        del model_e
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stats_dict = paired_statistics(group_c, group_e)
    return {"group_c": group_c, "group_e": group_e, "statistics": stats_dict}


def _validation_correlation(model, data, split, reference, cfg, device) -> float:
    """返回离散验证集上的相关系数。"""
    from .evaluate import evaluate_indices

    metrics = evaluate_indices(model, data.t, reference, split.val_idx, cfg, device)
    return float(metrics["R"])
