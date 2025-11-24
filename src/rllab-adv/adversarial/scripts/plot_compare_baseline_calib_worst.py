#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
画 baseline RARL 和 calibrated RARL 的「最差表现」（worst-case）曲线，
而不是平均值：对每一轮迭代，取所有实验中的最小 return。

用法（在 Docker 里，挂载 /results 后）示例：

  python plot_compare_baseline_calib_worst.py \
    --baseline /results/RARL-env-....p \
    --calib /results/RARL_calib_....p \
    --out /results/compare_baseline_calib_worst.png
"""

from __future__ import print_function
import os
import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_pickle(path):
    print("Loading:", path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        print("  keys:", list(data.keys()))
    return data


def extract_worst_curve(data, key_candidates):
    """
    data: dict from pickle
    key_candidates: list of possible keys, e.g. ['zero_test', 'const_test']

    返回：
        worst (T,) 或 None
    其中 worst[t] 是第 t 轮迭代中所有实验的「最小」return。
    """
    if not isinstance(data, dict):
        return None

    chosen_key = None
    for k in key_candidates:
        if k in data:
            chosen_key = k
            break
    if chosen_key is None:
        return None

    arr = data[chosen_key]
    # 期望结构：list over experiments, each is list over iterations
    if not isinstance(arr, list) or len(arr) == 0:
        return None

    # 单个实验：直接当作一条曲线
    if not isinstance(arr[0], list):
        y = np.asarray(arr, dtype=float)
        return y

    # 多个实验：按最短长度截断，然后按维度取最小值
    lengths = [len(x) for x in arr if isinstance(x, list) and len(x) > 0]
    if len(lengths) == 0:
        return None
    T = min(lengths)
    ys = np.zeros((len(arr), T), dtype=float)
    for i, exp in enumerate(arr):
        ys[i, :] = np.asarray(exp[:T], dtype=float)

    worst = ys.min(axis=0)
    return worst


def plot_pair_worst(ax, x, base_worst, calib_worst, title):
    has_any = False
    if base_worst is not None:
        ax.plot(x, base_worst, label='baseline (worst)', color='C0')
        has_any = True
    if calib_worst is not None:
        ax.plot(x, calib_worst, label='calib (worst)', color='C1')
        has_any = True
    if has_any:
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Worst-case return')
        ax.legend(loc='best')
    else:
        ax.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True,
                        help='baseline RARL pickle file (train_adversary.py)')
    parser.add_argument('--calib', type=str, required=True,
                        help='calibrated RARL pickle file (rarl_with_calib.py or train_adversary.py with calib)')
    parser.add_argument('--out', type=str, default=None,
                        help='output PNG path; if not set, use baseline path with suffix "_worst.png"')
    args = parser.parse_args()

    base_data = load_pickle(args.baseline)
    calib_data = load_pickle(args.calib)

    # 各测试场景：注意 zero/const 的键名差异与平均版脚本保持一致
    groups = [
        # baseline: 键名 'zero_test'
        # calib: 可能是 'const_test'（来自 rarl_with_calib.py），
        #        也可能是 'zero_test'（来自带 TRPO 校准的 train_adversary.py）
        ('Zero / Const Adv', ['zero_test'], ['const_test', 'zero_test']),
        ('Random Adv', ['rand_test'], ['rand_test']),
        ('Step Adv', ['step_test'], ['step_test']),
        ('Rand Step Adv', ['rand_step_test'], ['rand_step_test']),
        ('Learnt Adv', ['adv_test'], ['adv_test']),
    ]

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(8, 3 * n_groups), sharex=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes_list = [axes]
    else:
        axes_list = list(axes)

    max_T = 0
    curves = []
    for title, b_keys, c_keys in groups:
        b_worst = extract_worst_curve(base_data, b_keys)
        c_worst = extract_worst_curve(calib_data, c_keys)

        T = 0
        if b_worst is not None:
            T = max(T, len(b_worst))
        if c_worst is not None:
            T = max(T, len(c_worst))
        max_T = max(max_T, T)
        curves.append((title, b_worst, c_worst, T))

    if max_T == 0:
        print("No usable curves found in either file.")
        return

    x = np.arange(max_T)
    for ax, (title, b_worst, c_worst, T) in zip(axes_list, curves):
        if T == 0:
            ax.set_visible(False)
            continue
        xx = x[:T]
        bw = b_worst[:T] if b_worst is not None else None
        cw = c_worst[:T] if c_worst is not None else None
        plot_pair_worst(ax, xx, bw, cw, title)

    plt.tight_layout()

    if args.out is None:
        base_root, _ = os.path.splitext(args.baseline)
        out_path = base_root + '_vs_calib_worst.png'
    else:
        out_path = args.out

    print("Saving worst-case figure to:", out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()


