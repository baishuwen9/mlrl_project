#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare baseline RARL and calibrated RARL results.

Usage (inside Docker, with /results mounted):

  python plot_compare_baseline_calib.py \
    --baseline /results/RARL-env-....p \
    --calib /results/RARL_calib_....p \
    --out /results/compare_baseline_calib.png
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


def extract_curves(data, key_candidates):
    """
    data: dict from pickle
    key_candidates: list of possible keys, e.g. ['zero_test', 'const_test']

    Returns:
        mean (T,), std (T,) or (None, None) if not found
    """
    if not isinstance(data, dict):
        return None, None

    chosen_key = None
    for k in key_candidates:
        if k in data:
            chosen_key = k
            break
    if chosen_key is None:
        return None, None

    arr = data[chosen_key]
    # arr expected shape: list over experiments, each is list over iterations
    if not isinstance(arr, list) or len(arr) == 0:
        return None, None

    if not isinstance(arr[0], list):
        # single experiment, just one curve
        y = np.asarray(arr, dtype=float)
        return y, np.zeros_like(y)

    # multiple experiments: trim to common length
    lengths = [len(x) for x in arr if isinstance(x, list)]
    if len(lengths) == 0:
        return None, None
    T = min(lengths)
    ys = np.zeros((len(arr), T), dtype=float)
    for i, exp in enumerate(arr):
        ys[i, :] = np.asarray(exp[:T], dtype=float)

    mean = ys.mean(axis=0)
    std = ys.std(axis=0)
    return mean, std


def plot_pair(ax, x, base_mean, base_std, calib_mean, calib_std, title):
    has_any = False
    if base_mean is not None:
        ax.plot(x, base_mean, label='baseline', color='C0')
        if base_std is not None:
            ax.fill_between(x, base_mean - base_std, base_mean + base_std,
                            color='C0', alpha=0.2)
        has_any = True
    if calib_mean is not None:
        ax.plot(x, calib_mean, label='calib', color='C1')
        if calib_std is not None:
            ax.fill_between(x, calib_mean - calib_std, calib_mean + calib_std,
                            color='C1', alpha=0.2)
        has_any = True
    if has_any:
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average return')
        ax.legend(loc='best')
    else:
        ax.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True,
                        help='baseline RARL pickle file (train_adversary.py)')
    parser.add_argument('--calib', type=str, required=True,
                        help='calibrated RARL pickle file (rarl_with_calib.py)')
    parser.add_argument('--out', type=str, default=None,
                        help='output PNG path; if not set, use baseline path with suffix')
    args = parser.parse_args()

    base_data = load_pickle(args.baseline)
    calib_data = load_pickle(args.calib)

    # Define test groups: (title, baseline_keys, calib_keys)
    groups = [
        ('Zero / Const Adv', ['zero_test'], ['const_test']),
        ('Random Adv', ['rand_test'], ['rand_test']),
        ('Step Adv', ['step_test'], ['step_test']),
        ('Rand Step Adv', ['rand_step_test'], ['rand_step_test']),
        ('Learnt Adv', ['adv_test'], ['adv_test']),
    ]

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(8, 3 * n_groups), sharex=True)
    # axes may be a single Axes if n_groups == 1
    if not isinstance(axes, (list, np.ndarray)):
        axes_list = [axes]
    else:
        axes_list = list(axes)

    max_T = 0
    curves = []
    for title, b_keys, c_keys in groups:
        b_mean, b_std = extract_curves(base_data, b_keys)
        c_mean, c_std = extract_curves(calib_data, c_keys)
        T = 0
        if b_mean is not None:
            T = max(T, len(b_mean))
        if c_mean is not None:
            T = max(T, len(c_mean))
        max_T = max(max_T, T)
        curves.append((title, b_mean, b_std, c_mean, c_std, T))

    if max_T == 0:
        print("No usable curves found in either file.")
        return

    x = np.arange(max_T)
    for ax, (title, b_mean, b_std, c_mean, c_std, T) in zip(axes_list, curves):
        if T == 0:
            ax.set_visible(False)
            continue
        # trim to this curve's length
        xx = x[:T]
        bm = b_mean[:T] if b_mean is not None else None
        bs = b_std[:T] if b_std is not None else None
        cm = c_mean[:T] if c_mean is not None else None
        cs = c_std[:T] if c_std is not None else None
        plot_pair(ax, xx, bm, bs, cm, cs, title)

    plt.tight_layout()

    if args.out is None:
        base_root, _ = os.path.splitext(args.baseline)
        out_path = base_root + '_vs_calib.png'
    else:
        out_path = args.out

    print("Saving figure to:", out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()


