#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 adversary policy 的「校准误差」并画 reliability curve（连续动作版本）。

思路（per-policy）：
  1. 从训练结果 pickle 中读取 env / adv_policy / pro_policy。
  2. 用和训练时一致的环境跑若干条 rollout，收集：
       - adversary 的动作 a
       - 对应高斯分布参数 mean / log_std
  3. 计算标准化残差 z = (a - mean) / std，并在一组置信区间水平上：
       - 理论覆盖度 alpha_theory(z) = P(|Z| <= z), Z~N(0,1)
       - 经验覆盖度 alpha_emp = 1/N * sum[ I(|z_i| <= z) ]
  4. 画出 (alpha_theory, alpha_emp) 的 reliability 曲线。

然后同时对 baseline 和 calibrated 两个 pickle 做这件事，在同一张图里对比。

注意：
  - 这里只依赖 policy 当前给出的 mean/log_std 和采样的动作，不修改任何训练脚本。
  - 「特定 iteration」可以通过传入不同的 checkpoint pickle 路径来实现
    （例如 baseline 最终 .p，calib 使用 .iter100.p / .iter200.p 等）。
"""

from __future__ import print_function
import os
import argparse
import pickle
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.sampler.utils import rollout


def load_pickle(path):
    print("Loading:", path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        print("  keys:", list(data.keys()))
    return data


def get_attr(args_obj, name, default=None):
    """兼容 args 既可能是 argparse.Namespace 也可能是 dict 的情况。"""
    if isinstance(args_obj, dict):
        return args_obj.get(name, default)
    return getattr(args_obj, name, default)


def collect_adv_gaussian_samples(data, n_traj=10, max_path_length=None, seed=0):
    """
    从结果 dict 中恢复 env / pro_policy / adv_policy，并采样若干条 rollout，
    收集 adversary 的动作和对应的 (mean, log_std)。
    返回：
        actions: (N, D)
        means:   (N, D)
        log_stds:(N, D)
    """
    args = data['args']
    env_name = get_attr(args, 'env')
    adv_fraction = float(get_attr(args, 'adv_fraction', 0.25))
    reset_noise_scale = get_attr(args, 'reset_noise_scale', None)
    path_length = int(get_attr(args, 'path_length', 1000))
    if max_path_length is None:
        max_path_length = path_length

    # 构造和训练时 adversary 用的一样的环境
    env = normalize(GymEnv(env_name, adv_fraction, reset_noise_scale=reset_noise_scale))

    pro_policy = data['pro_policy']
    adv_policy = data['adv_policy']

    rng = np.random.RandomState(seed)

    all_actions = []
    all_means = []
    all_log_stds = []

    for _ in range(n_traj):
        # 这里使用 rollout，并确保 test=False，才能得到随机采样的动作和 agent_infos
        path = rollout(env, pro_policy, max_path_length=max_path_length,
                       animated=False, adv_agent=adv_policy, test=False)
        adv_actions = path['adv_actions']          # (T, D)
        adv_infos = path['adv_agent_infos']        # dict of arrays
        means = adv_infos.get('mean', None)
        log_stds = adv_infos.get('log_std', None)

        if means is None or log_stds is None:
            print("Warning: adv_agent_infos does not contain 'mean'/'log_std'; "
                  "skip this trajectory.")
            continue

        all_actions.append(adv_actions)
        all_means.append(means)
        all_log_stds.append(log_stds)

    if len(all_actions) == 0:
        raise RuntimeError("No valid trajectories collected (no mean/log_std in adv_agent_infos).")

    actions = np.concatenate(all_actions, axis=0)
    means = np.concatenate(all_means, axis=0)
    log_stds = np.concatenate(all_log_stds, axis=0)
    return actions, means, log_stds


def compute_reliability_curve(actions, means, log_stds, z_grid=None, max_points=None):
    """
    给定动作和对应的高斯参数，计算基于标准化残差的 reliability 曲线。
    - actions, means, log_stds: 形状 (N, D)
    - z_grid: 一组阈值 z >= 0，用于定义对称置信区间 [-z, z]
    返回：
      alpha_theory: (K,) 理论覆盖度（标准正态下 P(|Z|<=z_k)）
      alpha_emp:    (K,) 经验覆盖度
    """
    if z_grid is None:
        z_grid = np.linspace(0.1, 2.5, num=12)

    stds = np.exp(log_stds)
    z = (actions - means) / (stds + 1e-8)
    abs_z = np.abs(z).reshape(-1)

    if max_points is not None and abs_z.shape[0] > max_points:
        idx = np.random.choice(abs_z.shape[0], size=max_points, replace=False)
        abs_z = abs_z[idx]

    alpha_theory = np.array([math.erf(zv / math.sqrt(2.0)) for zv in z_grid])
    alpha_emp = np.array([(abs_z <= zv).mean() for zv in z_grid])

    return alpha_theory, alpha_emp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True,
                        help='baseline RARL pickle file (no calibration)')
    parser.add_argument('--calib', type=str, required=True,
                        help='calibrated RARL pickle file')
    parser.add_argument('--out', type=str, default=None,
                        help='output PNG path; if not set, use baseline path with suffix "_calib_reliability.png"')
    parser.add_argument('--n_traj', type=int, default=10,
                        help='number of trajectories to sample per policy')
    parser.add_argument('--max_points', type=int, default=50000,
                        help='max number of (time_step, dim) points used for reliability (subsampled if larger)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for subsampling and environment (partially)')
    args = parser.parse_args()

    # baseline
    base_data = load_pickle(args.baseline)
    base_actions, base_means, base_log_stds = collect_adv_gaussian_samples(
        base_data, n_traj=args.n_traj, max_path_length=None, seed=args.seed
    )
    base_alpha_theory, base_alpha_emp = compute_reliability_curve(
        base_actions, base_means, base_log_stds, max_points=args.max_points
    )

    # calibrated
    calib_data = load_pickle(args.calib)
    calib_actions, calib_means, calib_log_stds = collect_adv_gaussian_samples(
        calib_data, n_traj=args.n_traj, max_path_length=None, seed=args.seed + 1
    )
    calib_alpha_theory, calib_alpha_emp = compute_reliability_curve(
        calib_actions, calib_means, calib_log_stds, max_points=args.max_points
    )

    # 画图
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(base_alpha_theory, base_alpha_emp, 'o-', label='baseline')
    ax.plot(calib_alpha_theory, calib_alpha_emp, 's-', label='calib')
    # 理想对角线
    xs = np.linspace(0.0, 1.0, 100)
    ax.plot(xs, xs, 'k--', label='perfect')

    ax.set_xlabel('Nominal coverage (theory, Gaussian)')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Adversary policy calibration (reliability curve)')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if args.out is None:
        base_root, _ = os.path.splitext(args.baseline)
        out_path = base_root + '_calib_reliability.png'
    else:
        out_path = args.out

    print("Saving reliability figure to:", out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()


