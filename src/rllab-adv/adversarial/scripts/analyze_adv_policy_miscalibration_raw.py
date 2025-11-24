#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于「原始（未校准）policy 分布」来度量 miscalibration，并画 reliability curve。

对每个 pickle（baseline / calib）：
  - baseline（无校准）：raw_mean == mean，动作直接从该分布采样，应该接近完美校准；
  - calib（带线性校准）：raw_mean/raw_log_std 表示原始 policy 的高斯分布，
    实际执行的动作是经过校准后的 action（仍然存在 adv_actions 里）。
  用 raw_mean/raw_log_std + 实际执行的动作来计算标准化残差 z，并画出
  (alpha_theory, alpha_emp) 的 reliability 曲线。
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
    if isinstance(args_obj, dict):
        return args_obj.get(name, default)
    return getattr(args_obj, name, default)


def collect_adv_raw_model_and_actions(data, n_traj=10, max_path_length=None, seed=0):
    """
    从结果 dict 中恢复 env / pro_policy / adv_policy，并采样若干条 rollout，
    收集：
      - 实际执行的 adversary 动作 a_exec（adv_actions）
      - 对应的「原始高斯分布」参数 raw_mean/raw_log_std（policy 内部导出）
    返回：
      actions_exec: (N, D)
      raw_means:    (N, D)
      raw_log_stds: (N, D)
    """
    args = data['args']
    env_name = get_attr(args, 'env')
    adv_fraction = float(get_attr(args, 'adv_fraction', 0.25))
    reset_noise_scale = get_attr(args, 'reset_noise_scale', None)
    path_length = int(get_attr(args, 'path_length', 1000))
    if max_path_length is None:
        max_path_length = path_length

    env = normalize(GymEnv(env_name, adv_fraction, reset_noise_scale=reset_noise_scale))

    pro_policy = data['pro_policy']
    adv_policy = data['adv_policy']

    np.random.seed(seed)

    all_actions = []
    all_raw_means = []
    all_raw_log_stds = []

    for _ in range(n_traj):
        path = rollout(env, pro_policy, max_path_length=max_path_length,
                       animated=False, adv_agent=adv_policy, test=False)
        adv_actions = path['adv_actions']          # (T, D) 实际动作（可能被校准）
        adv_infos = path['adv_agent_infos']        # dict of arrays

        raw_means = adv_infos.get('raw_mean', None)
        raw_log_stds = adv_infos.get('raw_log_std', None)
        if raw_means is None or raw_log_stds is None:
            # 向后兼容：如果老 pickle 里没有 raw_*，就退化为用 mean/log_std
            print("Warning: adv_agent_infos has no 'raw_mean'/'raw_log_std'; "
                  "fall back to 'mean'/'log_std'.")
            raw_means = adv_infos.get('mean', None)
            raw_log_stds = adv_infos.get('log_std', None)

        if raw_means is None or raw_log_stds is None:
            print("Warning: no usable Gaussian params for this trajectory; skip.")
            continue

        all_actions.append(adv_actions)
        all_raw_means.append(raw_means)
        all_raw_log_stds.append(raw_log_stds)

    if len(all_actions) == 0:
        raise RuntimeError("No valid trajectories collected (no raw_mean/log_std info).")

    actions = np.concatenate(all_actions, axis=0)
    raw_means = np.concatenate(all_raw_means, axis=0)
    raw_log_stds = np.concatenate(all_raw_log_stds, axis=0)
    return actions, raw_means, raw_log_stds


def compute_reliability_curve_mis(actions_exec, raw_means, raw_log_stds,
                                  z_grid=None, max_points=None):
    """
    使用「原始高斯」(raw_means, raw_log_stds) 作为模型分布，
    实际动作为 actions_exec，计算标准化残差：
        z = (a_exec - raw_mean) / exp(raw_log_std)
    并在一系列对称区间 [-z_k, z_k] 上计算理论/经验覆盖度。
    """
    if z_grid is None:
        z_grid = np.linspace(0.1, 2.5, num=12)

    stds = np.exp(raw_log_stds)
    z = (actions_exec - raw_means) / (stds + 1e-8)
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
                        help='output PNG path; if not set, use baseline path with suffix "_raw_miscalib.png"')
    parser.add_argument('--n_traj', type=int, default=10,
                        help='number of trajectories to sample per policy')
    parser.add_argument('--max_points', type=int, default=50000,
                        help='max number of (time_step, dim) points used for reliability (subsampled if larger)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for subsampling and env reset')
    args = parser.parse_args()

    # baseline: raw == final，高斯与动作自洽 → 预期接近完美对角线
    base_data = load_pickle(args.baseline)
    base_actions, base_raw_means, base_raw_log_stds = collect_adv_raw_model_and_actions(
        base_data, n_traj=args.n_traj, max_path_length=None, seed=args.seed
    )
    base_alpha_theory, base_alpha_emp = compute_reliability_curve_mis(
        base_actions, base_raw_means, base_raw_log_stds, max_points=args.max_points
    )

    # calibrated：raw 表示原始 policy，高斯；动作是经过校准后的实际执行动作
    calib_data = load_pickle(args.calib)
    calib_actions, calib_raw_means, calib_raw_log_stds = collect_adv_raw_model_and_actions(
        calib_data, n_traj=args.n_traj, max_path_length=None, seed=args.seed + 1
    )
    calib_alpha_theory, calib_alpha_emp = compute_reliability_curve_mis(
        calib_actions, calib_raw_means, calib_raw_log_stds, max_points=args.max_points
    )

    # 画图
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(base_alpha_theory, base_alpha_emp, 'o-', label='baseline (vs raw)')
    ax.plot(calib_alpha_theory, calib_alpha_emp, 's-', label='calib (vs raw)')
    xs = np.linspace(0.0, 1.0, 100)
    ax.plot(xs, xs, 'k--', label='perfect')

    ax.set_xlabel('Nominal coverage (theory, raw Gaussian)')
    ax.set_ylabel('Empirical coverage (using executed actions)')
    ax.set_title('Adversary miscalibration w.r.t. raw policy')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if args.out is None:
        base_root, _ = os.path.splitext(args.baseline)
        out_path = base_root + '_raw_miscalib.png'
    else:
        out_path = args.out

    print("Saving miscalibration figure to:", out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()


