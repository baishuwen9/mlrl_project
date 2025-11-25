#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 Platt 校准层对 adversary Gaussian policy 的 calibration 影响，
画出「未校准 vs Platt 校准后」的 reliability curve，并与 baseline（无校准）对比。

用法示例：

  python analyze_platt_calibration_reliability.py \
    --baseline /results/RARL-env-InvertedDoublePendulumAdv-v1_...baseline....p \
    --calib    /results/RARL-env-Platt-InvertedDoublePendulumAdv-v1_...calib....p \
    --out      /results/compare_platt_reliability.png

说明：
  - baseline 文件来自原始 train_adversary.py（没有 Platt 校准层）；
  - calib 文件来自 train_adversary_platt.py（包含 'platt_state'）；
  - 对 baseline：只画「raw Gaussian」的 reliability 曲线；
  - 对 calib：在同一批 on-policy 数据上，同时画
        * raw Gaussian (mu, log_std)
        * Platt 校准后的 Gaussian (mu_c, log_std_c)
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

import torch
import torch.nn as nn

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.sampler.utils import rollout


class PlattCalibrator(nn.Module):
    """
    与 train_adversary_platt.py 中相同的 per-dim 线性校准层：

        mu_c = w * mu + b
        log_std_c = log_std + log(|w|)
    """

    def __init__(self, action_dim):
        super(PlattCalibrator, self).__init__()
        self.w = nn.Parameter(torch.ones(action_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, mu_raw, logstd_raw):
        # mu_raw, logstd_raw: (N, D)
        w = self.w.view(1, -1)
        b = self.b.view(1, -1)
        mu_c = mu_raw * w + b
        logstd_c = logstd_raw + torch.log(torch.abs(w) + 1e-6)
        return mu_c, logstd_c


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


def collect_adv_gaussian_samples(data, n_traj=10, max_path_length=None, seed=0):
    """
    从结果 dict 中恢复 env / pro_policy / adv_policy，并采样若干条 rollout，
    收集 adversary 的动作和对应的 (mean, log_std)。

    返回：
      actions: (N, D)   实际动作（来自当前 adv_policy，高斯采样）
      means:   (N, D)   policy 返回的 mean
      logstds: (N, D)   policy 返回的 log_std
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
    all_means = []
    all_log_stds = []

    for _ in range(n_traj):
        path = rollout(env, pro_policy, max_path_length=max_path_length,
                       animated=False, adv_agent=adv_policy, test=False)
        adv_actions = path['adv_actions']
        adv_infos = path['adv_agent_infos']
        means = adv_infos.get('mean', None)
        log_stds = adv_infos.get('log_std', None)

        if means is None or log_stds is None:
            print("Warning: adv_agent_infos lacks mean/log_std; skip one traj.")
            continue

        all_actions.append(adv_actions)
        all_means.append(means)
        all_log_stds.append(log_stds)

    if len(all_actions) == 0:
        raise RuntimeError("No valid trajectories collected (no mean/log_std).")

    actions = np.concatenate(all_actions, axis=0)
    means = np.concatenate(all_means, axis=0)
    log_stds = np.concatenate(all_log_stds, axis=0)
    return actions, means, log_stds


def compute_reliability_curve(actions, means, log_stds, z_grid=None, max_points=None):
    """
    通用的 reliability 计算：
      - actions, means, log_stds: (N, D)
      - z = (a - mean) / std
      - z_grid: 一系列阈值 z >= 0，用于定义 |z| <= z 的对称区间。

    返回：
      alpha_theory: (K,)  理论覆盖度（标准正态）
      alpha_emp:    (K,)  经验覆盖度
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
                        help='baseline RARL pickle file (train_adversary.py, no Platt)')
    parser.add_argument('--calib', type=str, required=True,
                        help='Platt-calibrated RARL pickle file (train_adversary_platt.py)')
    parser.add_argument('--out', type=str, default=None,
                        help='output PNG path; if not set, use baseline root with suffix "_platt_reliability.png"')
    parser.add_argument('--n_traj', type=int, default=10,
                        help='number of trajectories to sample per policy')
    parser.add_argument('--max_points', type=int, default=50000,
                        help='max number of (time_step, dim) points used for reliability (subsampled if larger)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for subsampling and env reset')
    args = parser.parse_args()

    # 1. baseline: 只计算 raw Gaussian 的 reliability
    base_data = load_pickle(args.baseline)
    base_actions, base_means, base_logstds = collect_adv_gaussian_samples(
        base_data, n_traj=args.n_traj, max_path_length=None, seed=args.seed
    )
    base_alpha_theory, base_alpha_emp = compute_reliability_curve(
        base_actions, base_means, base_logstds, max_points=args.max_points
    )

    # 2. calib 文件：需要加载 PlattCalibrator，再在同一批数据上计算「Platt 校准后」的 reliability
    calib_data = load_pickle(args.calib)
    if 'platt_state' not in calib_data:
        raise RuntimeError("calib pickle does not contain 'platt_state'; "
                           "please use a file saved by train_adversary_platt.py.")

    # 用 calib 的 args/env 重新 roll out（on-policy w.r.t calibrated run 的 policy）
    calib_actions, calib_means_raw, calib_logstds_raw = collect_adv_gaussian_samples(
        calib_data, n_traj=args.n_traj, max_path_length=None, seed=args.seed + 1
    )

    # 构造并加载 PlattCalibrator
    args_c = calib_data['args']
    env_name_c = get_attr(args_c, 'env')
    adv_fraction_c = float(get_attr(args_c, 'adv_fraction', 0.25))
    reset_noise_scale_c = get_attr(args_c, 'reset_noise_scale', None)

    env_c = normalize(GymEnv(env_name_c, adv_fraction_c, reset_noise_scale=reset_noise_scale_c))
    try:
        action_dim = int(env_c.adv_action_space.flat_dim)
    except Exception:
        try:
            action_dim = int(env_c._wrapped_env.adv_action_space.flat_dim)
        except Exception:
            action_dim = calib_means_raw.shape[1]

    platt = PlattCalibrator(action_dim)
    platt.load_state_dict(calib_data['platt_state'])
    platt.eval()

    with torch.no_grad():
        mu_raw_t = torch.tensor(calib_means_raw, dtype=torch.float32)
        logstd_raw_t = torch.tensor(calib_logstds_raw, dtype=torch.float32)
        mu_c_t, logstd_c_t = platt(mu_raw_t, logstd_raw_t)
        mu_c = mu_c_t.cpu().numpy()
        logstd_c = logstd_c_t.cpu().numpy()

    calib_alpha_theory_calib, calib_alpha_emp_calib = compute_reliability_curve(
        calib_actions, mu_c, logstd_c, max_points=args.max_points
    )

    # 3. 画图：baseline raw vs calib Platt（只保留“无校准”和“有校准”两条曲线）
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(base_alpha_theory, base_alpha_emp, 'o-', label='baseline (raw)')
    ax.plot(calib_alpha_theory_calib, calib_alpha_emp_calib, 's-', label='calibrated (Platt)')

    xs = np.linspace(0.0, 1.0, 100)
    ax.plot(xs, xs, 'k--', label='perfect')

    ax.set_xlabel('Nominal coverage (Gaussian)')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Adversary policy reliability: raw vs Platt calibration')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if args.out is None:
        base_root, _ = os.path.splitext(args.baseline)
        out_path = base_root + '_platt_reliability.png'
    else:
        out_path = args.out

    print("Saving Platt calibration reliability figure to:", out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()


