#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在同一个 Platt 训练得到的 adversary 上，比较「未校准 Gaussian」和「Platt 校准后 Gaussian」
的 reliability curve（只用一个 .p 文件，不需要 baseline 文件）。

用法示例：

  python analyze_platt_reliability_single.py \
    --calib /results/RARL-env-Platt-InvertedDoublePendulumAdv-v1_...iterXXX....p \
    --out   /results/platt_reliability_iterXXX.png

说明：
  - calib 文件来自 train_adversary_platt.py（必须包含 'platt_state'）；
  - 我们用同一批 on-policy rollout 数据：
        actions ~ 当前 adv_policy（未显式使用 Platt 采样）
    在这批数据上：
        * 用 (mean_raw, log_std_raw) 计算「未校准」的 reliability；
        * 用 Platt(mean_raw, log_std_raw) 得到 (mu_c, log_std_c) 后再计算「校准后」的 reliability；
  - 这样可以直接对比 calibration 之前和之后的 calibration error。
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
    从 calib 结果 dict 中恢复 env / pro_policy / adv_policy，
    采样若干条 rollout，收集：
      - actions: 实际执行动作（未显式使用 Platt）
      - means:   Gaussian policy 返回的 mean
      - logstds: Gaussian policy 返回的 log_std
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
    标准化残差 z = (a - mean) / std 的 reliability curve。
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
    parser.add_argument('--calib', type=str, required=True,
                        help='Platt-calibrated RARL pickle (train_adversary_platt.py, with platt_state)')
    parser.add_argument('--out', type=str, default=None,
                        help='output PNG path; default: calib root + "_platt_single.png"')
    parser.add_argument('--n_traj', type=int, default=10,
                        help='number of trajectories to sample')
    parser.add_argument('--max_points', type=int, default=50000,
                        help='max number of (time_step, dim) points used for reliability')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for rollout and subsampling')
    args = parser.parse_args()

    data = load_pickle(args.calib)
    if 'platt_state' not in data:
        raise RuntimeError("Given calib pickle does not contain 'platt_state'. "
                           "Please pass a file produced by train_adversary_platt.py.")

    # 收集一批 on-policy 样本
    actions, means_raw, logstds_raw = collect_adv_gaussian_samples(
        data, n_traj=args.n_traj, max_path_length=None, seed=args.seed
    )

    # 1) 未校准（raw Gaussian）的 reliability
    alpha_theory_raw, alpha_emp_raw = compute_reliability_curve(
        actions, means_raw, logstds_raw, max_points=args.max_points
    )

    # 2) 加载 Platt 校准层，对 (mean_raw, logstd_raw) 做变换后再算一次 reliability
    args_c = data['args']
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
            action_dim = means_raw.shape[1]

    platt = PlattCalibrator(action_dim)
    platt.load_state_dict(data['platt_state'])
    platt.eval()

    with torch.no_grad():
        mu_raw_t = torch.tensor(means_raw, dtype=torch.float32)
        logstd_raw_t = torch.tensor(logstds_raw, dtype=torch.float32)
        mu_c_t, logstd_c_t = platt(mu_raw_t, logstd_raw_t)
        mu_c = mu_c_t.cpu().numpy()
        logstd_c = logstd_c_t.cpu().numpy()

    alpha_theory_calib, alpha_emp_calib = compute_reliability_curve(
        actions, mu_c, logstd_c, max_points=args.max_points
    )

    # 画图：同一条 policy 上的 raw vs Platt
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(alpha_theory_raw, alpha_emp_raw, 'o-', label='raw Gaussian')
    ax.plot(alpha_theory_calib, alpha_emp_calib, 's-', label='Platt calibrated')

    xs = np.linspace(0.0, 1.0, 100)
    ax.plot(xs, xs, 'k--', label='perfect')

    ax.set_xlabel('Nominal coverage (Gaussian)')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Adversary reliability: raw vs Platt (single run)')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if args.out is None:
        calib_root, _ = os.path.splitext(args.calib)
        out_path = calib_root + '_platt_single.png'
    else:
        out_path = args.out

    print("Saving single-run Platt reliability figure to:", out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()


