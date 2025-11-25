from __future__ import print_function

"""
train_adversary_platt.py

基于原始的 train_adversary.py：
  - 保持 adversary policy 为标准 GaussianMLPPolicy（raw），由 TRPO 更新；
  - 额外引入一个独立的“Platt scaling”式线性校准层，只用 Gaussian NLL 训练；
  - 每轮使用 on-policy 数据近似做到「大约 80% 给 TRPO，20% 给 calibration」：
      * TRPO 仍然用 batch_size 个样本（和原脚本一样）；
      * 额外收集大约 calib_frac * batch_size 个 on-policy 样本，只用来更新校准层。

注意：
  - 没有改动原有脚本 train_adversary.py，本文件是一个全新的实验脚本。
  - 校准层目前只训练、不在环境交互中真正使用动作（可以在 test 阶段封装一个
    “calibrated adversary policy” 再做对比或 reliability 分析）。
"""

import argparse
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.constant_control_policy import ConstantControlPolicy
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
from rllab.sampler.utils import rollout

from test import test_const_adv, test_rand_adv, test_learnt_adv, test_rand_step_adv, test_step_adv
import pickle


class PlattCalibrator(nn.Module):
    """
    简单的 per-dim 线性校准层，用于对 Gaussian policy 的 (mu, log_std) 做仿射变换：

        mu_c = w * mu + b
        log_std_c = log_std + log(|w|)

    这里只负责根据 Gaussian NLL 更新 (w, b)，不改动原始 policy 参数。
    """

    def __init__(self, action_dim):
        super(PlattCalibrator, self).__init__()
        self.w = nn.Parameter(torch.ones(action_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, mu_raw, logstd_raw):
        """
        mu_raw, logstd_raw: (N, D)
        返回校准后的 (mu_c, logstd_c)
        """
        w = self.w.view(1, -1)
        b = self.b.view(1, -1)
        mu_c = mu_raw * w + b
        logstd_c = logstd_raw + torch.log(torch.abs(w) + 1e-6)
        return mu_c, logstd_c


def gaussian_nll(mu, logstd, actions):
    """
    actions ~ N(mu, std)，计算 per-sample NLL 并取均值。
    形状：
      - mu, logstd, actions: (N, D)
    """
    std = torch.exp(logstd)
    var = std ** 2
    nll = 0.5 * (((actions - mu) ** 2) / var + 2 * logstd + math.log(2 * math.pi))
    return nll.sum(dim=1).mean()


## 命令行参数 ##
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True, help='Name of adversarial environment')
parser.add_argument('--path_length', type=int, default=1000, help='maximum episode length')
parser.add_argument('--layer_size', nargs='+', type=int, default=[100, 100, 100], help='layer definition')
parser.add_argument('--if_render', type=int, default=0, help='Should we render?')
parser.add_argument('--after_render', type=int, default=100, help='After how many to animate')
parser.add_argument('--n_exps', type=int, default=5, help='Number of training instances to run')
parser.add_argument('--n_itr', type=int, default=25, help='Number of iterations of the alternating optimization')
parser.add_argument('--n_pro_itr', type=int, default=1, help='Number of iterations for the portagonist')
parser.add_argument('--n_adv_itr', type=int, default=1, help='Number of interations for the adversary')
parser.add_argument('--batch_size', type=int, default=4000,
                    help='Number of training samples for each TRPO iteration (same as baseline)')
parser.add_argument('--save_every', type=int, default=100, help='Save checkpoint every save_every iterations')
parser.add_argument('--n_process', type=int, default=1, help='Number of parallel threads for sampling environment')
parser.add_argument('--adv_fraction', type=float, default=0.25,
                    help='fraction of maximum adversarial force to be applied')
parser.add_argument('--step_size', type=float, default=0.01, help='kl step size for TRPO')
parser.add_argument('--gae_lambda', type=float, default=0.97, help='gae_lambda for learner')
parser.add_argument('--reset_noise_scale', type=float, default=None,
                    help='reset noise scale for environment initialization')
parser.add_argument('--folder', type=str, default=os.environ.get('HOME', '.'),
                    help='folder to save result in')

# 校准相关参数
parser.add_argument('--calib_frac', type=float, default=0.2,
                    help='approx fraction of extra on-policy samples per iter for calibration (relative to batch_size)')
parser.add_argument('--calib_lr', type=float, default=1e-3, help='learning rate for Platt calibrator')
parser.add_argument('--calib_updates', type=int, default=5,
                    help='number of gradient steps on calibration data per iteration')
parser.add_argument('--calib_max_traj', type=int, default=10,
                    help='max number of extra trajectories per iteration for calibration data collection')


## 解析参数 ##
args = parser.parse_args()
env_name = args.env
path_length = args.path_length
layer_size = tuple(args.layer_size)
ifRender = bool(args.if_render)
afterRender = args.after_render
n_exps = args.n_exps
n_itr = args.n_itr
n_pro_itr = args.n_pro_itr
n_adv_itr = args.n_adv_itr
batch_size = args.batch_size
save_every = args.save_every
n_process = args.n_process
adv_fraction = args.adv_fraction
step_size = args.step_size
gae_lambda = args.gae_lambda
reset_noise_scale = args.reset_noise_scale
save_dir = args.folder

calib_frac = args.calib_frac
calib_lr = args.calib_lr
calib_updates = args.calib_updates
calib_max_traj = args.calib_max_traj


## 为结果准备目录和文件前缀 ##
os.makedirs(save_dir, exist_ok=True)
noise_str = '_noise{}'.format(reset_noise_scale) if reset_noise_scale is not None else ''
save_prefix = 'RARL-env-Platt-{}_Exp{}_Itr{}_BS{}_Adv{}_stp{}_lam{}{}_{}'.format(
    env_name, n_exps, n_itr, batch_size, adv_fraction, step_size, gae_lambda, noise_str, random.randint(0, 1000000)
)
save_name = os.path.join(save_dir, save_prefix + '.p')


## 初始化 summary ##
const_test_rew_summary = []
rand_test_rew_summary = []
step_test_rew_summary = []
rand_step_test_rew_summary = []
adv_test_rew_summary = []


def collect_calibration_data(env, pro_policy, adv_policy, target_steps, max_traj):
    """
    额外使用当前 policy 在 env 上 roll out 若干条 on-policy 轨迹，
    收集 (mu, log_std, a) 作为校准数据。

    只依赖 rllab 自带的 rollout，不改变 TRPO 内部采样代码。
    """
    mus = []
    logstds = []
    acts = []
    steps_collected = 0
    traj_count = 0

    while steps_collected < target_steps and traj_count < max_traj:
        path = rollout(env, pro_policy, max_path_length=path_length,
                       animated=False, adv_agent=adv_policy, test=False)
        adv_actions = path['adv_actions']          # (T, D)
        adv_infos = path['adv_agent_infos']        # dict of arrays
        if 'mean' not in adv_infos or 'log_std' not in adv_infos:
            logger.log("Warning: adv_agent_infos missing mean/log_std for calibration; skip this traj.")
            traj_count += 1
            continue

        mu = adv_infos['mean']
        logstd = adv_infos['log_std']
        assert mu.shape == logstd.shape == adv_actions.shape

        mus.append(mu)
        logstds.append(logstd)
        acts.append(adv_actions)
        steps_collected += adv_actions.shape[0]
        traj_count += 1

    if len(mus) == 0:
        return None, None, None

    mus = np.concatenate(mus, axis=0)
    logstds = np.concatenate(logstds, axis=0)
    acts = np.concatenate(acts, axis=0)
    return mus, logstds, acts


## 主训练循环（多个独立实验） ##
for ne in range(n_exps):
    ## 环境 ##
    env = normalize(GymEnv(env_name, adv_fraction, reset_noise_scale=reset_noise_scale))
    env_orig = normalize(GymEnv(env_name, 1.0, reset_noise_scale=reset_noise_scale))

    ## Protagonist policy ##
    pro_policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=layer_size,
        is_protagonist=True
    )
    pro_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## Zero adversary（只用于测试）##
    zero_adv_policy = ConstantControlPolicy(
        env_spec=env.spec,
        is_protagonist=False,
        constant_val=0.0
    )

    ## Adversary policy（Gaussian raw）##
    adv_policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=layer_size,
        is_protagonist=False
    )
    adv_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## 确定 adversary 动作维度，初始化校准器 ##
    try:
        action_dim = int(env.adv_action_space.flat_dim)
    except Exception:
        try:
            action_dim = int(env._wrapped_env.adv_action_space.flat_dim)
        except Exception:
            action_dim = 1
    logger.log('Detected adversary action_dim = {}'.format(action_dim))

    platt_calib = PlattCalibrator(action_dim)
    calib_optimizer = optim.Adam(platt_calib.parameters(), lr=calib_lr, weight_decay=1e-6)

    ## 初始化并行采样 ##
    parallel_sampler.initialize(n_process)

    ## Protagonist TRPO ##
    pro_algo = TRPO(
        env=env,
        pro_policy=pro_policy,
        adv_policy=adv_policy,
        pro_baseline=pro_baseline,
        adv_baseline=adv_baseline,
        batch_size=batch_size,
        max_path_length=path_length,
        n_itr=n_pro_itr,
        discount=0.995,
        gae_lambda=gae_lambda,
        step_size=step_size,
        is_protagonist=True
    )

    ## Adversary TRPO ##
    adv_algo = TRPO(
        env=env,
        pro_policy=pro_policy,
        adv_policy=adv_policy,
        pro_baseline=pro_baseline,
        adv_baseline=adv_baseline,
        batch_size=batch_size,
        max_path_length=path_length,
        n_itr=n_adv_itr,
        discount=0.995,
        gae_lambda=gae_lambda,
        step_size=step_size,
        is_protagonist=False,
        scope='adversary_optim'
    )

    ## 测试 summary（未使用校准层的性能） ##
    const_testing_rews = []
    const_testing_rews.append(test_const_adv(env_orig, pro_policy, path_length=path_length))
    rand_testing_rews = []
    rand_testing_rews.append(test_rand_adv(env_orig, pro_policy, path_length=path_length))
    step_testing_rews = []
    step_testing_rews.append(test_step_adv(env_orig, pro_policy, path_length=path_length))
    rand_step_testing_rews = []
    rand_step_testing_rews.append(test_rand_step_adv(env_orig, pro_policy, path_length=path_length))
    adv_testing_rews = []
    adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))

    ## 迭代优化 ##
    for ni in range(n_itr):
        logger.log('\n\n\n#### [Platt] expNO{} global itr# {} n_pro_itr# {} ####\n\n\n'
                   .format(ne, ni, n_pro_itr))

        # 1) 正常 TRPO 训练（raw Gaussian policy）
        pro_algo.train()
        logger.log('Protag Reward: {}'.format(np.array(pro_algo.rews).mean()))

        adv_algo.train()
        logger.log('Advers Reward: {}'.format(np.array(adv_algo.rews).mean()))

        # 2) 使用额外 on-policy 数据训练校准层
        target_steps = int(calib_frac * batch_size)
        mus_np, logstds_np, acts_np = collect_calibration_data(
            env, pro_policy, adv_policy, target_steps=target_steps, max_traj=calib_max_traj
        )
        if mus_np is not None:
            mu_t = torch.tensor(mus_np, dtype=torch.float32)
            logstd_t = torch.tensor(logstds_np, dtype=torch.float32)
            a_t = torch.tensor(acts_np, dtype=torch.float32)

            # 可选：打乱一下
            idx = torch.randperm(mu_t.shape[0])
            mu_t = mu_t[idx]
            logstd_t = logstd_t[idx]
            a_t = a_t[idx]

            for _ in range(calib_updates):
                calib_optimizer.zero_grad()
                mu_c, logstd_c = platt_calib(mu_t, logstd_t)
                loss = gaussian_nll(mu_c, logstd_c, a_t)
                loss.backward()
                calib_optimizer.step()

            logger.log('[Platt] calib NLL loss (last step) = {:.6f}'.format(loss.item()))
        else:
            logger.log('[Platt] No calibration data collected this iteration.')

        # 3) 常规测试（此处仍然使用 raw adversary policy，不带校准）
        const_testing_rews.append(test_const_adv(env, pro_policy, path_length=path_length))
        rand_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
        step_testing_rews.append(test_step_adv(env, pro_policy, path_length=path_length))
        rand_step_testing_rews.append(test_rand_step_adv(env, pro_policy, path_length=path_length))
        adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))

        if ni % afterRender == 0 and ifRender:
            test_const_adv(env, pro_policy, path_length=path_length, n_traj=1, render=True)

        # 4) checkpoint（包含校准器状态）
        if ni != 0 and ni % save_every == 0:
            ckpt_dict = {
                'args': args,
                'pro_policy': pro_policy,
                'adv_policy': adv_policy,
                'platt_state': platt_calib.state_dict(),
                'zero_test': const_test_rew_summary,
                'rand_test': rand_test_rew_summary,
                'step_test': step_test_rew_summary,
                'rand_step_test': rand_step_test_rew_summary,
                'iter_save': ni,
                'exp_save': ne,
                'adv_test': adv_test_rew_summary,
            }
            # 临时覆盖版本
            pickle.dump(ckpt_dict, open(save_name + '.temp', 'wb'))
            # 带 iteration 编号的快照，方便之后选取特定迭代分析
            iter_path = save_name + '.iter{}.p'.format(ni)
            pickle.dump(ckpt_dict, open(iter_path, 'wb'))

    ## 关闭并行 worker ##
    pro_algo.shutdown_worker()
    adv_algo.shutdown_worker()

    ## 聚合结果 ##
    const_test_rew_summary.append(const_testing_rews)
    rand_test_rew_summary.append(rand_testing_rews)
    step_test_rew_summary.append(step_testing_rews)
    rand_step_test_rew_summary.append(rand_step_testing_rews)
    adv_test_rew_summary.append(adv_testing_rews)


## 最终保存（包含最后一轮的校准器状态）##
pickle.dump(
    {
        'args': args,
        'pro_policy': pro_policy,
        'adv_policy': adv_policy,
        'platt_state': platt_calib.state_dict(),
        'zero_test': const_test_rew_summary,
        'rand_test': rand_test_rew_summary,
        'step_test': step_test_rew_summary,
        'rand_step_test': rand_step_test_rew_summary,
        'adv_test': adv_test_rew_summary,
    },
    open(save_name, 'wb')
)

logger.log('\n\n\n#### [Platt] DONE ####\n\n\n')


