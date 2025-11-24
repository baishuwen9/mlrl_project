# rarl_with_calib.py (Python 3.5 compatible)
"""
RARL + Linear Recalibration module (post-hoc & joint)
Usage example:
python rarl_with_calib.py --env InvertedPendulum-v2 --reset_noise_scale 0.01 --mode joint
Modes: 'none' (no calib), 'posthoc' (collect then offline train), 'joint' (online train calibrator)
"""
import os
import argparse
import random
import pickle
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

# rllab imports (keep same as your baseline)
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.constant_control_policy import ConstantControlPolicy
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler

# your test helpers (assumed available)
from test import test_const_adv, test_rand_adv, test_learnt_adv, test_rand_step_adv, test_step_adv

# ------------------------------
# Calibration module (PyTorch)
# ------------------------------
class LinearCalibrator(nn.Module):
    """
    Simple per-dim linear recalibration:
    a_calib = w * a_raw + b
    We'll treat w as unconstrained (learn), but during NLL we use abs(w) for scale handling.
    """
    def __init__(self, action_dim):
        super(LinearCalibrator, self).__init__()
        self.action_dim = action_dim
        # initialize w to ones and b to zeros
        self.w = nn.Parameter(torch.ones(action_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, a_raw):
        # a_raw: tensor (N, action_dim) or (action_dim,)
        return a_raw * self.w.view(1, -1) + self.b.view(1, -1)

def get_action_space(env):
    # Case 1: GymEnv wrapper has spec
    try:
        return env.spec.action_space
    except:
        pass

    # Case 2: underlying gym env
    try:
        return env._wrapped_env.action_space
    except:
        pass

    # Case 3: adversarial env might have env.env
    try:
        return env.env.action_space
    except:
        pass

    raise ValueError("Cannot find action_space in env or wrapper")

def gaussian_nll_from_policy(mu, logstd, z_real, w_tensor):
    """
    Compute NLL of z_real under calibrated Gaussian:
      a_raw ~ N(mu, std)
      a_calib = w * a_raw + b  (we assume b accounted separately; here mu/logstd are raw)
    For calibrator we will compute calibrated mu' = w*mu + b, std' = |w| * std
    Inputs are torch tensors, shapes: (N, D)
    w_tensor: shape (D,) (value of w)
    """
    # mu, logstd, z_real: (N, D)
    std = torch.exp(logstd)
    # calibrated mean and std
    w = w_tensor.view(1, -1)
    mu_c = mu * w
    std_c = std * torch.abs(w) + 1e-8
    var_c = std_c ** 2
    # older torch may not have torch.pi, use math.pi instead
    nll = 0.5 * (((z_real - mu_c) ** 2) / var_c + 2 * torch.log(std_c) + math.log(2 * math.pi))
    return nll.sum(dim=1).mean()

# ------------------------------
# Utilities to collect policy outputs
# ------------------------------
def policy_get_action_and_info(policy, obs):
    """
    Wrapper to get action and agent_info from rllab policy.
    Returns (action_np, info_dict)
    info_dict may contain 'mean' and 'log_std' or some other keys.
    """
    a, info = policy.get_action(obs)
    return a, info


def extract_mean_logstd_from_info(info, action_dim):
    """
    Try common keys to extract mean/log_std as numpy arrays.
    If not available, return (None,None)
    """
    if isinstance(info, dict):
        if 'mean' in info and 'log_std' in info:
            mu = np.array(info['mean']).reshape(-1)
            logstd = np.array(info['log_std']).reshape(-1)
            return mu, logstd
        if 'mean' in info and 'log_std' not in info:
            mu = np.array(info['mean']).reshape(-1)
            return mu, None
    return None, None

# ------------------------------
# Main script (based on your original)
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--path_length', type=int, default=1000)
    parser.add_argument('--layer_size', nargs='+', type=int, default=[100,100,100])
    parser.add_argument('--if_render', type=int, default=0)
    parser.add_argument('--after_render', type=int, default=100)
    parser.add_argument('--n_exps', type=int, default=1)
    parser.add_argument('--n_itr', type=int, default=25)
    parser.add_argument('--n_pro_itr', type=int, default=1)
    parser.add_argument('--n_adv_itr', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--n_process', type=int, default=1)
    parser.add_argument('--adv_fraction', type=float, default=0.25)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--gae_lambda', type=float, default=0.97)
    parser.add_argument('--reset_noise_scale', type=float, default=None)
    parser.add_argument('--folder', type=str, default=os.environ.get('HOME', '.'))
    parser.add_argument('--mode', type=str, default='joint', choices=['none','posthoc','joint'])
    parser.add_argument('--calib_lr', type=float, default=1e-3)
    parser.add_argument('--calib_steps_posthoc', type=int, default=200)
    args = parser.parse_args()

    env_name = args.env
    reset_noise_scale = args.reset_noise_scale
    n_itr = args.n_itr
    n_exps = args.n_exps
    layer_size = tuple(args.layer_size)

    # prepare folder
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    save_prefix = 'RARL_calib_{}_noise{}_mode{}_{}'.format(env_name, reset_noise_scale, args.mode, random.randint(0,999999))
    save_path = os.path.join(args.folder, save_prefix + '.p')

    # summaries
    const_test_rew_summary = []
    rand_test_rew_summary = []
    step_test_rew_summary = []
    rand_step_test_rew_summary = []
    adv_test_rew_summary = []

    for ne in range(n_exps):
        env = normalize(GymEnv(env_name, args.adv_fraction, reset_noise_scale=reset_noise_scale))
        env_orig = normalize(GymEnv(env_name, 0.0, reset_noise_scale=reset_noise_scale))

        pro_policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=layer_size, is_protagonist=True)
        pro_baseline = LinearFeatureBaseline(env_spec=env.spec)

        zero_adv_policy = ConstantControlPolicy(env_spec=env.spec, is_protagonist=False, constant_val=0.0)

        adv_policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=layer_size, is_protagonist=False)
        adv_baseline = LinearFeatureBaseline(env_spec=env.spec)

        # get adversary action dim directly from env's adv_action_space (rllab-adv uses pro/adv spaces)
        try:
            # normalized env exposes adv_action_space
            action_dim = int(env.adv_action_space.flat_dim)
        except Exception:
            # very defensive fallback: try wrapped env
            try:
                action_dim = int(env._wrapped_env.adv_action_space.flat_dim)
            except Exception:
                # last resort: assume 1-dim (should not happen in our tasks)
                action_dim = 1
        logger.log('Detected adversary action_dim = {}'.format(action_dim))

        calibrator = LinearCalibrator(action_dim)
        calib_optimizer = optim.Adam(calibrator.parameters(), lr=args.calib_lr, weight_decay=1e-6)

        # storage for post-hoc data
        mu_store = []
        logstd_store = []
        a_raw_store = []

        parallel_sampler.initialize(args.n_process)

        pro_algo = TRPO(env=env, pro_policy=pro_policy, adv_policy=adv_policy,
                        pro_baseline=pro_baseline, adv_baseline=adv_baseline,
                        batch_size=args.batch_size, max_path_length=args.path_length,
                        n_itr=args.n_pro_itr, discount=0.995, gae_lambda=args.gae_lambda,
                        step_size=args.step_size, is_protagonist=True)

        adv_algo = TRPO(env=env, pro_policy=pro_policy, adv_policy=adv_policy,
                        pro_baseline=pro_baseline, adv_baseline=adv_baseline,
                        batch_size=args.batch_size, max_path_length=args.path_length,
                        n_itr=args.n_adv_itr, discount=0.995, gae_lambda=args.gae_lambda,
                        step_size=args.step_size, is_protagonist=False, scope='adversary_optim')

        const_testing_rews = [test_const_adv(env_orig, pro_policy, path_length=args.path_length)]
        rand_testing_rews = [test_rand_adv(env_orig, pro_policy, path_length=args.path_length)]
        step_testing_rews = [test_step_adv(env_orig, pro_policy, path_length=args.path_length)]
        rand_step_testing_rews = [test_rand_step_adv(env_orig, pro_policy, path_length=args.path_length)]
        adv_testing_rews = [test_learnt_adv(env, pro_policy, adv_policy, path_length=args.path_length)]

        obs = env.reset()
        for ni in range(n_itr):
            logger.log('\n\n\n####expNO{} global itr# {} n_pro_itr# {}####\n\n\n'.format(ne, ni, args.n_pro_itr))
            # Train protagonist (TRPO)
            pro_algo.train()
            logger.log('Protag Reward: {}'.format(np.array(pro_algo.rews).mean()))

            # Train adversary
            adv_algo.train()
            logger.log('Advers Reward: {}'.format(np.array(adv_algo.rews).mean()))

            # After each full alternation, collect a small dataset by running env with adv_policy,
            # but instead of using sampled a directly we capture (mu, logstd, a_sample) from policy info.
            # We'll run a few rollouts to collect enough points.
            collect_n_traj = 4
            collect_max_len = min(200, args.path_length)  # small buffer
            collected = 0
            traj_count = 0
            while traj_count < collect_n_traj:
                o = env.reset()
                done = False
                t = 0
                while not done and t < collect_max_len:
                    # protagonist action from its current policy
                    pro_a, pro_info = pro_policy.get_action(o)

                    # adversary raw action and dist info (for calibration)
                    a_raw, info = policy_get_action_and_info(adv_policy, o)
                    a_raw = np.array(a_raw).reshape(-1)
                    mu, logstd = extract_mean_logstd_from_info(info, action_dim)

                    # if policy does not provide mean/logstd, collect multiple samples later for fit
                    if mu is None:
                        mu = None
                        logstd = None

                    # store raw + dist stats
                    a_raw_store.append(a_raw.copy())
                    if mu is not None:
                        mu_store.append(mu.copy())
                    if logstd is not None:
                        logstd_store.append(logstd.copy())

                    # apply calibration online if mode == 'joint'
                    if args.mode == 'joint':
                        a_t = torch.tensor(a_raw.reshape(1, -1), dtype=torch.float32)
                        with torch.no_grad():
                            a_calib = calibrator(a_t).cpu().numpy().reshape(-1)
                        adv_a = a_calib
                    else:
                        adv_a = a_raw

                    # construct combined action object with .pro and .adv as expected by NormalizedEnv
                    class temp_action(object):
                        pro = None
                        adv = None
                    cum_a = temp_action()
                    cum_a.pro = pro_a
                    cum_a.adv = adv_a

                    o, r, done, info_env = env.step(cum_a)
                    t += 1
                traj_count += 1

            # ONLINE calibrator update: train calibrator for a few steps on collected data
            if args.mode == 'joint':
                # prefer using entries where mu/logstd exist; if not, approximate mu/logstd by sample mean/std over a sliding window
                # build tensors
                if len(mu_store) > 0 and len(logstd_store) > 0:
                    mu_np = np.stack(mu_store[-len(logstd_store):], axis=0) if len(mu_store)>=len(logstd_store) else np.stack(mu_store, axis=0)
                    logstd_np = np.stack(logstd_store, axis=0)
                    a_np = np.stack(a_raw_store[-logstd_np.shape[0]:], axis=0)
                    mu_t = torch.tensor(mu_np, dtype=torch.float32)
                    logstd_t = torch.tensor(logstd_np, dtype=torch.float32)
                    a_t = torch.tensor(a_np, dtype=torch.float32)
                    # train calibrator small number of steps
                    for _ in range(10):
                        calib_optimizer.zero_grad()
                        # use calibrator.w directly so gradients can flow into calibrator parameters
                        w_val = calibrator.w
                        nll = gaussian_nll_from_policy(mu_t, logstd_t, a_t, w_val)
                        nll.backward()
                        calib_optimizer.step()
                else:
                    # fallback: fit mu/logstd from recent samples (simple)
                    if len(a_raw_store) >= 16:
                        recent = np.stack(a_raw_store[-128:], axis=0)
                        mu_est = recent.mean(axis=0, keepdims=True)
                        std_est = recent.std(axis=0, keepdims=True) + 1e-6
                        mu_t = torch.tensor(np.repeat(mu_est, repeats=recent.shape[0], axis=0), dtype=torch.float32)
                        logstd_t = torch.tensor(np.log(std_est), dtype=torch.float32)
                        a_t = torch.tensor(recent, dtype=torch.float32)
                        for _ in range(5):
                            calib_optimizer.zero_grad()
                            w_val = calibrator.w
                            nll = gaussian_nll_from_policy(mu_t, logstd_t, a_t, w_val)
                            nll.backward()
                            calib_optimizer.step()

            # append test evaluations
            const_testing_rews.append(test_const_adv(env_orig, pro_policy, path_length=args.path_length))
            rand_testing_rews.append(test_rand_adv(env_orig, pro_policy, path_length=args.path_length))
            step_testing_rews.append(test_step_adv(env_orig, pro_policy, path_length=args.path_length))
            rand_step_testing_rews.append(test_rand_step_adv(env_orig, pro_policy, path_length=args.path_length))
            adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=args.path_length))

            # optionally save checkpoints
            if (ni != 0) and (ni % args.save_every == 0):
                pickle.dump({
                    'args': vars(args),
                    'pro_policy': pro_policy,
                    'adv_policy': adv_policy,
                    'calibrator_state': calibrator.state_dict(),
                    'mu_store': mu_store,
                    'logstd_store': logstd_store,
                    'a_raw_store': a_raw_store,
                    'const_test': const_test_rew_summary,
                    'rand_test': rand_test_rew_summary,
                    'step_test': step_test_rew_summary,
                    'adv_test': adv_test_rew_summary
                }, open(save_path + '.iter{}.p'.format(ni), 'wb'))

        # post-hoc calibration training if selected
        if args.mode == 'posthoc':
            # if mu/logstd were recorded, use them; otherwise fit gaussian to stored samples
            if len(mu_store) > 0 and len(logstd_store) > 0:
                mu_np = np.stack(mu_store, axis=0)
                logstd_np = np.stack(logstd_store, axis=0)
                a_np = np.stack(a_raw_store, axis=0)[:mu_np.shape[0]]
            else:
                # estimate mu/logstd by sample statistics
                a_np = np.stack(a_raw_store, axis=0)
                mu_np = np.mean(a_np, axis=0, keepdims=True).repeat(a_np.shape[0], axis=0)
                logstd_np = np.log(np.std(a_np, axis=0, keepdims=True) + 1e-8).repeat(a_np.shape[0], axis=0)

            # pytorch tensors
            mu_t = torch.tensor(mu_np, dtype=torch.float32)
            logstd_t = torch.tensor(logstd_np, dtype=torch.float32)
            a_t = torch.tensor(a_np, dtype=torch.float32)

            # train calibrator
            for epoch in range(args.calib_steps_posthoc):
                calib_optimizer.zero_grad()
                w_val = calibrator.w
                loss = gaussian_nll_from_policy(mu_t, logstd_t, a_t, w_val)
                loss.backward()
                calib_optimizer.step()
                if epoch % 20 == 0:
                    print('[posthoc] epoch {} nll {:.4f}'.format(epoch, loss.item()))
            # save calibrator
            torch.save(calibrator.state_dict(), save_path + '.calibrator.pth')

        # shutdown
        pro_algo.shutdown_worker()
        adv_algo.shutdown_worker()

        # aggregate results
        const_test_rew_summary.append(const_testing_rews)
        rand_test_rew_summary.append(rand_testing_rews)
        step_test_rew_summary.append(step_testing_rews)
        rand_step_test_rew_summary.append(rand_step_testing_rews)
        adv_test_rew_summary.append(adv_testing_rews)

    # final save
    pickle.dump({
        'args': vars(args),
        'pro_policy': pro_policy,
        'adv_policy': adv_policy,
        'calibrator_state': calibrator.state_dict(),
        'const_test': const_test_rew_summary,
        'rand_test': rand_test_rew_summary,
        'step_test': step_test_rew_summary,
        'rand_step_test': rand_step_test_rew_summary,
        'adv_test': adv_test_rew_summary
    }, open(save_path, 'wb'))

    logger.log('\n\n\n#### DONE ####\n\n\n')


if __name__ == '__main__':
    main()
