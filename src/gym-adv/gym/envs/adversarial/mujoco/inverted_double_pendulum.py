import numpy as np
from gym import utils, spaces
from gym.envs.adversarial.mujoco import mujoco_env

class InvertedDoublePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, reset_noise_scale=0.01):
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum.xml', 5)
        utils.EzPickle.__init__(self)
        self.reset_noise_scale = reset_noise_scale
        self._reset_count = 0  # 计数器，用于控制打印频率
	## Adversarial setup
        self._adv_f_bname = b'pole2' #Byte String name of body on which the adversary force will be applied
        bnames = self.model.body_names
        self._adv_bindex = bnames.index(self._adv_f_bname) #Index of the body on which the adversary force will be applied
        adv_max_force = 5.
        high_adv = np.ones(2)*adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space

    def _adv_to_xfrc(self, adv_act):
        new_xfrc = self.model.data.xfrc_applied*0.0
        new_xfrc[self._adv_bindex] = np.array([adv_act[0], 0., adv_act[1], 0., 0., 0.])
        self.model.data.xfrc_applied = new_xfrc

    def sample_action(self):
        class act(object):
            def __init__(self,pro=None,adv=None):
                self.pro=pro
                self.adv=adv
        sa = act(self.pro_action_space.sample(), self.adv_action_space.sample())
        return sa

    def _step(self, action):
        if hasattr(action, '__dict__'):
            self._adv_to_xfrc(action.adv)
            a = action.pro
        else:
            a = action

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        site_name = b'tip'
        site_index = self.model.site_names.index(site_name)
        x, _, y = self.model.data.site_xpos[site_index]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.model.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = (alive_bonus - dist_penalty - vel_penalty)[0]
        done = bool(y <= 1)
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos[:1],  # cart x pos
            np.sin(self.model.data.qpos[1:]),  # link angles
            np.cos(self.model.data.qpos[1:]),
            np.clip(self.model.data.qvel, -10, 10),
            np.clip(self.model.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale
        qpos_noise = self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel_noise = self.np_random.randn(self.model.nv) * self.reset_noise_scale
        
        # 每 100 个 episode 打印一次，或者前 5 个 episode 都打印
        self._reset_count += 1
        if self._reset_count <= 5 or self._reset_count % 100 == 0:
            print("[Reset Noise #{}] reset_noise_scale={:.6f}, qpos_noise_range=[{:.6f}, {:.6f}], qvel_noise_range=[{:.6f}, {:.6f}]".format(
                self._reset_count,
                self.reset_noise_scale,
                qpos_noise.min(), qpos_noise.max(),
                qvel_noise.min(), qvel_noise.max()
            ))
        
        self.set_state(
            self.init_qpos + qpos_noise,
            self.init_qvel + qvel_noise
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid=0
        v.cam.distance = v.model.stat.extent * 0.5
        v.cam.lookat[2] += 3#v.model.stat.center[2]
