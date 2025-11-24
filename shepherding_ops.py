import numpy as np
from controllers.base import Controller
from env import SurveillanceEnv


class ShepherdingOPSController(Controller):
    """
    Metaheurisztikus shepherding algoritmus
    - behind + lateral positioning
    - wall safe viselkedés
    - stagnálás elleni korrekció
    """

    def __init__(self, env: SurveillanceEnv):
        self.env = env
        self.k = env.cfg.num_obs
        self.behind_margin = 12.0
        self.wall_margin = 10.0
        self._stuck_counter = [0] * self.k
        self._last_goal_max = np.inf

    def reset(self, seed=None):
        self._stuck_counter = [0] * self.k
        self._last_goal_max = np.inf

    def act(self, state: np.ndarray) -> np.ndarray:
        obs_pos, obs_heading, xT, yT, _ = self.env._split_state(state)
        c = np.array(self.env.cfg.goal_center)
        L = self.env.cfg.env_size
        R = self.env.cfg.R

        u = np.zeros(self.k)

        v_sc = np.array([xT, yT]) - c
        if np.linalg.norm(v_sc) < 1e-6:
            dir_out = np.array([1.0, 0.0])
        else:
            dir_out = v_sc / np.linalg.norm(v_sc)

        perp = np.array([-dir_out[1], dir_out[0]])

        max_sheep_dist = np.linalg.norm([xT - c[0], yT - c[1]])

        for i in range(self.k):
            pos_i = obs_pos[i]

            behind = np.array([xT, yT]) + self.behind_margin * dir_out

            side_sign = np.sign(np.cross(
                np.append(dir_out, 0),
                np.append((pos_i - np.array([xT, yT])) / (np.linalg.norm(pos_i - np.array([xT, yT])) + 1e-9), 0)
            )[-1])
            if side_sign == 0:
                side_sign = 1.0

            behind_lateral = behind + side_sign * 0.6 * self.behind_margin * perp

            target = np.clip(behind_lateral, self.wall_margin, L - self.wall_margin)

            # stagnálás detektálás
            if max_sheep_dist >= self._last_goal_max - 1e-3:
                self._stuck_counter[i] += 1
            else:
                self._stuck_counter[i] = 0

            if self._stuck_counter[i] > 15:
                target += side_sign * perp * self.behind_margin

            self._last_goal_max = max_sheep_dist

            dir_vec = target - pos_i
            if np.linalg.norm(dir_vec) < 1e-6:
                continue

            desired_heading = np.arctan2(dir_vec[1], dir_vec[0])
            error = self.env._wrap_angle(desired_heading - obs_heading[i])

            u[i] = np.clip(error, -self.env.cfg.max_omega, self.env.cfg.max_omega)

        return u
