"""Surveillance környezet több observerrel és több feladattal (task)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
import numpy as np


class TaskType(str, Enum):
    CAPTURE = "capture"
    HERD_TO_GOAL = "herd_to_goal"
    # később: RING_SURVEILLANCE = "ring_surveillance"


@dataclass
class SurveillanceEnvConfig:
    env_size: float = 100.0
    v_target: float = 2.0

    # observer legyen gyorsabb
    alpha: float = 1.8  # v_obs / v_target

    R: float = 10.0
    dt: float = 0.2
    max_omega: float = 1.0

    # observer-ek száma
    num_obs: int = 1

    # kör akadályok
    obstacles: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (30.0, 45.0, 8.0),
            (70.0, 60.0, 10.0),
            (55.0, 25.0, 6.0),
        ]
    )

    obstacle_penalty: float = 1e4

    goal_center: Tuple[float, float] = (80.0, 20.0)
    goal_radius: float = 12.0

    # target (menekülő) paraméterek
    target_turn_gain: float = 0.9
    target_noise: float = 0.08
    target_max_turn: float = 1.0

    # task típus
    task: TaskType = TaskType.CAPTURE


class SurveillanceEnv:
    """
    State vektora:

        x = [xO1, yO1, psi1, ..., xOk, yOk, psik, xT, yT, phiT]

    ahol k = cfg.num_obs.
    Vezérlés: u ∈ R^k, az egyes observer-ek szögsebessége.
    """

    def __init__(self, cfg: SurveillanceEnvConfig):
        self.cfg = cfg
        self.v_obs = cfg.alpha * cfg.v_target

    # ------------------------------------------------------------------
    # STATE SZÉTBONTÁSA
    # ------------------------------------------------------------------

    def _split_state(self, x: np.ndarray):
        """Visszaadja (obs_xy [k,2], obs_psi [k], xT, yT, phiT)."""
        k = self.cfg.num_obs
        obs = x[: 3 * k].reshape(k, 3)
        obs_xy = obs[:, :2]
        obs_psi = obs[:, 2]
        xT, yT, phiT = x[3 * k :]
        return obs_xy, obs_psi, float(xT), float(yT), float(phiT)

    def _assemble_state(self, obs_xy: np.ndarray, obs_psi: np.ndarray,
                        xT: float, yT: float, phiT: float) -> np.ndarray:
        k = self.cfg.num_obs
        obs = np.hstack([obs_xy, obs_psi.reshape(k, 1)])
        return np.concatenate([obs.reshape(-1), np.array([xT, yT, phiT], dtype=float)])

    # ------------------------------------------------------------------
    # AKADÁLYBÓL KITOLÁS
    # ------------------------------------------------------------------

    def _project_out_of_obstacles(self, x: float, y: float) -> tuple[float, float]:
        x_new, y_new = x, y
        for ox, oy, r in self.cfg.obstacles:
            dx = x_new - ox
            dy = y_new - oy
            d = np.hypot(dx, dy)
            if d < r:
                if d < 1e-6:
                    dx, dy = 1.0, 0.0
                    d = 1.0
                scale = (r + 1e-3) / d
                x_new = ox + dx * scale
                y_new = oy + dy * scale
        return x_new, y_new

    # ------------------------------------------------------------------
    # DINAMIKA
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, u) -> np.ndarray:
        """
        Egy lépés szimuláció.
        x: [xO1, yO1, psi1, ..., xOk, yOk, psik, xT, yT, phiT]
        u: szögsebesség-vektor (shape: [k])
        """
        dt = self.cfg.dt
        env_size = self.cfg.env_size
        k = self.cfg.num_obs

        obs_xy, obs_psi, xT, yT, phiT = self._split_state(x)

        u_vec = np.asarray(u, dtype=float).reshape(-1)
        if u_vec.size == 1 and k > 1:
            u_vec = np.repeat(u_vec, k)
        assert u_vec.size == k, "u mérete != num_obs"

        # szögsebesség szaturáció
        omega = np.clip(u_vec, -self.cfg.max_omega, self.cfg.max_omega)

        # observer-ek frissítése
        psi_next = obs_psi + omega * dt
        xO_next = obs_xy[:, 0] + self.v_obs * np.cos(obs_psi) * dt
        yO_next = obs_xy[:, 1] + self.v_obs * np.sin(obs_psi) * dt
        obs_xy_next = np.stack([xO_next, yO_next], axis=1)

        # target frissítése
        target_omega = self._target_policy(obs_xy, xT, yT, phiT)
        phiT_next = phiT + target_omega * dt
        xT_next = xT + self.cfg.v_target * np.cos(phiT_next) * dt
        yT_next = yT + self.cfg.v_target * np.sin(phiT_next) * dt

        # pályán belül tartás
        obs_xy_next[:, 0] = np.clip(obs_xy_next[:, 0], 0.0, env_size)
        obs_xy_next[:, 1] = np.clip(obs_xy_next[:, 1], 0.0, env_size)
        xT_next = float(np.clip(xT_next, 0.0, env_size))
        yT_next = float(np.clip(yT_next, 0.0, env_size))

        # akadályból kitaszítás
        if self.cfg.obstacles:
            for i in range(k):
                obs_xy_next[i, 0], obs_xy_next[i, 1] = self._project_out_of_obstacles(
                    obs_xy_next[i, 0], obs_xy_next[i, 1]
                )
            xT_next, yT_next = self._project_out_of_obstacles(xT_next, yT_next)

        return self._assemble_state(obs_xy_next, psi_next, xT_next, yT_next, phiT_next)

    # ------------------------------------------------------------------
    # TASK-FÜGGŐ KÖLTSÉGEK
    # ------------------------------------------------------------------

    def running_cost(self, x: np.ndarray, u) -> float:
        if self.cfg.task == TaskType.CAPTURE:
            return self._cost_capture(x, u)
        elif self.cfg.task == TaskType.HERD_TO_GOAL:
            return self._cost_herd_to_goal(x, u)
        else:
            raise ValueError(f"Unknown task: {self.cfg.task}")

    def terminal_cost(self, x: np.ndarray) -> float:
        if self.cfg.task == TaskType.CAPTURE:
            return self._terminal_capture(x)
        elif self.cfg.task == TaskType.HERD_TO_GOAL:
            return self._terminal_herd_to_goal(x)
        else:
            raise ValueError(f"Unknown task: {self.cfg.task}")

    # ---- CAPTURE task ----

    def _cost_capture(self, x: np.ndarray, u) -> float:
        env_size = self.cfg.env_size
        obs_xy, obs_psi, xT, yT, phiT = self._split_state(x)

        # legközelebbi observer
        dists = np.linalg.norm(obs_xy - np.array([xT, yT]), axis=1)
        d = float(dists.min())
        idx_min = int(dists.argmin())

        # fő cél: kis távolság
        c_chase = 3.0 * (d ** 2)

        # gyenge heading tag csak a legközelebbi observerre
        desired_heading = np.arctan2(yT - obs_xy[idx_min, 1], xT - obs_xy[idx_min, 0])
        heading_error = self._wrap_angle(desired_heading - obs_psi[idx_min])
        c_heading = 0.02 * (heading_error ** 2)

        # pályáról lemenés (bármely observerre)
        if np.any(obs_xy[:, 0] < 0.0) or np.any(obs_xy[:, 0] > env_size) or \
           np.any(obs_xy[:, 1] < 0.0) or np.any(obs_xy[:, 1] > env_size):
            c_env = 1e6
        else:
            c_env = 0.0

        # akadálybüntetés összes observerre
        c_obs = 0.0
        for i in range(self.cfg.num_obs):
            c_obs += self._obstacle_penalty(obs_xy[i, 0], obs_xy[i, 1])

        # target goal zónába húzása – kis súllyal
        gx, gy = self.cfg.goal_center
        d_goal = np.sqrt((xT - gx) ** 2 + (yT - gy) ** 2)
        slack = max(0.0, d_goal - self.cfg.goal_radius)
        c_goal = 0.1 * (slack ** 2)

        # vezérlés büntetés – összes observerre
        u_vec = np.asarray(u, dtype=float).reshape(-1)
        c_u = 0.05 * float(np.sum(u_vec ** 2))

        return float(c_chase + c_heading + c_env + c_obs + c_goal + c_u)

    def _terminal_capture(self, x: np.ndarray) -> float:
        obs_xy, obs_psi, xT, yT, phiT = self._split_state(x)
        dists = np.linalg.norm(obs_xy - np.array([xT, yT]), axis=1)
        d = float(dists.min())
        return float(d ** 2)

    # ---- HERD_TO_GOAL task ----

    def _cost_herd_to_goal(self, x: np.ndarray, u) -> float:
        """
        Tereléses cost (multi-agent):
        - erősen bünteti, ha a target messze van a goal zónától,
        - bünteti, ha NINCS observer a target és a goal között (fal-képzés),
        - bünteti, ha az ügynökök szögben egy oldalra torlódnak,
        - bünteti, ha az ügynökök átlagosan messzire elkóborolnak a targettől,
        - gyengébben figyeli, hogy a legközelebbi observer ne szakadjon le,
        - plusz heading / obstacle / control regularizáció.
        """
        env_size = self.cfg.env_size
        obs_xy, obs_psi, xT, yT, phiT = self._split_state(x)
        k = self.cfg.num_obs

        # --- 1) target–goal távolság (fő cél) ---
        gx, gy = self.cfg.goal_center
        d_goal = np.sqrt((xT - gx) ** 2 + (yT - gy) ** 2)
        slack_goal = max(0.0, d_goal - self.cfg.goal_radius)
        c_goal = 3.0 * (slack_goal ** 2)

        # --- 2) legközelebbi observer–target távolság (követés) ---
        dists = np.linalg.norm(obs_xy - np.array([xT, yT]), axis=1)
        d_min = float(dists.min())
        idx_min = int(dists.argmin())
        c_follow = 0.15 * (d_min ** 2)   # kicsit kisebb súly, mint korábban

        # --- 3) heading – csak a legközelebbi observerre ---
        desired_heading = np.arctan2(yT - obs_xy[idx_min, 1],
                                     xT - obs_xy[idx_min, 0])
        heading_error = self._wrap_angle(desired_heading - obs_psi[idx_min])
        c_heading = 0.02 * (heading_error ** 2)

        # --- 4) "Fal" a target és a goal között (legalább egy observer) ---
        g_vec = np.array([gx - xT, gy - yT])
        g_norm = float(np.linalg.norm(g_vec))
        if g_norm > 1e-6 and k > 0:
            g_unit = g_vec / g_norm
            rel = obs_xy - np.array([xT, yT])
            proj = rel @ g_unit  # vetítés a target→goal irányra

            # 0 <= proj <= g_norm → observer a target és a goal között
            violation = np.where(
                proj < 0.0,
                -proj,
                np.where(proj > g_norm, proj - g_norm, 0.0),
            )
            # legalább EGY observer legyen jó helyen → a legkisebb sértés legyen kicsi
            min_viol = float(np.min(violation))
            c_wall = 3.0 * (min_viol ** 2)   # erősebb fal-súly
        else:
            c_wall = 0.0

        # --- 5) Szögbeli szétterülés a target körül ---
        if k >= 2:
            angles = np.arctan2(obs_xy[:, 1] - yT, obs_xy[:, 0] - xT)
            angles_sorted = np.sort(angles)

            diffs = np.diff(angles_sorted)
            wrap_diff = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
            diffs = np.concatenate([diffs, [wrap_diff]])

            max_gap = float(np.max(diffs))          # legnagyobb "lyuk" a körön
            coverage = 2.0 * np.pi - max_gap        # lefedett szög-tartomány
            target_coverage = np.pi                 # legalább félkör legyen körülötte
            slack_spread = max(0.0, target_coverage - coverage)
            c_spread = 1.0 * (slack_spread ** 2)    # erősebb szétterülés-súly
        else:
            c_spread = 0.0

        # --- 6) Elkóborlás büntetése: minden ügynök legyen nagyjából a közelben ---
        d_all = np.linalg.norm(obs_xy - np.array([xT, yT]), axis=1)
        mean_d2 = float(np.mean(d_all ** 2))
        c_stay_near = 0.3 * mean_d2

        # --- 7) pályáról lemenés büntetés ---
        if np.any(obs_xy[:, 0] < 0.0) or np.any(obs_xy[:, 0] > env_size) or \
           np.any(obs_xy[:, 1] < 0.0) or np.any(obs_xy[:, 1] > env_size):
            c_env = 1e6
        else:
            c_env = 0.0

        # --- 8) akadálybüntetés összes observerre ---
        c_obs = 0.0
        for i in range(k):
            c_obs += self._obstacle_penalty(obs_xy[i, 0], obs_xy[i, 1])

        # --- 9) vezérlés büntetés (összes ügynökre) ---
        u_vec = np.asarray(u, dtype=float).reshape(-1)
        c_u = 0.05 * float(np.sum(u_vec ** 2))

        return float(
            c_goal
            + c_follow
            + c_heading
            + c_wall
            + c_spread
            + c_stay_near
            + c_env
            + c_obs
            + c_u
        )



    def _terminal_herd_to_goal(self, x: np.ndarray) -> float:
        obs_xy, obs_psi, xT, yT, phiT = self._split_state(x)
        gx, gy = self.cfg.goal_center
        d_goal = np.sqrt((xT - gx) ** 2 + (yT - gy) ** 2)
        slack_goal = max(0.0, d_goal - self.cfg.goal_radius)
        return float(slack_goal ** 2)

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self, random_target: bool = True) -> np.ndarray:
        L = self.cfg.env_size
        k = self.cfg.num_obs

        if random_target:
            # observer-ek: pálya szélén, nagyjából szétosztva
            obs_xy = np.zeros((k, 2), dtype=float)
            obs_psi = np.zeros(k, dtype=float)
            for i in range(k):
                side = (i % 4)
                if side == 0:
                    obs_xy[i] = [0.0, np.random.rand() * L]
                    obs_psi[i] = 0.0
                elif side == 1:
                    obs_xy[i] = [L, np.random.rand() * L]
                    obs_psi[i] = np.pi
                elif side == 2:
                    obs_xy[i] = [np.random.rand() * L, 0.0]
                    obs_psi[i] = np.pi / 2.0
                else:
                    obs_xy[i] = [np.random.rand() * L, L]
                    obs_psi[i] = -np.pi / 2.0

            # target: közép környéke
            xT = L * (0.4 + 0.2 * np.random.rand())
            yT = L * (0.4 + 0.2 * np.random.rand())
            phiT = float(np.random.uniform(-np.pi, np.pi))
        else:
            obs_xy = np.zeros((k, 2), dtype=float)
            obs_psi = np.zeros(k, dtype=float)
            for i in range(k):
                obs_xy[i] = [0.1 * L, 0.1 * L + i * 3.0]
                obs_psi[i] = np.arctan2(0.5 * L - obs_xy[i, 1], 0.5 * L - obs_xy[i, 0])

            xT = 0.5 * L
            yT = 0.5 * L
            phiT = 0.0

        return self._assemble_state(obs_xy, obs_psi, xT, yT, phiT)

    # ------------------------------------------------------------------
    # AKADÁLY SEGÉDFÜGGVÉNYEK
    # ------------------------------------------------------------------

    def _is_in_obstacle(self, x: float, y: float) -> bool:
        for ox, oy, r in self.cfg.obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= r ** 2:
                return True
        return False

    def _obstacle_penalty(self, x: float, y: float) -> float:
        penalty = 0.0
        for ox, oy, r in self.cfg.obstacles:
            d = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if d < r:
                penalty += self.cfg.obstacle_penalty * (1.0 + (r - d))
        return penalty

    # ------------------------------------------------------------------
    # TARGET POLICY: legközelebbi observer elől menekülés
    # ------------------------------------------------------------------

    def _target_policy(self, obs_xy: np.ndarray,
                       xT: float, yT: float, phiT: float) -> float:
        env_size = self.cfg.env_size

        # legközelebbi observer
        dists = np.linalg.norm(obs_xy - np.array([xT, yT]), axis=1)
        idx_min = int(dists.argmin())
        away_vec = np.array([xT - obs_xy[idx_min, 0], yT - obs_xy[idx_min, 1]])
        if np.linalg.norm(away_vec) < 1e-6:
            away_vec = np.array([1.0, 0.0])

        # akadály taszítás
        repulse_vec = np.zeros(2)
        for ox, oy, r in self.cfg.obstacles:
            d = np.sqrt((xT - ox) ** 2 + (yT - oy) ** 2)
            if d < r + 8.0:
                strength = max(0.0, (r + 8.0 - d) / 8.0)
                repulse_vec += strength * np.array([xT - ox, yT - oy])

        # fal közelében visszanyomás
        margin = 5.0
        wall_force = np.zeros(2)
        if xT < margin:
            wall_force[0] += (margin - xT)
        if xT > env_size - margin:
            wall_force[0] -= (xT - (env_size - margin))
        if yT < margin:
            wall_force[1] += (margin - yT)
        if yT > env_size - margin:
            wall_force[1] -= (yT - (env_size - margin))

        combined = away_vec + repulse_vec + 0.8 * wall_force
        desired_heading = np.arctan2(combined[1], combined[0])

        heading_error = self._wrap_angle(desired_heading - phiT)
        omega = self.cfg.target_turn_gain * heading_error
        omega = np.clip(omega, -self.cfg.target_max_turn, self.cfg.target_max_turn)
        omega += self.cfg.target_noise * np.random.randn()

        return float(omega)

    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi
