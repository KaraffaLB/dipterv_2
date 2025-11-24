"""MPPI controller többdimenziós vezérléshez (num_obs)."""

from typing import Tuple
import numpy as np

from env import SurveillanceEnv


class MPPIController:
    def __init__(
        self,
        env: SurveillanceEnv,
        horizon: int = 10,
        num_samples: int = 80,
        lambda_: float = 1.0,
        sigma: float = 0.5,
        control_dim: int | None = None,
    ):
        self.env = env
        self.H = horizon
        self.N = num_samples
        self.lambda_ = lambda_
        self.sigma = sigma

        # vezérlés dimenzió: observer-ek száma
        if control_dim is None:
            self.m = getattr(env.cfg, "num_obs", 1)
        else:
            self.m = control_dim

        # kezdeti vezérlési szekvencia
        self.U = np.zeros((self.H, self.m), dtype=float)

    def plan(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """MPPI iteráció, vektoriális u-val. Visszaadja (u0, teljes U)."""
        N, H, m = self.N, self.H, self.m
        lambda_ = self.lambda_
        sigma = self.sigma

        costs = np.zeros(N)
        eps = np.zeros((N, H, m))

        for i in range(N):
            x = x0.copy()
            J = 0.0
            for k in range(H):
                noise = sigma * np.random.randn(m)
                eps[i, k, :] = noise
                u_k = self.U[k] + noise
                x = self.env.step(x, u_k)
                J += self.env.running_cost(x, u_k)
            J += self.env.terminal_cost(x)
            costs[i] = J

        # súlyok
        J_min = np.min(costs)
        w = np.exp(-(costs - J_min) / lambda_)
        w_sum = np.sum(w)
        if w_sum == 0.0:
            w = np.ones_like(w) / len(w)
        else:
            w /= w_sum

        # vezérlés frissítése
        for k in range(H):
            self.U[k] = self.U[k] + np.sum(w[:, None] * eps[:, k, :], axis=0)

        # első lépés + horizont csúsztatás
        u0 = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]

        return u0, self.U.copy()
