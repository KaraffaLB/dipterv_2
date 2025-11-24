import numpy as np
from controllers.base import Controller
from env import SurveillanceEnv
from mppi_controller import MPPIController


class MPPIWrapper(Controller):
    """MPPI wrapper, több observerrel is működik (u vektort ad vissza)."""

    def __init__(self, env: SurveillanceEnv):
        self.env = env
        self.ctrl = MPPIController(
            env,
            horizon=35,
            num_samples=300,
            lambda_=4.0,
            sigma=0.18,
        )

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.ctrl.U[:] = 0.0

    def act(self, state: np.ndarray) -> np.ndarray:
        u, _ = self.ctrl.plan(state)
        return u.reshape(-1)
