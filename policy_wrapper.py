import numpy as np
from controllers.base import Controller
from mppi_controller import MPPIController


class MPPIWrapper(Controller):
    """
    MPPI vezérlő wrapper stabil, nem-köröző viselkedéssel.
    """

    def __init__(self, env):
        self.env = env

        self.ctrl = MPPIController(
            env,
            horizon=35,
            num_samples=300,
            lambda_=4.0,
            sigma=0.18,   # csökkentett zaj -> nincs oszcilláció
        )

    def reset(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        # MPPI belső vezérlési sorozat nullázása
        self.ctrl.U[:] = 0.0

    def act(self, state: np.ndarray) -> float:
        u, _ = self.ctrl.plan(state)
        return float(u)
