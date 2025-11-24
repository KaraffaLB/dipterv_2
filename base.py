import numpy as np
from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        pass

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        """state -> u (vektor, shape [num_obs])."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
