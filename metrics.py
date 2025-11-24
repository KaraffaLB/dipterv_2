from dataclasses import dataclass
import numpy as np


@dataclass
class EpisodeResult:
    steps: int
    caught: bool
    time_to_capture: float | None
    avg_distance: float
    frac_close_to_ring: float
    frac_in_goal_zone: float


def compute_episode_metrics(distances,
                            in_zone,
                            dt: float,
                            capture_step: int | None,
                            cfg) -> EpisodeResult:
    distances = np.array(distances)
    in_zone = np.array(in_zone)

    avg_d = float(distances.mean())
    frac_close = float(np.mean(np.abs(distances - cfg.R) < 3.0))
    frac_in_zone = float(in_zone.mean())

    if capture_step is not None:
        time_to_capture = dt * capture_step
        caught = True
        steps = capture_step + 1
    else:
        time_to_capture = None
        caught = False
        steps = len(distances)

    return EpisodeResult(
        steps=steps,
        caught=caught,
        time_to_capture=time_to_capture,
        avg_distance=avg_d,
        frac_close_to_ring=frac_close,
        frac_in_goal_zone=frac_in_zone,
    )
