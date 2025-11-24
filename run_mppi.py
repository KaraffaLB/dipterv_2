"""MPPI rollout, task és observer-szám választással."""

import argparse

from env import SurveillanceEnv, SurveillanceEnvConfig, TaskType
from controllers.mppi_wrapper import MPPIWrapper
from eval.runner import run_episode


def main():
    parser = argparse.ArgumentParser(description="MPPI rollout különböző feladatokkal")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_target", action="store_true")
    parser.add_argument("--capture_radius", type=float, default=1.5)
    parser.add_argument(
        "--task",
        choices=[t.value for t in TaskType],
        default=TaskType.CAPTURE.value,
    )
    parser.add_argument(
        "--num_obs",
        type=int,
        default=1,
        help="Observer-ek száma (ügynökök száma)",
    )
    args = parser.parse_args()

    cfg = SurveillanceEnvConfig(
        task=TaskType(args.task),
        num_obs=args.num_obs,
    )
    env = SurveillanceEnv(cfg)
    ctrl = MPPIWrapper(env)

    result = run_episode(
        env=env,
        controller=ctrl,
        random_target=args.random_target,
        max_steps=args.steps,
        capture_radius=args.capture_radius,
        seed=args.seed,
        visualize=True,
    )

    print("\n=== MPPI episode result ===")
    print(f"task: {cfg.task.value}")
    print(f"num_obs: {cfg.num_obs}")
    print(f"steps: {result.steps}")
    print(f"caught: {result.caught}")
    print(f"time_to_capture: {result.time_to_capture}")
    print(f"avg_distance: {result.avg_distance:.2f}")
    print(f"frac_close_to_ring: {result.frac_close_to_ring:.2f}")
    print(f"frac_in_goal_zone: {result.frac_in_goal_zone:.2f}")


if __name__ == "__main__":
    main()
