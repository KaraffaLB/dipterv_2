import argparse
import numpy as np


from env import SurveillanceEnvConfig, SurveillanceEnv
from controllers.mppi_wrapper import MPPIWrapper
from controllers.policy_wrapper import PolicyWrapper
from eval.runner import run_episode




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()


    cfg = SurveillanceEnvConfig()
    env = SurveillanceEnv(cfg)


    controllers = [
    MPPIWrapper(env),
    ]


    try:
        controllers.append(PolicyWrapper(env, "policy_model.pth"))
    except Exception:
        pass


    for controller in controllers:
        results = []
        for ep in range(args.episodes):
            res = run_episode(controller, cfg, True, 600, 1.5, args.seed + ep)
            results.append(res)


            caught_rate = np.mean([r.caught for r in results])
            avg_dist = np.mean([r.avg_distance for r in results])


            print(f"Controller: {controller.name}")
            print(f" Caught rate: {100*caught_rate:.1f}%")
            print(f" Avg distance: {avg_dist:.2f}")




if __name__ == "__main__":
    main()