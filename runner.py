import numpy as np
from env import SurveillanceEnv, TaskType
from controllers.base import Controller
from eval.metrics import compute_episode_metrics


def run_episode(
    env: SurveillanceEnv,
    controller: Controller,
    random_target: bool,
    max_steps: int,
    capture_radius: float,
    seed: int,
    visualize: bool = True,
):
    np.random.seed(seed)
    controller.reset(seed)
    x = env.reset(random_target=random_target)

    k = env.cfg.num_obs

    traj_obs = [[] for _ in range(k)]  # list of list[(x,y)]
    traj_tgt: list[tuple[float, float]] = []
    distances: list[float] = []
    in_zone: list[bool] = []
    capture_step: int | None = None

    if visualize:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        plt.ion()
        fig, ax = plt.subplots()

    for t in range(max_steps):
        u = controller.act(x)
        x = env.step(x, u)

        # state szétbontása
        obs_xy, obs_psi, xT, yT, phiT = env._split_state(x)  # használjuk a segédfüggvényt
        dists = np.linalg.norm(obs_xy - np.array([xT, yT]), axis=1)
        d = float(dists.min())
        idx_min = int(dists.argmin())

        distances.append(d)

        if capture_step is None and d <= capture_radius:
            capture_step = t

        gx, gy = env.cfg.goal_center
        in_zone.append(np.linalg.norm([xT - gx, yT - gy]) <= env.cfg.goal_radius)

        for i in range(k):
            traj_obs[i].append((obs_xy[i, 0], obs_xy[i, 1]))
        traj_tgt.append((xT, yT))

        if visualize:
            ax.clear()
            ax.set_xlim(0, env.cfg.env_size)
            ax.set_ylim(0, env.cfg.env_size)
            ax.set_aspect("equal", "box")

            if env.cfg.task == TaskType.CAPTURE:
                status = "CAUGHT" if capture_step is not None else "running"
            elif env.cfg.task == TaskType.HERD_TO_GOAL:
                status = "herding"
            else:
                status = "running"

            ax.set_title(
                f"MPPI | d(obs,tgt)={d:.1f} (goal {env.cfg.R}) | "
                f"time-in-zone={100*np.mean(in_zone):.1f}% | {status}"
            )

            from matplotlib.patches import Circle
            # akadályok
            for ox, oy, r in env.cfg.obstacles:
                ax.add_patch(Circle((ox, oy), r, color="grey", alpha=0.3))

            # goal zóna
            ax.add_patch(Circle((gx, gy), env.cfg.goal_radius, color="green", alpha=0.2))

            # gyűrű a target körül
            ax.add_patch(
                Circle((xT, yT), env.cfg.R, color="blue",
                       fill=False, linestyle="--", alpha=0.5)
            )

            # pályák
            for i in range(k):
                if len(traj_obs[i]) > 1:
                    xo, yo = zip(*traj_obs[i])
                    ax.plot(xo, yo, "m-")

            if len(traj_tgt) > 1:
                xt, yt = zip(*traj_tgt)
                ax.plot(xt, yt, "b-")

            # aktuális pozíciók
            for i in range(k):
                ax.plot(obs_xy[i, 0], obs_xy[i, 1], "m*", label="Observer" if i == 0 else "")
            ax.plot(xT, yT, "bo", label="Target")
            ax.legend(loc="upper right")

            import matplotlib.pyplot as plt  # alias pause-hoz
            plt.pause(0.005)

        # csak CAPTURE tasknál állunk le elkapáskor
        if env.cfg.task == TaskType.CAPTURE and capture_step is not None:
            break

    if visualize:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()

    return compute_episode_metrics(
        distances=distances,
        in_zone=in_zone,
        dt=env.cfg.dt,
        capture_step=capture_step,
        cfg=env.cfg,
    )
