import tkinter as tk
from tkinter import ttk, messagebox

from env import SurveillanceEnv, SurveillanceEnvConfig, TaskType
from controllers.mppi_wrapper import MPPIWrapper
from eval.runner import run_episode


class MPPIGui(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MPPI GUI")
        self.resizable(False, False)

        row = 0

        # Task
        tk.Label(self, text="Task:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.task_var = tk.StringVar(value=TaskType.CAPTURE.value)
        self.task_combo = ttk.Combobox(
            self,
            textvariable=self.task_var,
            values=[t.value for t in TaskType],
            state="readonly",
            width=15,
        )
        self.task_combo.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        # Observer-ek száma
        tk.Label(self, text="Num observers:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.num_obs_var = tk.StringVar(value="1")
        tk.Entry(self, textvariable=self.num_obs_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5, pady=5
        )
        row += 1

        # Steps
        tk.Label(self, text="Steps (max):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.steps_var = tk.StringVar(value="600")
        tk.Entry(self, textvariable=self.steps_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5, pady=5
        )
        row += 1

        # Seed
        tk.Label(self, text="Seed:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.seed_var = tk.StringVar(value="0")
        tk.Entry(self, textvariable=self.seed_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5, pady=5
        )
        row += 1

        # Capture radius
        tk.Label(self, text="Capture radius:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.capr_var = tk.StringVar(value="1.5")
        tk.Entry(self, textvariable=self.capr_var, width=10).grid(
            row=row, column=1, sticky="w", padx=5, pady=5
        )
        row += 1

        # Random target
        self.random_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Random start (target)", variable=self.random_var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )
        row += 1

        # Run gomb
        run_btn = tk.Button(self, text="Run MPPI", command=self.run_mppi_clicked)
        run_btn.grid(row=row, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

    def run_mppi_clicked(self):
        try:
            steps = int(self.steps_var.get())
            seed = int(self.seed_var.get())
            capture_radius = float(self.capr_var.get())
            num_obs = int(self.num_obs_var.get())
            if num_obs < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input error", "Steps, seed, capture radius, num_obs legyenek érvényes számok (num_obs >= 1).")
            return

        task_str = self.task_var.get()
        try:
            task = TaskType(task_str)
        except ValueError:
            messagebox.showerror("Input error", f"Ismeretlen task: {task_str}")
            return

        random_target = self.random_var.get()

        cfg = SurveillanceEnvConfig(task=task, num_obs=num_obs)
        env = SurveillanceEnv(cfg)
        ctrl = MPPIWrapper(env)

        try:
            result = run_episode(
                env=env,
                controller=ctrl,
                random_target=random_target,
                max_steps=steps,
                capture_radius=capture_radius,
                seed=seed,
                visualize=True,
            )
        except Exception as e:
            messagebox.showerror("Run error", f"Hiba futtatás közben:\n{e}")
            return

        msg_lines = [
            f"Task: {cfg.task.value}",
            f"Num observers: {cfg.num_obs}",
            f"Steps: {result.steps}",
            f"Caught: {result.caught}",
            f"Time to capture: {result.time_to_capture}",
            f"Avg distance: {result.avg_distance:.2f}",
            f"Frac close to ring: {result.frac_close_to_ring:.2f}",
            f"Frac in goal zone: {result.frac_in_goal_zone:.2f}",
        ]
        messagebox.showinfo("Result", "\n".join(msg_lines))


if __name__ == "__main__":
    app = MPPIGui()
    app.mainloop()
