import numpy as np
from nengo.utils.progress import ProgressTracker

class GenericSimulator(object):
    def __init__(self, dt=0.001, progress_bar=True):
        self.dt = dt
        self.progress_bar = progress_bar
        self.n_steps = 0
        self.data = {}

    def run(self, time_in_seconds, progress_bar=None):
        steps = int(np.round(float(time_in_seconds) / self.dt))
        self.run_steps(steps, progress_bar=progress_bar)

    def run_steps(self, steps, progress_bar=None):
        if progress_bar is None:
            progress_bar = self.progress_bar
        with ProgressTracker(steps, progress_bar, "Simulating") as progress:
            for i in range(steps):
                self.step()
                progress.step()

    def step(self):
        self.n_steps += 1

    def trange(self, dt=None):
        dt = self.dt if dt is None else dt
        n_steps = int(self.n_steps * (self.dt / dt))
        return dt * np.arange(1, n_steps + 1)

    def __enter__(self):
        pass

    def close(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
