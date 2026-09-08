import numpy as np
from lerobot.datasets import compute_stats
from lerobot.datasets.compute_stats import RunningQuantileStats


class Float64RunningQuantileStats(RunningQuantileStats):
    def update(self, batch: np.ndarray) -> None:
        super().update(batch.astype(np.float64, copy=False))


def install_float64_stats() -> None:
    compute_stats.RunningQuantileStats = Float64RunningQuantileStats
