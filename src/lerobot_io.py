import torch
import lerobot.datasets.lerobot_dataset as lerobot_dataset
from lerobot.datasets.dataset_reader import DatasetReader


def query_columns(self: DatasetReader, query_indices: dict[str, list[int]]) -> dict:
    result = {}
    for key, indices in query_indices.items():
        if key in self._meta.video_keys:
            continue
        relative = (indices if self._absolute_to_relative_idx is None
                    else [self._absolute_to_relative_idx[index] for index in indices])
        result[key] = torch.stack(self.hf_dataset.select_columns([key])[relative][key])
    return result


class ProjectedDatasetReader(DatasetReader):
    _query_hf_dataset = query_columns


def install_column_projection() -> None:
    """Construct a pickleable projected reader for LeRobot 0.6.1 spawn workers."""
    lerobot_dataset.DatasetReader = ProjectedDatasetReader
