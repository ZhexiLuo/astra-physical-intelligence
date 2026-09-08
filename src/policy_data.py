import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset

from src.lerobot_stats import install_float64_stats


CAMERAS = ("top", "wrist")
JOINT_NAMES = [f"arm_{side}_{index}_joint" for side in ("left", "right")
               for index in range(1, 8)]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def policy_observation(state: np.ndarray, images: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    observation = {"observation.state": torch.from_numpy(state.astype(np.float32))}
    for camera in CAMERAS:
        image = torch.from_numpy(images[f"images_{camera}"]).permute(2, 0, 1)
        observation[f"observation.images.{camera}"] = image.float() / 255
    return observation


def dataset_features() -> dict:
    features = {
        "observation.state": {
            "dtype": "float32", "shape": (28,),
            "names": [f"{joint}.{quantity}" for quantity in ("position", "velocity")
                      for joint in JOINT_NAMES],
        },
        "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "image", "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def append_episode(dataset: LeRobotDataset, episode: Path) -> dict:
    metadata = json.loads((episode / "episode.json").read_text())
    trajectory_path = episode / metadata["trajectory"]
    images_path = episode / "observations.npz"
    audit_path = episode / "render-audit.json"
    rendering = json.loads(audit_path.read_text())
    with np.load(trajectory_path) as trajectory, np.load(images_path) as images:
        states, actions = trajectory["state"], trajectory["action"]
        camera_frames = {camera: images[f"images_{camera}"] for camera in CAMERAS}
        for index, action in enumerate(actions):
            frame = {"observation.state": states[index].astype(np.float32),
                     "action": action.astype(np.float32),
                     "task": "Sweep all six blocks into the dustpan."}
            frame.update({f"observation.images.{camera}": camera_frames[camera][index]
                          for camera in CAMERAS})
            dataset.add_frame(frame)
    dataset.save_episode(parallel_encoding=False)
    return {"episode": str(episode.resolve()), "metadata": metadata,
            "renderer": rendering["renderer"], "sensor_config": rendering["sensor_config"],
            "sha256": {path.name: file_hash(path) for path in
                       (episode / "episode.json", trajectory_path, images_path, audit_path)}}


def convert_episodes(episodes: list[Path], output: Path, repo_id: str) -> None:
    install_float64_stats()
    dataset = LeRobotDataset.create(
        repo_id=repo_id, root=output, fps=20, robot_type="tiago_dual",
        features=dataset_features(), use_videos=False,
        image_writer_processes=0, image_writer_threads=2, video_backend="pyav",
    )
    sources = [append_episode(dataset, episode) for episode in episodes]
    dataset.finalize()
    (output / "source-manifest.json").write_text(json.dumps(sources, indent=2) + "\n")
    (output / "conversion.json").write_text(json.dumps({
        "lerobot_version": importlib.metadata.version("lerobot"),
        "statistics": "official sampled RunningQuantileStats with float64 accumulation",
        "source_sha256": {path.name: file_hash(path) for path in
                          (Path(__file__), Path(__file__).with_name("lerobot_stats.py"))},
        "stats_sha256": file_hash(output / "meta/stats.json"),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-file", type=Path, required=True,
                        help="JSON list of episode directories, relative to this JSON file.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/deskclean")
    args = parser.parse_args()
    directories = json.loads(args.episodes_file.read_text())
    episodes = [(args.episodes_file.parent / directory).resolve() for directory in directories]
    convert_episodes(episodes, args.output, args.repo_id)


if __name__ == "__main__":
    main()
