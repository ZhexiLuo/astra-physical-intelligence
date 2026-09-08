import argparse
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

import torch
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.policies.act import ACTConfig
from lerobot.policies.diffusion import DiffusionConfig
from lerobot.scripts.lerobot_train import train

from src.lerobot_io import install_column_projection
from src.policy_data import file_hash


def policy_config(name: str, device: str) -> ACTConfig | DiffusionConfig:
    common = {"device": device, "push_to_hub": False,
              "pretrained_backbone_weights": None, "n_action_steps": 4}
    if name == "act":
        return ACTConfig(**common, chunk_size=16, dim_model=256, dim_feedforward=1024)
    return DiffusionConfig(
        **common, n_obs_steps=2, horizon=16, down_dims=(128, 256, 512),
        use_separate_rgb_encoder_per_camera=False, noise_scheduler_type="DDIM",
        num_inference_steps=10, do_mask_loss_for_padding=True, drop_n_last_frames=0,
    )


def run_training(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    install_column_projection()
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.repo_id, root=str(args.dataset),
                              use_imagenet_stats=False, video_backend="pyav"),
        policy=policy_config(args.policy, args.device), output_dir=args.output,
        seed=args.seed, steps=args.steps, batch_size=args.batch_size,
        num_workers=args.workers, prefetch_factor=2, log_freq=100,
        env_eval_freq=0, save_freq=args.save_freq, cudnn_deterministic=True,
        wandb=WandBConfig(enable=True, mode="offline", project="deskclean-imitation",
                          disable_artifact=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.output.parent / f"{args.output.name}.attempt.json"
    audit = {"status": "running", "started_unix": time.time(),
             "arguments": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
             "packages": {name: importlib.metadata.version(name) for name in
                          ("lerobot", "datasets", "torch", "torchvision", "diffusers", "numpy")},
             "dataset_manifest_sha256": file_hash(args.dataset / "source-manifest.json"),
             "dataset_stats_sha256": file_hash(args.dataset / "meta/stats.json"),
             "dataset_conversion_sha256": file_hash(args.dataset / "conversion.json"),
             "source_sha256": {path.name: file_hash(path) for path in
                               (Path(__file__), Path(__file__).with_name("policy_data.py"),
                                Path(__file__).with_name("lerobot_io.py"))},
             "dataset_query": "pickleable ProjectedDatasetReader; LeRobot 0.6.1 / datasets 4.8.5",
             "dataloader_multiprocessing_context": config.dataloader_multiprocessing_context,
             "official_dataset_reader_sha256": file_hash(Path(inspect.getfile(DatasetReader))),
             "backbone_initialization": "random; no pretrained weights"}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    train(config)
    audit.update(status="completed", finished_unix=time.time())
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/deskclean")
    parser.add_argument("--policy", choices=("act", "diffusion"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
