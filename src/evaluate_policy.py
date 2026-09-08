import argparse
import copy
import json
import os
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio_ffmpeg
import mujoco
import numpy as np
import torch
from lerobot.policies import PreTrainedPolicy, make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.configs import PreTrainedConfig

from src.augmentation import JointTargetEnv, forbidden_collision_penetration
from src.policy_data import file_hash, policy_observation
from src.simulation import (capture_frame, contact_record, interaction_metrics,
                            penetration_metrics, save_trajectory, task_metrics)


def load_policy(checkpoint: Path, device: str, action_steps: int | None = None) -> tuple:
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    if action_steps is not None:
        config.n_action_steps = action_steps
    classes = {"act": ACTPolicy, "diffusion": DiffusionPolicy}
    policy = classes[config.type].from_pretrained(checkpoint, config=config)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


class RolloutRecorder:
    def __init__(self, env: JointTargetEnv) -> None:
        self.env = env
        self.states = {"time": [env.data.time], "qpos": [env.data.qpos.copy()],
                       "qvel": [env.data.qvel.copy()]}
        self.controls = []
        self.contacts = []
        self.frames = [capture_frame(env.data)]
        self.indices = [0]

    def record(self) -> None:
        data = self.env.data
        self.controls.append(data.ctrl.copy())
        self.states["time"].append(data.time)
        self.states["qpos"].append(data.qpos.copy())
        self.states["qvel"].append(data.qvel.copy())
        self.contacts.extend(contact_record(self.env.model, data))

    def frame(self) -> None:
        self.frames.append(capture_frame(self.env.data))
        self.indices.append(len(self.controls))

    def save(self, output: Path) -> None:
        save_trajectory(output, self.env.model, self.states, self.frames,
                        self.indices, self.controls, self.contacts)


def rollout(env: JointTargetEnv, renderer, policy: PreTrainedPolicy,
            preprocessor, postprocessor, steps: int, output: Path,
            donor_images: dict[str, np.ndarray] | None = None) -> dict:
    state = env.reset()
    policy.reset()
    recorder = RolloutRecorder(env)
    predictions, latencies, render_times, sensor_frames = [], [], [], []
    video = imageio_ffmpeg.write_frames(
        str(output / "rollout.mp4"), (448, 224), fps=20, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", ffmpeg_log_level="error",
        output_params=["-threads", "2"],
    )
    video.send(None)
    try:
        for index in range(steps):
            render_start = time.perf_counter()
            images = renderer.observe(env.data)
            sensor_frames.append(renderer.last_audit)
            render_times.append(time.perf_counter() - render_start)
            video.send(np.concatenate([images["images_top"], images["images_wrist"]], axis=1))
            policy_images = images if donor_images is None else {
                key: values[index % len(values)] for key, values in donor_images.items()}
            observation = policy_observation(state, policy_images)
            start = time.perf_counter()
            with torch.inference_mode():
                action = postprocessor(policy.select_action(preprocessor(observation)))[0]
            action = action.cpu().numpy().astype(np.float64)
            latencies.append(time.perf_counter() - start)
            predictions.append(action)
            state = env.step(action, record=recorder.record)
            recorder.frame()
        render_start = time.perf_counter()
        images = renderer.observe(env.data)
        sensor_frames.append(renderer.last_audit)
        render_times.append(time.perf_counter() - render_start)
        video.send(np.concatenate([images["images_top"], images["images_wrist"]], axis=1))
    finally:
        video.close()
    recorder.save(output)
    (output / "sensor-audit.json").write_text(json.dumps({
        "renderer": "Blender", "config": renderer.config_audit, "frames": sensor_frames,
    }, indent=2) + "\n")
    np.savez_compressed(output / "predictions.npz", action=np.asarray(predictions),
                        inference_seconds=np.asarray(latencies),
                        render_seconds=np.asarray(render_times),
                        timestamps=np.arange(steps) / 20)
    return {"duration": env.data.time, "logical_policy_hz": 20,
            "max_penetration_m": penetration_metrics(recorder.contacts),
            "forbidden_collision_penetration_m": forbidden_collision_penetration(env.model, recorder.contacts),
            "interactions": interaction_metrics(recorder.contacts, env.model.opt.timestep),
            "render_seconds_mean": float(np.mean(render_times)),
            "render_seconds_p95": float(np.quantile(render_times, 0.95)),
            "inference_seconds_mean": float(np.mean(latencies)),
            "inference_seconds_p95": float(np.quantile(latencies, 0.95))}


def save_scene_source(source: Path, spec: dict, output: Path) -> None:
    tree = ET.parse(source)
    compiler = tree.getroot().find("compiler")
    assets = (source.parent / compiler.get("meshdir")).resolve()
    compiler.set("meshdir", os.path.relpath(assets, output))
    tree.write(output / "scene.xml", encoding="unicode")
    exported_spec = copy.deepcopy(spec)
    exported_spec["simulation"]["render_fps"] = 20
    (output / "resolved-scene-spec.json").write_text(json.dumps(exported_spec, indent=2) + "\n")


def physical_outcome(metrics: dict, env: JointTargetEnv, spec: dict) -> dict:
    penetration = metrics["max_penetration_m"]
    pan_height = float(env.data.body("dustpan").xpos[2] - spec["table"]["surface_z"])
    checks = {"no_warnings": not metrics["warnings"], "pan_lifted": pan_height > 0.03,
              "global_penetration": penetration["all"] < 0.0015,
              "pan_penetration": penetration["block_dustpan"] < 0.0015,
              "robot_self_penetration": penetration["robot_self"] < 0.001,
              "forbidden_collision": metrics["forbidden_collision_penetration_m"] < 0.001}
    task_success = metrics["blocks_in_dustpan"] == 6
    return {"success": task_success, "task_success": task_success,
            "physical_success": task_success and all(checks.values()),
            "physical_checks": checks, "final_pan_height_m": pan_height,
            "success_definition": "final six-block cavity inclusion; physical_success adds shared physical checks"}


def evaluate_episode(episode: Path, output: Path, components: tuple, steps: int,
                     seed: int, donor_images: dict | None, renderer_factory) -> dict:
    output.mkdir(parents=True)
    started = time.perf_counter()
    metadata = json.loads((episode / "episode.json").read_text())
    scene_path = episode / metadata["scene_xml"]
    save_scene_source(scene_path, metadata["variant"]["spec"], output)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    with np.load(episode / metadata["trajectory"]) as trajectory:
        env = JointTargetEnv(model, trajectory["initial_qpos"], trajectory["initial_qvel"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    with renderer_factory(model, metadata["variant"]["visual"],
                          metadata["variant"]["spec"]) as renderer:
        metrics = rollout(env, renderer, *components, steps, output, donor_images)
    metrics.update(task_metrics(model, env.data, metadata["variant"]["spec"]))
    metrics.update(physical_outcome(metrics, env, metadata["variant"]["spec"]))
    wall_seconds = time.perf_counter() - started
    metrics.update(sampling_seed=seed,
                   episode_wall_seconds=wall_seconds,
                   simulation_seconds_per_wall_second=env.data.time / wall_seconds,
                   episode=str(episode.resolve()),
                   observation_mode="live" if donor_images is None else "unrelated_episode_images",
                   scene_sha256=file_hash(scene_path),
                   sensor_audit_sha256=file_hash(output / "sensor-audit.json"),
                   initialization_trajectory_sha256=file_hash(episode / metadata["trajectory"]),
                   metadata_sha256=file_hash(episode / "episode.json"))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    from src.blender_sensor import BlenderRenderClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-steps", type=int)
    parser.add_argument("--image-donor", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    directories = json.loads(args.episodes_file.read_text())
    episodes = [(args.episodes_file.parent / item).resolve() for item in directories]
    donor = None
    if args.image_donor is not None:
        with np.load(args.image_donor) as images:
            donor = {key: images[key] for key in ("images_top", "images_wrist")}
    args.output.mkdir(parents=True)
    audit = {"status": "running", "started_unix": time.time(),
             "checkpoint": str(args.checkpoint.resolve()),
             "model_sha256": file_hash(args.checkpoint / "model.safetensors"),
             "checkpoint_files_sha256": {path.name: file_hash(path) for path in
                                         sorted(args.checkpoint.iterdir()) if path.is_file()},
             "evaluation_source_sha256": file_hash(Path(__file__)),
             "episode_list_sha256": file_hash(args.episodes_file),
             "image_donor_sha256": None if args.image_donor is None else file_hash(args.image_donor),
             "arguments": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()}}
    (args.output / "evaluation.json").write_text(json.dumps(audit, indent=2) + "\n")
    components = load_policy(args.checkpoint, args.device, args.action_steps)
    with (args.output / "results.jsonl").open("w") as stream:
        for index, episode in enumerate(episodes):
            result = evaluate_episode(episode, args.output / f"{index:05d}", components,
                                      args.steps, args.seed + index, donor, BlenderRenderClient)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
    audit.update(status="completed", finished_unix=time.time())
    (args.output / "evaluation.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
