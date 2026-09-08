import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from src.blender_sensor import BlenderRenderClient


def render_episode(directory: Path) -> dict:
    started = time.perf_counter()
    metadata = json.loads((directory / "episode.json").read_text())
    source = directory / metadata["trajectory"]
    model = mujoco.MjModel.from_xml_path(str(directory / metadata["scene_xml"]))
    data = mujoco.MjData(model)
    with np.load(source) as trajectory:
        timestamps = trajectory["timestamps"]
        positions, velocities, times = trajectory["qpos"], trajectory["qvel"], trajectory["time"]
        indices = np.rint(timestamps / model.opt.timestep).astype(int)
        images = {f"images_{name}": [] for name in ["top", "wrist"]}
        audits = []
        with BlenderRenderClient(model, metadata["variant"]["visual"],
                                 metadata["variant"]["spec"]) as renderer:
            for index in indices:
                data.qpos[:] = positions[index]
                data.qvel[:] = velocities[index]
                data.time = float(times[index])
                mujoco.mj_forward(model, data)
                for name, image in renderer.observe(data).items():
                    images[name].append(image)
                audits.append(renderer.last_audit)
            sensor_config = renderer.config_audit
        temporary = directory / "observations.tmp.npz"
        np.savez_compressed(temporary, **{name: np.stack(frames) for name, frames in images.items()},
                            timestamps=timestamps, physics_indices=indices)
    output = directory / "observations.npz"
    temporary.replace(output)
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"episode": str(directory), "observations": output.name,
              "frames_per_camera": len(indices), "cameras": ["top", "wrist"],
              "resolution": [224, 224], "trajectory_sha256": source_hash,
              "renderer": "Blender", "sensor_config": sensor_config,
              "sensor_frames": audits,
              "seconds": time.perf_counter() - started, "bytes": output.stat().st_size}
    (directory / "render-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    directories = json.loads(args.episodes_file.read_text())
    episodes = [(args.episodes_file.parent / directory).resolve() for directory in directories]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.log.open("w") as log:
        for report in pool.map(render_episode, episodes):
            line = json.dumps(report)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)


if __name__ == "__main__":
    main()
