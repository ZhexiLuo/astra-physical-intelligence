import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from mujoco.usd import exporter
from pxr import UsdGeom


def export_run(run_dir: Path, output: Path) -> Path:
    model = mujoco.MjModel.from_xml_path(str(run_dir / "scene.xml"))
    trajectory = np.load(run_dir / "trajectory.npz")
    spec = json.loads((run_dir / "resolved-scene-spec.json").read_text())
    data = mujoco.MjData(model)
    for index in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[index])).name
        name = model.geom(index).name
        if body in ["dustpan", "brush"] or name in ["floor", "table"]:
            model.geom_group[index] = 5
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    exp = exporter.USDExporter(model, output_directory=output.name,
                              output_directory_root=str(output.parent), verbose=False)
    fps = spec["simulation"]["render_fps"]
    for index in trajectory["frame_indices"]:
        data.qpos[:] = trajectory["qpos"][index]
        data.qvel[:] = trajectory["qvel"][index]
        data.time = float(trajectory["time"][index])
        mujoco.mj_forward(model, data)
        exp.update_scene(data, scene_option=option)
    exp.stage.SetTimeCodesPerSecond(fps)
    exp.stage.SetFramesPerSecond(fps)
    UsdGeom.SetStageMetersPerUnit(exp.stage, 1.0)
    exp.save_scene(filetype="usdc")
    path = output / "frames" / f"frame_{exp.frame_count}.usdc"
    exp.stage.SetEndTimeCode(exp.frame_count - 1)
    exp.stage.Export(str(path))
    report = {"file":str(path), "frames":exp.frame_count, "fps":fps,
              "units":"meter", "up_axis":"Z", "source_run":str(run_dir),
              "exporter":"mujoco.usd.exporter.USDExporter",
              "excluded_visuals":["dustpan", "brush", "floor", "table"]}
    (output / "export-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("agent/out/simulation"))
    parser.add_argument("--output", type=Path, default=Path("agent/out/usd"))
    args = parser.parse_args()
    export_run(args.run_dir.resolve(), args.output.resolve())
