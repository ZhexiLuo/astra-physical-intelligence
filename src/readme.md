# Pipeline modules

`simulation.py` builds the MJCF, executes the skill, and records physical states.
`robot_control.py` solves tool poses and produces actuator targets.
`create_assets.py` builds source-based Blender tools.
`export_usd.py` uses the official MuJoCo USD exporter.
`blender_scene.py` defines stage materials and lighting; `import_replay.py` bakes
the recorded rigid-body motion and mounts three cameras. `render.py` renders PNGs.

Run `bash scripts/reproduce.sh` from the project root. See the root README for the
required assets, environment, and output paths.

`augmentation.py` defines the continuous scene/path generator, teacher audit,
and the 20 Hz joint-target environment. `generate_demos.py` records every sampled
attempt; `generate_eval_scenes.py` constructs independent initialization-only
evaluation scenes from registered domains.

`blender_state.py` binds physical transforms to Blender geometry.
`blender_sensor_server.py` renders current state in a persistent Blender process;
`blender_sensor.py` is its synchronous two-camera client. `render_demos.py`
collects demonstrations through that same renderer. `visualize_augmentation.py`
builds the workstation array and accumulated tool trajectories.

`policy_data.py` writes LeRobot datasets with image provenance. `train_policy.py`
uses the official ACT/DP training loop; `lerobot_io.py` supplies the pinned
column-query compatibility fix. `lerobot_stats.py` uses float64 accumulation in
the official dataset statistics algorithm. `evaluate_policy.py` returns learned actions to
MuJoCo while receiving current Blender images at every observation step.

`summarize_experiments.py` reads registered offline W&B histories and physical
evaluation records, reports each seed, and pairs outcomes by scene hash.
