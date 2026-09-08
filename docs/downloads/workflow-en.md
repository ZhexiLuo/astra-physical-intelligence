# From six seconds of video to a robot world you can run

A tabletop cleaning clip is a small task with a surprisingly useful engineering test inside it: can a robot hold the dustpan, sweep loose blocks into its open cavity, and leave a record that explains why the motion worked?

Deskclean turns that observation into an executable MuJoCo scene around the commercial TIAGo++ robot model. The verified run collects **6 of 6 blocks**. Replaying the saved controls reproduces the recorded configuration trajectory with **zero numerical difference**. Disable only brush–block contact, while keeping the initial state and controls fixed, and collection falls to **0 of 6**.

The video informs a scripted skill and an inverse-kinematics controller. Tool dimensions are estimated, the base is fixed, and the tools start held by the original parallel grippers. The project does not train a policy or recover calibrated geometry from the video. These choices define what the experiment demonstrates.

## Run the complete pipeline

Extract `reproduce.zip` and run the following from the extracted project directory. Use Python 3.12, Blender 4.2.1, and an available NVIDIA CUDA device for rendering.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BLENDER=/absolute/path/to/blender
export CUDA_VISIBLE_DEVICES=0
bash scripts/reproduce.sh
```

The script resolves FFmpeg through the installed `imageio-ffmpeg` package. Each invocation creates a fresh `agent/out/reproductions/run-XXXXXXXX/` directory, copies the required project inputs there, and prints its absolute location. All new simulation, rendering, and log outputs stay inside that directory. The archived reference run remains intact.

The sequence is simulation → independent audit → USD export → visual assets → Blender replay → scene checks → three video renders. Stage logs are in the new run's `agent/log/`; videos are in its `agent/out/videos/`. On a Mac, open the supplied self-contained `scene.blend` to inspect the recorded scene without repeating the CUDA render.

## 1. Read the task before rebuilding it

The source is a six-second, 1280 × 720, 24 fps first-person video. One hand positions the dustpan while the other sweeps colored blocks with a brush and then withdraws. Three blocks are clearly visible at the start; six are visible near the end. Occlusion makes the visible count an unreliable physical state record.

The simulation therefore starts with six free bodies and preserves all six throughout the run. Nothing is spawned into the dustpan during execution.

To inspect the original observations, use the following optional commands. These manual inspection commands require `ffmpeg` and `ffprobe` on `PATH`; the main reproduction script supplies its own FFmpeg path.

```bash
mkdir -p agent/out/source
ffprobe -v quiet -show_format -show_streams inputs/deskclean.mp4
ffmpeg -i inputs/deskclean.mp4 -vf 'fps=2,scale=480:-1,tile=4x3' -frames:v 1 agent/out/source/contact-sheet.jpg
```

The modeled blocks are 22 mm wide, the dustpan opening is 280 mm wide, its cavity is 230 mm deep, and the tabletop is 730 mm high. These values are assumptions stored in `agent/out/scene_spec.json`, not measurements recovered from a calibrated camera.

## 2. Give appearance and contact a shared frame

The world uses meters with Z up. The robot faces +X, and its left side is +Y. The dustpan's local origin is at the opening center, with +Y pointing into its cavity. The brush's origin is at the bottom center of the bristles; its long axis is X and its sweep direction is +Y.

A workspace translation of `[-0.06, -0.12, 0]` meters places the task within reach. The resolved geometry is saved separately in `resolved-scene-spec.json`, so the original observation assumptions remain available.

`src/create_assets.py` builds the Blender dustpan shell, handle, entry lip, brush head, and 624 visual bristle filaments. These visuals share their origins and dimensions with the physical tools. The bristles use a rigid collision proxy in MuJoCo; the individual filaments provide appearance.

A single convex hull around a concave dustpan would close its opening. The physical model instead uses a thin bottom, an entry ramp, two sides, and a rear wall. The bottom's local y/z profile is `[[-0.015, -0.001], [0, 0.002], [0.23, 0.009]]`. The front edge tapers to zero thickness; the rest is 2 mm thick. Blender uses the same entry profile.

See the [MuJoCo collision documentation](https://mujoco.readthedocs.io/en/3.4.0/computation.html#collision-detection) for the collision representation used by the engine.

## 3. Keep the robot real

TIAGo++ is a commercial PAL Robotics platform. The model comes from Google DeepMind's MuJoCo Menagerie, pinned to revision `8161bba264d7fa7c99ca301e91e7fb44737676ad`. The required model files were checked against their Git blob hashes; the original Apache-2.0 license is preserved in the archive.

The task model removes the base free joint to hold the wheeled base fixed. It retains the robot meshes, arm joints, original joint force limits, and actuator force limits. The tools are rigidly attached at the gripper wrists, with the original fingers held at the grasp opening. This experiment covers tool use from a pre-grasped state, rather than autonomous tool acquisition.

Sources: [PAL Robotics TIAGo](https://pal-robotics.com/robot/tiago/) and the [pinned Menagerie model and license](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/pal_tiago_dual).

## 4. Separate planning from execution

`src/robot_control.py` computes forward kinematics in an independent MuJoCo state and uses SciPy `least_squares` to solve tool poses. The scripted sequence positions the dustpan, advances the brush, withdraws it, and lifts the dustpan. Wrist orientation preserves clearance around the robot shells.

The actual simulation uses a 0.002 s timestep. Its runtime sets actuator `ctrl` and advances through `mj_step`. Every block has a free joint. After initialization, the runtime does not overwrite block poses, move them with mocap, or apply scripted external forces. The saved controls are the commands actually sent to the engine.

The state contract is `state[T+1]` and `ctrl[T]`: the initial state plus every post-step state, alongside one control vector per step. Contact records include the solved time, participating bodies, distance, position, and the six-dimensional contact force. For rendering, a separate forward computation refreshes geometry transforms at the saved configurations.

## 5. Test the result, then remove its cause

An early run appeared to collect all six blocks but let them sink roughly 6.1 mm into the thin dustpan bottom. Isolating the bottom and a single block reproduced the problem without the robot. The first failing contract was contact softness. Setting the task contact time constant to 0.006 s reduced penetration to about 0.35 mm in the isolated check and 0.668 mm over the complete final rollout.

The final audit checks five properties:

1. All eight corners of each of the six blocks lie inside the dustpan cavity, with a 1 mm tolerance at the bottom.
2. Replaying the saved controls from the initial state reproduces every saved `qpos`, with maximum absolute error zero.
3. Saved states, controls, and rendering frames follow the same indexing contract.
4. The runtime advances through controls and physical integration.
5. Joint and actuator force limits match the original robot model.

The verified run collects six blocks. Maximum robot self-collision penetration is zero. The largest IK tool-origin position error is 0.232 mm. Accumulated normal impulse is 0.6191124 N·s for brush–block contact and 1.8636634 N·s for dustpan–block contact. The audit writes `success=true` only after its checks pass.

The additional runs use fresh models and the same saved control sequence. With identical initial state and brush–block contact disabled, collection drops to 0/6. Two initial block-position perturbations within ±3 mm and a +20% change to block geometry friction each collect 6/6. These are three finite perturbation checks, not a general robustness evaluation. Their outputs are preserved separately in `agent/out/validation/`.

An isolated reproduction smoke test also rebuilt the simulation from copied source and robot assets. Its `qpos`, `qvel`, and `ctrl` arrays matched the reference exactly, while hashes confirmed the five reference inputs remained unchanged.

## 6. Render the physics that actually happened

The official `mujoco.usd.exporter.USDExporter` exports the robot and blocks. The export explicitly sets 24 fps, meters, and Z up, and corrects the exporter's extra declared hidden end frame.

Blender imports the USD and bakes rigid-body world transforms into keyframes. Active USD dependencies are removed. The detailed dustpan and brush follow the same saved body poses. The delivered `.blend` has no external USD or unpacked texture dependency; the checked tool-position transfer error is approximately `2.86e-8 m`.

The three cameras are a fixed global view, an ego camera attached to the robot's head, and a local camera rigidly associated with the right gripper. The gripper camera is positioned to show contact ahead of the bristles without letting the brush head obscure the blocks.

The physical motion lasts 7.2 s. Its 174 frames at 24 fps produce a 7.25 s video. The website compares source and simulation by their relative progress rather than asserting a frame-for-frame timing match. The interactive 3D viewer is a browser copy of this recorded scene and animation; the physics execution remains in MuJoCo.

Sources: [MuJoCo USD exporter](https://mujoco.readthedocs.io/en/3.4.0/python.html#usd-exporter) and [Blender 4.2 Python API](https://docs.blender.org/api/4.2/).

## 7. Preserve the real workspace

Modeling, simulation, and rendering run through scripts. The actual Blender GUI is used to load scenes, inspect cameras, and play the recorded motion. A continuous 3 fps Xvfb capture records those sessions; the website's highlight edit is cut and accelerated from that recording. Source code, command logs, the full recording, edit intervals, and GUI events preserve the surrounding process.

The GUI control channel uses a main-thread timer to read an atomically replaced Python command file. One command completes before the next is submitted. During development, an unavailable screen context and an asynchronous workspace switch each stopped the timer with the original traceback intact. The recording continued through recovery. These were GUI inspection failures and did not change the physical rollout.

To record another session, use Linux with Xvfb and FFmpeg installed. Choose an unused display number and start Xvfb in one terminal:

```bash
Xvfb :117 -screen 0 1280x720x24
```

Once the display is ready, start the recorder from a second terminal:

```bash
mkdir -p agent/out/process
ffmpeg -f x11grab -framerate 3 -video_size 1280x720 -i :117.0 -c:v libx264 -threads 2 agent/out/process/workflow-live.mkv
```

With capture running, launch Blender from another terminal with `DISPLAY=:117`. The included `agent/scripts/blender_control.py` reads `DESKCLEAN_CONTROL_DIR`. Create that directory and an initial `command.py` containing `pass`. Submit each command by writing a temporary file and replacing `command.py` atomically. Wait for the matching command timestamp and `done` state in `status.json` before submitting another. Stop FFmpeg with SIGINT so it finishes the container.

Final rendering used one A100 GPU and two workers sharing eight logical CPUs, at 1280 × 720 and 48 samples. The Mac was used for source inspection, editing, and lightweight preview. The original recording and replay remain separate artifacts.

## What is in the package?

- `src/`: scene construction, control, USD export, and Blender preparation and rendering.
- `scripts/reproduce.sh`: the complete pipeline in a fresh output directory.
- `thirdparty/mujoco_menagerie/`: the pinned robot assets and original licenses.
- `inputs/deskclean.mp4`: the original six-second observation.
- `agent/out/simulation/`: MJCF, states, controls, contacts, resolved specification, and audited metrics.
- `agent/out/usd/`: the official exported trajectory.
- `agent/out/deskclean.blend`: the self-contained animated scene and three cameras.
- `agent/out/validation/`: negative-control and perturbation evidence.
- `agent/log/`: command output and reproduction audit records.

The downloadable artifact manifest records the ZIP and Blender file checksums and a SHA-256 for every archived member. Together, these files let you trace the rendered motion back to the controls, contacts, and modeling assumptions that produced it.
