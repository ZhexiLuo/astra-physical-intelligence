# Astra: a video, a world, and the policies it can teach

[Read the project report](https://zhexiluo.github.io/astra-physical-intelligence/).

A source-grounded tabletop cleaning scene, a commercial TIAGo++ robot, an audited
MuJoCo rollout, and three Blender camera views. The supplied six-second video is
interpreted into a scripted bimanual cleaning skill. Tool dimensions are estimated;
tools start grasped, and bristles use a rigid contact proxy.

The robot retains the Menagerie geometry, joints, and force limits. Six free bodies
are moved by contact. A separate inverse-kinematics state supplies joint targets;
the simulation advances through actuator controls and `mj_step`.

The extended pipeline continuously samples scene geometry, robot placement,
tool paths, and appearance; audits every physical attempt; and renders the
accepted demonstrations for official LeRobot ACT and Diffusion Policy training.
Both training and closed-loop inference use Blender RGB. MuJoCo supplies physical
state and contact dynamics. The public report is hosted in the dedicated
[project repository](https://github.com/ZhexiLuo/astra-physical-intelligence).

## Run

Use Python 3.12, MuJoCo 3.4.0, and Blender 4.2.1. Install `requirements.txt` into an
isolated environment. The reproduction archive includes the pinned robot assets,
the input scene specification, baseline audit tests, and the verified trajectory.
Start from the current source branch, then copy only those reference resources
from the frozen archive. Its older source files are not used by the extended
learning pipeline.

```bash
git clone --branch codex/learning-report https://github.com/ZhexiLuo/astra-physical-intelligence.git
cd astra-physical-intelligence
mkdir -p agent/out/downloads agent/out/reference-package
curl -fL https://zhexiluo.github.io/astra-physical-intelligence/downloads/reproduce.zip \
  -o agent/out/downloads/reference.zip
python -m zipfile -e agent/out/downloads/reference.zip agent/out/reference-package
cp -R agent/out/reference-package/deskclean-real2sim/thirdparty .
cp -R agent/out/reference-package/deskclean-real2sim/agent/. agent/
```

Run all subsequent commands from this clone's root directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BLENDER=/path/to/blender
export CUDA_VISIBLE_DEVICES=0
bash scripts/reproduce.sh
```

Rendering requires an NVIDIA CUDA device. A Mac can inspect the self-contained
`agent/out/deskclean.blend` without rerunning the GPU render.
Set `BLENDER` to an absolute executable path. FFmpeg is supplied by the installed
`imageio-ffmpeg` package; `FFMPEG` can optionally name another executable on `PATH`
or an absolute path.

Each invocation creates `agent/out/reproductions/run-XXXXXXXX/` and copies the
source, required tests, robot assets, and input specification there. All new
simulation, render, and log files are written inside that directory. The script
prints its absolute path before execution and prints the final video, Blender,
and log paths on completion. The archive's verified reference run stays intact.

## Layout

- `src/`: scene construction, control, USD export, Blender setup, and rendering.
- `scripts/reproduce.sh`: commands for the complete pipeline.
- `thirdparty/mujoco_menagerie/pal_tiago_dual/`: original robot model and license.
- `agent/out/simulation/`: MJCF, states, controls, contacts, and audited metrics.
- `agent/out/deskclean.blend`: baked rigid-body animation and three cameras.
- `agent/log/`: complete command output.
- `agent/out/reproductions/run-XXXXXXXX/`: an isolated rerun, with its own `agent/out/` and `agent/log/`.
- `agent/doc/workflow-zh.md`: detailed Chinese workflow and modeling decisions.

## Augmentation and visual learning

Create a separate Python 3.12 learning environment with
`requirements-learning.txt`. The physics environment and its frozen reference
outputs remain separate. `lerobot[training,diffusion]==0.6.1` and
`datasets==4.8.5` are pinned; a small column-query compatibility module avoids
decoding unrelated image columns during action-window queries.

```bash
python -m venv .venv-il
.venv-il/bin/pip install -r requirements-learning.txt
```

Dataset conversion also promotes the official running-statistics accumulator to
float64. On these bright, spatially correlated images, float32 reduction produced
incorrect variance, including a zero standard deviation for a varying channel.
Sampling, aggregation, and policy normalization retain their official definitions.
The conversion manifest records the adapter and statistics hashes; training
records those hashes with its source and checkpoint provenance.

Generate the registered training candidate pool with four CPU workers:

```bash
.venv/bin/python -m src.generate_demos \
  --output agent/out/augmentation-train-v1 \
  --count 512 --seed 41001 --scale 1 --split train \
  --workers 4 --target-successes 256
```

`proposed.jsonl` and `attempts.jsonl` retain all proposals, including failures.
`successful.json` contains every passing candidate; `training-episodes.json`
selects the first requested number in candidate order. Each episode saves its
MJCF, full physical trajectory, nominal targets, contacts, and audit outcomes.
The complete pool produced 303 eligible trajectories; the registered selection
contains 256. These are teacher-generation outcomes, not learned-policy results.

Start a persistent Blender sensor in a separate terminal. The reference `.blend`,
trajectory, and resolved specification are provided in the reference artifact
package. Set `BLENDER_SENSOR_SOCKET` to an absolute path shared by the server and
its client:

```bash
export BLENDER_SENSOR_SOCKET="$PWD/agent/out/sensors/main.sock"
export BLENDER=/absolute/path/to/blender
CUDA_VISIBLE_DEVICES=0 "$BLENDER" -b agent/out/deskclean.blend -t 4 \
  --python-exit-code 1 --python src/blender_sensor_server.py -- \
  --socket "$BLENDER_SENSOR_SOCKET" \
  --reference agent/out/simulation/trajectory.npz \
  --reference-spec agent/out/simulation/resolved-scene-spec.json --samples 8
```

The reference animation is cleared before applying live physical transforms.
One socket serves one active client at a time. Parallel collection uses separate
Blender processes and sockets with disjoint episode lists.

```bash
export BLENDER_SENSOR_SOCKET="$PWD/agent/out/sensors/main.sock"
.venv/bin/python -m src.render_demos \
  --episodes-file agent/out/augmentation-train-v1/training-episodes.json \
  --workers 1 --log agent/log/training-rgb.jsonl
.venv-il/bin/python -m src.policy_data \
  --episodes-file agent/out/augmentation-train-v1/training-episodes.json \
  --output agent/out/imitation/datasets/augmented-blender \
  --repo-id local/deskclean-augmented
```

Training accepts `--policy act` or `--policy diffusion`. Set an explicit update
budget and a new output directory for each run. Checkpoints include the official
preprocessors and normalizers; offline W&B files retain the training history.

```bash
CUDA_VISIBLE_DEVICES=1 .venv-il/bin/python -m src.train_policy \
  --dataset agent/out/imitation/datasets/augmented-blender \
  --repo-id local/deskclean-augmented --policy act --seed 11 \
  --steps 50000 --save-freq 10000 --batch-size 32 --workers 4 \
  --output agent/out/imitation/train/augmented-act-seed11
```

Evaluation consumes a JSON list of initialization-only scene directories and a
checkpoint. It waits for current Blender images, selects a fourteen-joint target,
and advances exactly 25 physical steps. Image shuffling is an explicit optional
intervention, recorded in the evaluation manifest.

Generate the registered initialization sets without expert rollouts:

```bash
.venv/bin/python -m src.generate_eval_scenes \
  --domains configs/evaluation-domains.json \
  --output agent/out/evaluation-scenes-v1 --workers 4
```

```bash
export BLENDER_SENSOR_SOCKET="$PWD/agent/out/sensors/main.sock"
CUDA_VISIBLE_DEVICES=1 .venv-il/bin/python -m src.evaluate_policy \
  --checkpoint agent/out/imitation/train/augmented-act-seed11/checkpoints/050000/pretrained_model \
  --episodes-file agent/out/evaluation-scenes-v1/test_id/episodes.json \
  --output agent/out/imitation/eval/augmented-act-seed11-test-id --steps 180 --action-steps 16
```

Evaluation saves predicted actions, all physical states and contacts, current
Blender sensor audits, a two-camera video, and separate task/physical outcomes.
The logical control frequency is 20 Hz; rendering pauses simulated time, so this
does not imply real-time wall-clock operation. The saved scene specification
sets exported playback to the same 20 Hz.

`--action-steps` changes the number of predicted actions executed before the next
prediction, while retaining the checkpoint weights and observation cadence.
Omitting it preserves the checkpoint configuration. The registered comparison
uses fixed 50,000-update checkpoints and selects one execution window per method
on forty validation scenes: 4 or 16 actions for ACT, 4 or 15 for Diffusion Policy.
Both training conditions and all three seeds share the selected window. Selection
maximizes physical successes, then collection successes, with ties retaining four
actions. All candidate validation outcomes are retained; the selected window is
frozen before evaluating the separate test domains.

All twelve formal runs completed 50,000 updates. All 960 validation rollouts and
3,480 test or image-intervention rollouts are complete. Validation selected
16 executed actions per prediction for ACT and 15 for Diffusion Policy. The
evaluation command above uses the selected ACT setting. The forty validation
scenes are reused across six models of each method;
their 240 model-scene outcomes per candidate are not independent scene samples.
Training curves, validation selection, and test generalization have separate
roles in the report. A collected block count does not substitute for the physical
criteria or prove reliance on the image stream.

## Provenance

TIAGo++ is a PAL Robotics product. Robot assets come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/pal_tiago_dual),
revision `8161bba264d7fa7c99ca301e91e7fb44737676ad`, under Apache-2.0.
The generated task MJCF fixes the base and adds the table, tools, and free blocks.
Original model license and notices are retained in the archive.
The site's pinned model-viewer bundle and its dependency licenses are preserved
in `docs/vendor/`; those licenses apply to their upstream components.

## Learning artifacts

The [v0.2.0 release](https://github.com/ZhexiLuo/astra-physical-intelligence/releases/tag/v0.2.0) provides portable source, its continuation, teacher and Blender training data, twelve checkpoints, and a compact bundle of four fixed presentation cases. Start with `reproduce-learning-v2.zip`, then extract `reproduce-continuation-v1.zip` into the same parent directory and follow its instructions.

The [English report](https://zhexiluo.github.io/astra-physical-intelligence/) includes all aggregate evaluation results, interactive scenes, and selected videos. Full per-rollout evaluation archives remain on the original server and are not part of the public release.
