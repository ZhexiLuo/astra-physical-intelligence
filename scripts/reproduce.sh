#!/usr/bin/env bash
set -eu

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
PYTHON=$("$PYTHON" -c 'import sys; print(sys.executable)')
BLENDER=${BLENDER:-blender}
FFMPEG=${FFMPEG:-$("$PYTHON" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

mkdir -p "$PROJECT_ROOT/agent/out/reproductions"
RUN_ROOT=$(mktemp -d "$PROJECT_ROOT/agent/out/reproductions/run-XXXXXXXX")
mkdir -p "$RUN_ROOT/agent/out" "$RUN_ROOT/agent/test"
cp -R "$PROJECT_ROOT/src" "$PROJECT_ROOT/thirdparty" "$RUN_ROOT/"
cp "$PROJECT_ROOT/agent/test/simulation_contract.py" "$PROJECT_ROOT/agent/test/check_blend.py" "$RUN_ROOT/agent/test/"
cp "$PROJECT_ROOT/agent/out/scene_spec.json" "$RUN_ROOT/agent/out/"
export DESKCLEAN_ROOT="$RUN_ROOT"
cd "$RUN_ROOT"
mkdir -p agent/log agent/out/render agent/out/videos
echo "Reproduction directory: $RUN_ROOT"

"$PYTHON" -m src.simulation > agent/log/simulation.log 2>&1
"$PYTHON" -m unittest agent.test.simulation_contract > agent/log/simulation-audit.log 2>&1
"$PYTHON" src/export_usd.py > agent/log/usd-export.log 2>&1
DESKCLEAN_SCENE_SPEC="$PWD/agent/out/simulation/resolved-scene-spec.json" "$BLENDER" -b -t 4 --python-exit-code 1 --python src/create_assets.py > agent/log/asset-build.log 2>&1
USD_PATH=$("$PYTHON" -c 'import json; print(json.load(open("agent/out/usd/export-report.json"))["file"])')
"$BLENDER" -b -t 4 --python-exit-code 1 --python src/import_replay.py -- --run-dir agent/out/simulation --usd "$USD_PATH" --output agent/out/deskclean.blend > agent/log/blender-import.log 2>&1
"$BLENDER" -b agent/out/deskclean.blend -t 4 --python-exit-code 1 --python agent/test/check_blend.py > agent/log/blend-validation.log 2>&1
"$BLENDER" -b agent/out/deskclean.blend -t 8 --python-exit-code 1 --python src/render.py -- --output agent/out/render > agent/log/render.log 2>&1
for camera in global ego gripper; do
    "$FFMPEG" -hide_banner -loglevel error -y -threads 2 -framerate 24 -start_number 0 -i "agent/out/render/$camera/frame-%04d.png" -c:v libx264 -threads 2 -crf 18 -pix_fmt yuv420p -movflags +faststart "agent/out/videos/$camera.mp4" > "agent/log/encode-$camera.log" 2>&1
done
echo "Videos: $RUN_ROOT/agent/out/videos/"
echo "Blender scene: $RUN_ROOT/agent/out/deskclean.blend"
echo "Logs: $RUN_ROOT/agent/log/"
