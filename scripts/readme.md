# Reproduction

From the project root, run `bash scripts/reproduce.sh` after installing the Python
dependencies and setting `BLENDER`, `CUDA_VISIBLE_DEVICES`, and optionally `FFMPEG`.
The default FFmpeg comes from the installed `imageio-ffmpeg` package.
Each invocation copies the minimal project into a new
`agent/out/reproductions/run-XXXXXXXX/` directory and prints its absolute path.
Stage logs are written to that run's `agent/log/`; videos and the Blender scene
are written to its `agent/out/`. The original verified outputs remain intact.
