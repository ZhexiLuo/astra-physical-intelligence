import argparse
import json
import sys
from pathlib import Path

import bpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=["global", "ego", "gripper"])
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--frames", type=int, nargs=2)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "CUDA"
    scene = bpy.context.scene
    scene.cycles.device = "GPU"
    scene.cycles.samples = args.samples
    scene.render.resolution_x = args.width
    scene.render.resolution_y = round(args.width * 9 / 16)
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_file_extension = True
    scene.render.use_persistent_data = True
    if args.frames:
        scene.frame_start, scene.frame_end = args.frames
    for name in args.cameras:
        scene.camera = bpy.data.objects[name]
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        if args.frame is None:
            scene.render.filepath = str(folder / "frame-")
            bpy.ops.render.render(animation=True)
        else:
            scene.frame_set(args.frame)
            scene.render.filepath = str(folder / f"frame-{args.frame:04d}.png")
            bpy.ops.render.render(write_still=True)
    report = {"cameras": args.cameras, "samples": args.samples, "width": args.width,
              "frame": args.frame, "frame_range":[scene.frame_start,scene.frame_end],
              "devices": [device.name for device in prefs.devices if device.use]}
    suffix = f"frame-{args.frame}" if args.frame is not None else f"{scene.frame_start}-{scene.frame_end}"
    (args.output / f"render-report-{suffix}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
