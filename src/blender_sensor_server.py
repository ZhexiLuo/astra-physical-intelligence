import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import tempfile
import time

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blender_state import SceneBinding
from src.sensor_protocol import receive_packet, send_packet


def configure_render(engine: str, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = scene.render.resolution_y = 224
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 0
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.7
    if engine == "CYCLES":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "CUDA"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == "CUDA"
        scene.cycles.device, scene.cycles.samples = "GPU", samples
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_use_gpu = True
        scene.cycles.max_bounces = 3
        scene.cycles.seed = 0
    else:
        scene.eevee.taa_render_samples = samples


def sensor_cameras() -> list:
    cameras = []
    for name in ["top", "wrist"]:
        data = bpy.data.cameras.new(f"sensor_{name}")
        data.sensor_fit, data.sensor_height = "VERTICAL", 36
        data.clip_start, data.clip_end = 0.015, 100
        camera = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(camera)
        cameras.append(camera)
    return cameras


def render_state(request: dict, binding: SceneBinding, cameras: list, scratch: Path) -> tuple:
    started = time.perf_counter()
    binding.apply_state(**{key: np.asarray(request[key]) for key in
                           ["geom_xpos", "geom_xmat", "body_xpos", "body_xquat"]})
    images = []
    for index, camera in enumerate(cameras):
        matrix = Matrix(np.array(request["camera_xmat"][index]).reshape(3, 3)).to_4x4()
        matrix.translation = request["camera_xpos"][index]
        camera.matrix_world = matrix
        scene = bpy.context.scene
        scene.camera = camera
        scene.render.filepath = str(scratch / f"{index}.png")
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
        images.append(Path(scene.render.filepath).read_bytes())
    report = {"physics_time": request["time"], "renderer": "Blender",
              "engine": bpy.context.scene.render.engine,
              "render_seconds": time.perf_counter() - started,
              "objects_updated": len(binding.bindings)}
    return report, images


def serve(connection: socket.socket, binding: SceneBinding, cameras: list,
          scratch: Path, sensor_audit: dict) -> None:
    with connection.makefile("rwb") as stream:
        while (payload := receive_packet(stream)) is not None:
            request = json.loads(payload)
            if request["command"] == "configure":
                binding.configure_scene(request["spec"], request["visual"])
                for camera, fovy in zip(cameras, request["camera_fovy"]):
                    camera.data.lens = 18 / math.tan(math.radians(fovy) / 2)
                send_packet(stream, json.dumps(sensor_audit).encode())
            else:
                report, images = render_state(request, binding, cameras, scratch)
                send_packet(stream, json.dumps(report).encode())
                for image in images:
                    send_packet(stream, image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-spec", type=Path, required=True)
    parser.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE_NEXT"], default="CYCLES")
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    binding = SceneBinding(args.reference, args.reference_spec)
    configure_render(args.engine, args.samples)
    cameras = sensor_cameras()
    files = [Path(bpy.data.filepath), args.reference, args.reference_spec,
             Path(__file__), Path(__file__).with_name("blender_state.py")]
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = {"blender": bpy.app.version_string, "engine": args.engine,
             "samples": args.samples, "exposure": bpy.context.scene.view_settings.exposure,
             "denoiser": "OPENIMAGEDENOISE" if args.engine == "CYCLES" else "none",
             "denoising_use_gpu": args.engine == "CYCLES",
             "observation_resolution": [224, 224], "sha256": hashes,
             "geometry_source": "current MuJoCo state", "reference_animation": "cleared"}
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(args.socket))
        server.listen(1)
        print(json.dumps({"status": "ready", "socket": str(args.socket)}), flush=True)
        with tempfile.TemporaryDirectory(prefix="deskclean-sensor-", dir="/dev/shm") as scratch:
            while True:
                connection, _ = server.accept()
                with connection:
                    serve(connection, binding, cameras, Path(scratch), audit)


if __name__ == "__main__":
    main()
