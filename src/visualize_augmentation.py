import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blender_state import POSE_KEYS, bind_template, pose_matrix


FPS = 20


@dataclass
class Episode:
    path: Path
    metadata: dict
    poses: dict
    times: np.ndarray

    @property
    def height(self) -> float:
        return self.metadata["variant"]["spec"]["table"]["surface_z"]


def display_offsets(count: int) -> list[tuple[float, float, float]]:
    columns = math.ceil(math.sqrt(count * 16 / 9))
    rows = math.ceil(count / columns)
    return [((i % columns - (columns - 1) / 2) * 2.6,
             (i // columns - (rows - 1) / 2) * 2.6, 0.0) for i in range(count)]


def height_color(value: float, limits: tuple[float, float]) -> tuple:
    fraction = (value - limits[0]) / max(limits[1] - limits[0], 1e-9)
    cold, warm = np.array([0.17, 0.52, 0.38]), np.array([0.90, 0.28, 0.10])
    return (*((1 - fraction) * cold + fraction * warm), 1.0)


def read_episode(path: Path) -> Episode:
    metadata = json.loads((path / "episode.json").read_text())
    with np.load(path / metadata["trajectory"]) as saved:
        poses = {key: saved[key] for key in POSE_KEYS}
        times = saved["time"][saved["frame_indices"]]
    return Episode(path, metadata, poses, times)


def keyframes(owner, path: str, values: np.ndarray, frames: np.ndarray) -> None:
    owner.animation_data_create()
    if owner.animation_data.action is None:
        owner.animation_data.action = bpy.data.actions.new(owner.name)
    action = owner.animation_data.action
    values = np.asarray(values).reshape(len(frames), -1)
    for axis in range(values.shape[1]):
        curve = action.fcurves.new(path, index=axis)
        curve.keyframe_points.add(len(frames))
        curve.keyframe_points.foreach_set("co", np.c_[frames, values[:, axis]].ravel())
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
        curve.update()


def animate_matrices(obj, matrices: list[Matrix], frames: np.ndarray) -> None:
    poses = [matrix.decompose() for matrix in matrices]
    locations, rotations, scales = map(list, zip(*poses))
    for previous, current in zip(rotations, rotations[1:]):
        if previous.dot(current) < 0:
            current.negate()
    obj.rotation_mode = "QUATERNION"
    for path, values in [("location", locations), ("rotation_quaternion", rotations), ("scale", scales)]:
        keyframes(obj, path, np.array(values), frames)


def clone(source, collection, name: str):
    obj = source.copy()
    obj.name = name
    obj.parent = None
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.constraints.clear()
    obj.animation_data_clear()
    collection.objects.link(obj)
    return obj


def pigment(name: str, color: tuple):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = 0.5
    shader.inputs["Emission Color"].default_value = color
    shader.inputs["Emission Strength"].default_value = 0.08
    return material


def label(collection, name: str, text: str, location: tuple, size: float, material):
    data = bpy.data.curves.new(name, "FONT")
    data.body, data.size, data.extrude = text, size, 0.0001
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.location = location
    data.materials.append(material)
    return obj


def line(collection, name: str, points: np.ndarray, material, radius: float):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.resolution_u, data.bevel_resolution, data.bevel_depth = 1, 3, radius
    spline = data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    spline.points.foreach_set("co", np.c_[points, np.ones(len(points))].ravel())
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    data.materials.append(material)
    return obj


def make_scene(name: str, span: float, scene=None):
    scene = bpy.data.scenes.new(name) if scene is None else scene
    scene.name = name
    scene.render.engine = "CYCLES"
    scene.cycles.device, scene.cycles.samples = "CPU", 32
    scene.cycles.use_denoising, scene.cycles.max_bounces = True, 4
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.resolution_percentage, scene.render.fps = 100, FPS
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -2.5
    scene.world = bpy.data.worlds.new(name)
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.11, 0.15, 0.19, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.45
    for number, position in enumerate([(-0.4 * span, -0.6 * span, span), (0.6 * span, 0.2 * span, span)]):
        data = bpy.data.lights.new(f"{name} softbox {number}", "AREA")
        data.energy, data.shape, data.size = 220 * span ** 2, "DISK", 0.8 * span
        obj = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(obj)
        obj.location = position
        obj.rotation_euler = (Vector((0.4, 0, 0.7)) - obj.location).to_track_quat("-Z", "Y").to_euler()
    return scene


def moving_camera(scene, target: tuple, positions: list, last_frame: int, lens: float = 42):
    data = bpy.data.cameras.new(f"{scene.name} camera")
    data.lens, data.clip_end = lens, 300
    obj = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(obj)
    matrices = [Matrix.LocRotScale(Vector(position),
                (Vector(target) - Vector(position)).to_track_quat("-Z", "Y"), Vector((1, 1, 1)))
                for position in positions]
    animate_matrices(obj, matrices, np.linspace(0, last_frame, len(positions)))
    scene.camera = obj
    scene.frame_start, scene.frame_end = 0, last_frame
    return obj


def add_workstation(scene, episode: Episode, number: int, offset: tuple, bindings: list,
                    reference_height: float, color: tuple) -> dict:
    collection = bpy.data.collections.new(f"Episode {number:03d}")
    scene.collection.children.link(collection)
    translation = Matrix.Translation(offset)
    frames = np.rint(episode.times * FPS)
    for binding in bindings:
        obj = clone(binding.source, collection, f"{number:03d}::{binding.source.name}")
        obj["source_mesh"], obj["binding_kind"] = binding.source.name, binding.kind
        obj["binding_index"], obj["pose_local"] = binding.index, [v for row in binding.local for v in row]
        if binding.kind == "leg":
            obj.matrix_world = translation @ binding.local
            delta = episode.height - reference_height
            height = binding.source.dimensions.z
            obj.location.z += delta / 2
            obj.scale.z *= (height + delta) / height
        else:
            matrices = [translation @ pose_matrix(episode.poses, binding.kind, binding.index, i) @ binding.local
                        for i in range(len(episode.times))]
            animate_matrices(obj, matrices, frames)
    accent = pigment(f"Episode {number:03d} height", color)
    label(collection, f"Episode {number:03d} label", f"{number + 1:02d}  |  TABLE {episode.height:.3f} m",
          (offset[0] - 0.1, offset[1] - 1.1, 0.006), 0.18, accent)
    return {"path": str(episode.path), "candidate_id": episode.metadata["candidate_id"],
            "collection": collection.name, "display_offset_m": offset,
            "table_height_m": episode.height, "sample_times_s": episode.times.tolist()}


def add_trajectories(scene, episodes: list[Episode], records: list, limits: tuple) -> None:
    for number, (episode, record) in enumerate(zip(episodes, records)):
        color = pigment(f"Trajectory height {number:03d}", height_color(episode.height, limits))
        start, end = 10 + number * 7, 10 + number * 7 + 14
        record["reveal_start_frame"], record["trajectory_curves"] = start, {}
        for body in ["dustpan", "brush"]:
            index = list(episode.poses["body_names"]).index(body)
            obj = line(scene.collection, f"{number:03d} {body} origin",
                       episode.poses["body_xpos"][:, index], color, 0.0015)
            keyframes(obj.data, "bevel_factor_end", [0.0, 1.0], np.array([start, end]))
            keyframes(obj, "hide_render", [True, False], np.array([start - 1, start]))
            keyframes(obj, "hide_viewport", [True, False], np.array([start - 1, start]))
            for curve in obj.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "CONSTANT"
            record["trajectory_curves"][body] = obj.name


def reference_grid(scene, table: dict, limits: tuple) -> None:
    x, y, z = table["center"]
    dx, dy = table["half_size"][:2]
    z = limits[0] - 0.012
    gray = pigment("World reference grid", (0.08, 0.14, 0.12, 1))
    white = pigment("Annotations", (0.7, 0.77, 0.66, 1))
    for value in np.linspace(x - dx, x + dx, 8):
        line(scene.collection, "World reference x", np.array([[value, y - dy, z], [value, y + dy, z]]), gray, 0.0006)
    for value in np.linspace(y - dy, y + dy, 10):
        line(scene.collection, "World reference y", np.array([[x - dx, value, z], [x + dx, value, z]]), gray, 0.0006)
    label(scene.collection, "World frame title", "RECORDED TOOL ORIGINS  /  WORLD FRAME", (x - dx, y - dy - 0.16, z), 0.038, white)
    label(scene.collection, "Reference height", f"Reference grid z={z:.3f} m  |  Curves preserve actual Z", (x - dx, y - dy - 0.22, z), 0.025, white)
    label(scene.collection, "Height legend title", "COLOR = TABLE HEIGHT (m)", (x - dx, y + dy + 0.12, z), 0.03, white)
    for i, value in enumerate(np.linspace(*limits, 9)):
        color = pigment(f"Height legend {i}", height_color(value, limits))
        start = np.array([x - dx + i * 0.05, y + dy + 0.07, z])
        line(scene.collection, "Height legend", np.array([start, start + [0.05, 0, 0]]), color, 0.008)
    label(scene.collection, "Height limits", f"{limits[0]:.3f}                              {limits[1]:.3f}", (x - dx, y + dy + 0.015, z), 0.025, white)


def source_record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def build(args) -> dict:
    started = time.perf_counter()
    folders = sorted(json.loads(args.episodes_file.read_text()))[:args.limit]
    episodes = [read_episode((args.episodes_file.parent / folder).resolve()) for folder in folders]
    with np.load(args.reference / "trajectory.npz") as saved:
        reference = {key: saved[key] for key in POSE_KEYS}
    spec = json.loads((args.reference / "resolved-scene-spec.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(args.template), load_ui=False)
    window = bpy.context.window_manager.windows[0]
    original_scene = window.scene
    original_scene.frame_set(0)
    objects = list(original_scene.objects)
    bindings = bind_template(objects, reference)
    offsets = display_offsets(len(episodes))
    span = max(offset[0] for offset in offsets) - min(offset[0] for offset in offsets) + 2.6
    limits = min(e.height for e in episodes), max(e.height for e in episodes)
    array = make_scene("Workstation array", span, original_scene)
    overlay = make_scene("World-space trajectories", 2.5)
    floor_material = bpy.data.objects["Studio floor"].data.materials[0]
    floor_material.diffuse_color = (0.025, 0.045, 0.039, 1)
    floor = floor_material.node_tree.nodes["Principled BSDF"]
    floor.inputs["Base Color"].default_value = (0.025, 0.045, 0.039, 1)
    floor.inputs["Roughness"].default_value = 0.82
    floor.inputs["Coat Weight"].default_value = 0
    for scene in [array, overlay]:
        clone(bpy.data.objects["Studio floor"], scene.collection, f"{scene.name} floor").matrix_world = bpy.data.objects["Studio floor"].matrix_world.copy()
    records = [add_workstation(array, episode, i, offset, bindings, spec["table"]["surface_z"],
                              height_color(episode.height, limits)) for i, (episode, offset) in enumerate(zip(episodes, offsets))]
    duration = max(episode.times[-1] for episode in episodes)
    moving_camera(array, (0.4, 0, 0.6), [(-0.1 * span, -0.85 * span, span),
                  (0.08 * span, -0.75 * span, 1.1 * span)], math.ceil(duration * FPS) + FPS, 36)
    add_trajectories(overlay, episodes, records, limits)
    reference_grid(overlay, spec["table"], limits)
    moving_camera(overlay, (0.6, -0.12, 0.75), [(1.45, -1.8, 2.5), (1.8, -1.65, 2.35)],
                  10 + (len(episodes) - 1) * 7 + 14 + 3 * FPS, 28)
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    window.scene = array
    for scene in [array, overlay]:
        scene.frame_set(0)
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "augmentation.blend"), compress=True)
    sources = [args.template, args.episodes_file, args.reference / "trajectory.npz",
               args.reference / "scene.xml", args.reference / "resolved-scene-spec.json"]
    sources += [episode.path / name for episode in episodes for name in ["scene.xml", "episode.json", "trajectory.npz"]]
    return {"episodes": records, "source_files": [source_record(path) for path in sources],
            "selection_rule": f"first {len(episodes)} accepted episodes in candidate directory order",
            "trajectory_coordinate_frame": "mujoco_world_m_z_up", "fps": FPS,
            "color_parameter": "variant.spec.table.surface_z", "color_limits_m": limits,
            "lighting": "Unified presentation lighting, exposure -2.5 and matte dark olive floor; source motion and world heights are retained.",
            "scenes": {s.name: {"frames": s.frame_end + 1, "camera": s.camera.name} for s in [array, overlay]},
            "build_seconds": time.perf_counter() - started,
            "peak_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("agent/out/simulation"))
    parser.add_argument("--template", type=Path, default=Path("agent/out/deskclean.blend"))
    parser.add_argument("--output", type=Path, default=Path("agent/out/augmentation-visuals"))
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    for name in ["episodes_file", "reference", "template", "output"]:
        setattr(args, name, getattr(args, name).resolve())
    report = build(args)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in {"episodes", "source_files"}}), flush=True)


if __name__ == "__main__":
    main()
