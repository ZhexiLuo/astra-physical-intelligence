import argparse
import json
import math
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_scene import add_stage, add_table, camera, configure_render, load_tools, set_viewport
from create_assets import material


def bake_objects(objects: list, frame_count: int) -> float:
    scene = bpy.context.scene
    matrices = []
    for frame in range(frame_count):
        scene.frame_set(frame)
        matrices.append([obj.matrix_world.copy() for obj in objects])
    for obj in objects:
        obj.parent = None
        obj.constraints.clear()
        obj.animation_data_clear()
        obj.rotation_mode = "QUATERNION"
    for frame, transforms in enumerate(matrices):
        for obj, transform in zip(objects, transforms):
            obj.location, obj.rotation_quaternion, obj.scale = transform.decompose()
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("rotation_quaternion", frame=frame)
            obj.keyframe_insert("scale", frame=frame)
    error = 0.0
    for frame in [0, frame_count // 2, frame_count - 1]:
        scene.frame_set(frame)
        for obj, expected in zip(objects, matrices[frame]):
            error = max(error, float(np.max(np.abs(np.array(obj.matrix_world) - np.array(expected)))))
    return error


def animate_body(obj: bpy.types.Object, body: str, trajectory: dict) -> None:
    index = list(trajectory["body_names"]).index(body)
    obj.rotation_mode = "QUATERNION"
    for frame, (position, quaternion) in enumerate(zip(trajectory["body_xpos"][:,index],
                                                     trajectory["body_xquat"][:,index])):
        obj.location = position
        obj.rotation_quaternion = quaternion
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("rotation_quaternion", frame=frame)


def add_cameras(trajectory: dict, tools: dict, spec: dict) -> None:
    global_camera = camera("global", (1.90,-2.70,1.95), (.36,-.05,.68), 40)
    head = bpy.data.objects.new("Head camera mount", None)
    bpy.context.collection.objects.link(head)
    animate_body(head, "head_2_link", trajectory)
    head_index = list(trajectory["body_names"]).index("head_2_link")
    position = Vector(trajectory["body_xpos"][0,head_index])
    rotation = Quaternion(trajectory["body_xquat"][0,head_index])
    world_matrix = Matrix.LocRotScale(position, rotation, Vector((1,1,1)))
    target = Vector(spec["dustpan"]["origin"]) + Vector((0,-.045,.012))
    ego = camera("ego", tuple(position + Vector((.22,0,.065))), tuple(target), 20)
    bpy.context.view_layer.update()
    ego_matrix = ego.matrix_world.copy()
    ego.parent = head
    ego.matrix_parent_inverse = Matrix.Identity(4)
    ego.matrix_basis = world_matrix.inverted() @ ego_matrix
    gripper = camera("gripper", (-.19,.10,.24), (.025,.045,.015), 19)
    gripper.parent = tools["brush"]
    set_viewport(global_camera)


def polish_robot(objects: list, spec: dict) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if obj.type != "MESH":
            continue
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        if "block_" in obj.name:
            bevel = obj.modifiers.new("Real block edge", "BEVEL")
            bevel.width, bevel.segments = .0007, 3
            obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
            index = int(re.search(r"block_(\d+)_", obj.name).group(1))
            rgb = spec["blocks"]["colors"][index][:3]
            linear = tuple(c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in rgb)
            obj.data.materials.clear()
            obj.data.materials.append(material(f"Block pigment {index}", (*linear,1), .36))
        for mat in obj.data.materials:
            mat.use_nodes = True
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    node.inputs["Roughness"].default_value = .32
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35), keep_sharp_edges=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    spec = json.loads((args.run_dir / "resolved-scene-spec.json").read_text())
    trajectory = np.load(args.run_dir / "trajectory.npz")
    frame_count = len(trajectory["frame_indices"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.usd_import(filepath=str(args.usd.resolve()), import_cameras=False,
                          import_lights=False, set_frame_range=True)
    imported = list(bpy.context.scene.objects)
    error = bake_objects(imported, frame_count)
    polish_robot(imported, spec)
    add_stage()
    add_table(spec)
    tools = load_tools(spec)
    for body, obj in tools.items():
        animate_body(obj, body, trajectory)
    add_cameras(trajectory, tools, spec)
    configure_render()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, frame_count - 1
    scene.render.fps = spec["simulation"]["render_fps"]
    for obj in scene.objects:
        if obj.animation_data and obj.animation_data.action:
            for curve in obj.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
    scene.frame_set(0)
    scene["source_run"] = str(args.run_dir)
    scene["grasp_model"] = spec["simulation"]["grasp_model"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    report = {"usd_bake_matrix_max_error":error, "frames":frame_count,
              "fps":scene.render.fps, "imported_objects":len(imported),
              "cameras":["global","ego","gripper"], "source_run":str(args.run_dir),
              "blend":str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
