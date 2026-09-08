from dataclasses import dataclass
import json
import re
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


POSE_KEYS = ["body_xpos", "body_xquat", "geom_xpos", "geom_xmat", "body_names", "geom_names"]


@dataclass
class Binding:
    source: bpy.types.Object
    kind: str
    index: int
    local: Matrix


def pose_matrix(poses: dict, kind: str, index: int, frame: int | None = None) -> Matrix:
    values = poses if frame is None else {key: value[frame] for key, value in poses.items()
                                         if key not in {"body_names", "geom_names"}}
    if kind == "body":
        matrix = Quaternion(values["body_xquat"][index]).to_matrix().to_4x4()
        matrix.translation = values["body_xpos"][index]
    else:
        matrix = Matrix(values["geom_xmat"][index].reshape(3, 3)).to_4x4()
        matrix.translation = values["geom_xpos"][index]
    return matrix


def bind_template(objects: list, reference: dict) -> list[Binding]:
    bindings = []
    for obj in objects:
        if obj.type != "MESH" or obj.name == "Studio floor":
            continue
        match = re.search(r"_id(\d+)_geom", obj.name)
        if match:
            kind, index = "geom", int(match.group(1))
        elif obj.parent and obj.parent.name in {"brush_visual", "dustpan_visual"}:
            kind, index = "body", list(reference["body_names"]).index(obj.parent.name[:-7])
        elif obj.name == "Tabletop • observed grid":
            kind, index = "geom", list(reference["geom_names"]).index("table")
        else:
            kind, index = "leg", -1
        local = obj.matrix_world.copy()
        if kind != "leg":
            local = pose_matrix(reference, kind, index, 0).inverted() @ local
        bindings.append(Binding(obj, kind, index, local))
    return bindings


class SceneBinding:
    def __init__(self, reference_npz: Path, reference_spec: Path | dict) -> None:
        self.reference_spec = (json.loads(Path(reference_spec).read_text())
                               if not isinstance(reference_spec, dict) else reference_spec)
        scene = bpy.context.scene
        scene.frame_set(0)
        with np.load(reference_npz) as saved:
            reference = {key: saved[key] for key in POSE_KEYS}
        self.bindings = bind_template(list(scene.objects), reference)
        names = {"brush_visual": "brush", "dustpan_visual": "dustpan", "Head camera mount": "head_2_link"}
        self.mounts = [(bpy.data.objects[name], list(reference["body_names"]).index(body))
                       for name, body in names.items()]
        self.leg_heights = {binding.source.name: binding.source.dimensions.z
                            for binding in self.bindings if binding.kind == "leg"}
        for obj in scene.objects:
            obj.animation_data_clear()
        for binding in self.bindings:
            binding.source.parent = None
            binding.source.constraints.clear()
        self.key_light = bpy.data.objects["Large softbox"]

    def apply_state(self, geom_xpos: np.ndarray, geom_xmat: np.ndarray,
                    body_xpos: np.ndarray, body_xquat: np.ndarray) -> None:
        poses = {"geom_xpos": geom_xpos, "geom_xmat": geom_xmat,
                 "body_xpos": body_xpos, "body_xquat": body_xquat}
        for binding in self.bindings:
            if binding.kind != "leg":
                binding.source.matrix_world = pose_matrix(poses, binding.kind, binding.index) @ binding.local
        for obj, index in self.mounts:
            obj.matrix_world = pose_matrix(poses, "body", index)
        bpy.context.view_layer.update()

    def configure_scene(self, spec: dict, visual: dict) -> None:
        delta = spec["table"]["surface_z"] - self.reference_spec["table"]["surface_z"]
        for binding in self.bindings:
            if binding.kind == "leg":
                obj = binding.source
                obj.matrix_world = binding.local.copy()
                obj.location.z += delta / 2
                obj.scale.z *= (self.leg_heights[obj.name] + delta) / self.leg_heights[obj.name]
        table = bpy.data.objects["Tabletop • observed grid"].data.materials[0]
        mix = next(node for node in table.node_tree.nodes if node.type == "MIX_RGB")
        mix.inputs[1].default_value = (*visual["table_rgb"], 1)
        floor = bpy.data.objects["Studio floor"].data.materials[0]
        floor.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*visual["floor_rgb"], 1)
        world = bpy.context.scene.world.node_tree.nodes["Background"]
        world.inputs[0].default_value = (*visual["background_rgb"], 1)
        world.inputs[1].default_value = visual["ambient"]
        azimuth, elevation = visual["key_azimuth"], visual["key_elevation"]
        direction = np.array([np.cos(elevation) * np.cos(azimuth),
                              np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
        self.key_light.location = np.array([0.6, -0.1, 0.7]) + 3 * direction
        self.key_light.rotation_euler = (Vector((0.6, -0.1, 0.7)) - self.key_light.location).to_track_quat("-Z", "Y").to_euler()
        self.key_light.data.energy = 500 * visual["key_intensity"]
        bpy.context.view_layer.update()
