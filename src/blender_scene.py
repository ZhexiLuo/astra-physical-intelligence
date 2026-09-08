import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_assets import material, rounded_box


ROOT = Path(os.environ.get("DESKCLEAN_ROOT", Path(__file__).resolve().parents[1]))


def grid_material(spacing: float) -> bpy.types.Material:
    mat = material("Woven grid cloth", (.65,.69,.65,1), .85)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    position = nodes.new("ShaderNodeNewGeometry")
    axes = nodes.new("ShaderNodeSeparateXYZ")
    links.new(position.outputs["Position"], axes.inputs[0])
    lines = []
    for axis in ["X", "Y"]:
        scale = nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = 1 / spacing
        links.new(axes.outputs[axis], scale.inputs[0])
        fraction = nodes.new("ShaderNodeMath")
        fraction.operation = "FRACT"
        links.new(scale.outputs[0], fraction.inputs[0])
        line = nodes.new("ShaderNodeMath")
        line.operation = "LESS_THAN"
        line.inputs[1].default_value = .025
        links.new(fraction.outputs[0], line.inputs[0])
        lines.append(line)
    union = nodes.new("ShaderNodeMath")
    union.operation = "MAXIMUM"
    for index, line in enumerate(lines):
        links.new(line.outputs[0], union.inputs[index])
    mix = nodes.new("ShaderNodeMixRGB")
    mix.inputs[1].default_value = (.65,.69,.65,1)
    mix.inputs[2].default_value = (.055,.08,.08,1)
    links.new(union.outputs[0], mix.inputs[0])
    shader = nodes.get("Principled BSDF")
    links.new(mix.outputs[0], shader.inputs["Base Color"])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 700
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .12
    bump.inputs["Distance"].default_value = .0004
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return mat


def look_at(obj: bpy.types.Object, target: tuple) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def camera(name: str, position: tuple, target: tuple, lens: float = 40) -> bpy.types.Object:
    data = bpy.data.cameras.new(name)
    data.lens = lens
    data.clip_start = .015
    data.clip_end = 100
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    look_at(obj, target)
    return obj


def area_light(name: str, position: tuple, power: float, size: float, color: tuple) -> None:
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.shape, data.size, data.color = power, "DISK", size, color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    look_at(obj, (.5, 0, .5))


def add_table(spec: dict) -> None:
    table = spec["table"]
    cloth = grid_material(table["grid_spacing"])
    rounded_box("Tabletop • observed grid", tuple(table["center"]), tuple(table["half_size"]), cloth, radius=.006)
    metal = material("Graphite powdercoat", (.025,.035,.04,1), .36)
    metal.node_tree.nodes.get("Principled BSDF").inputs["Metallic"].default_value = .6
    x, y, z = table["center"]
    leg_top, leg_bottom = z - table["half_size"][2], -.005
    for i, dx in enumerate([-.36,.36]):
        for j, dy in enumerate([-.55,.55]):
            rounded_box(f"Table leg {i}{j}", (x+dx,y+dy,(leg_top+leg_bottom)/2),
                        (.022,.022,(leg_top-leg_bottom)/2), metal, radius=.007)


def load_tools(spec: dict) -> dict:
    with bpy.data.libraries.load(str(ROOT / "agent/out/assets/tools.blend"), link=False) as (source, target):
        target.objects = source.objects
    for obj in target.objects:
        bpy.context.collection.objects.link(obj)
    tools = {name: bpy.data.objects[f"{name}_visual"] for name in ["dustpan", "brush"]}
    for name, obj in tools.items():
        obj.location = spec[name]["origin"]
    return tools


def add_stage() -> None:
    floor = material("Midnight studio floor", (.026,.037,.045,1), .5)
    rounded_box("Studio floor", (.5,0,-.045), (200,200,.04), floor, radius=.001)
    area_light("Large softbox", (.1,-2.0,3.2), 360, 3, (.77,.88,1))
    area_light("Warm rim", (-1.4,1.4,2.2), 420, 2, (1,.72,.47))
    area_light("Task fill", (1.6,.5,2.0), 170, 1.4, (.78,1,.93))
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs[0].default_value = (.12,.16,.20,1)
    world.node_tree.nodes.get("Background").inputs[1].default_value = .25


def configure_render() -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.resolution_percentage = 100
    scene.render.fps = 24
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.threads_mode, scene.render.threads = "FIXED", 8
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -1.5


def set_viewport(cam: bpy.types.Object) -> None:
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = "SOLID"
                area.spaces.active.shading.color_type = "MATERIAL"
                area.spaces.active.overlay.show_overlays = False
                region = area.spaces.active.region_3d
                region.view_perspective = "CAMERA"
    bpy.context.scene.camera = cam


def build_stage(spec: dict) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    add_stage()
    add_table(spec)
    load_tools(spec)
    for index, (position, color) in enumerate(zip(spec["blocks"]["positions"], spec["blocks"]["colors"])):
        edge = spec["blocks"]["edge"] / 2
        rounded_box(f"Reference block {index}", tuple(position), (edge,edge,edge),
                    material(f"Block color {index}", tuple(color), .3), radius=.001)
    cam = camera("global", (1.9,-2.5,1.95), (.45,0,.65), 44)
    camera("ego", (.13,0,1.30), (.65,0,.73), 23)
    camera("gripper", (.43,-.27,.97), (.66,-.22,.74), 20)
    configure_render()
    set_viewport(cam)
    bpy.context.scene["stage"] = "Source-based scene study; not a simulation result"
    output = ROOT / "agent/out/scene-study.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"Scene study saved: {output}")


if __name__ == "__main__":
    spec = json.loads((ROOT / "agent/out/scene_spec.json").read_text())
    build_stage(spec)
