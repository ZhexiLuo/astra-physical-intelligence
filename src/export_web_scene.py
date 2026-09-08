import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def bake_table_color() -> None:
    table = bpy.data.objects["Tabletop • observed grid"]
    bpy.ops.object.select_all(action="DESELECT")
    table.select_set(True)
    bpy.context.view_layer.objects.active = table
    minimum = Vector((min(v.co.x for v in table.data.vertices),
                      min(v.co.y for v in table.data.vertices)))
    extent = Vector((max(v.co.x for v in table.data.vertices),
                     max(v.co.y for v in table.data.vertices))) - minimum
    for loop in table.data.loops:
        vertex = table.data.vertices[loop.vertex_index].co
        table.data.uv_layers.active.data[loop.index].uv = (
            (vertex.x - minimum.x) / extent.x, (vertex.y - minimum.y) / extent.y)
    image = bpy.data.images.new("Table grid base color", width=2048, height=2048, alpha=False)
    material = table.data.materials[0]
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    material.node_tree.nodes.active = node
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, margin=8)
    image.pack()
    shader = material.node_tree.nodes.get("Principled BSDF")
    material.node_tree.links.new(node.outputs["Color"], shader.inputs["Base Color"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    bpy.context.scene.frame_set(0)
    bake_table_color()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(args.output.resolve()), export_format="GLB", export_apply=True,
        export_animations=True, export_animation_mode="SCENE", export_bake_animation=True,
        export_anim_scene_split_object=False,
        export_frame_range=True, export_frame_step=1, export_current_frame=True,
        export_optimize_animation_size=False, export_optimize_animation_keep_anim_object=True,
        export_cameras=False, export_lights=False,
        export_draco_mesh_compression_enable=False,
        export_copyright="Robot: PAL Robotics / MuJoCo Menagerie, Apache-2.0. "
                         "Task scene: frozen TIAGo++ deskclean replay.",
    )
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
