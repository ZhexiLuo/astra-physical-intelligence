import json
import math
import os
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(os.environ.get("DESKCLEAN_ROOT", Path(__file__).resolve().parents[1]))


def material(name: str, color: tuple, roughness: float = 0.3) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Coat Weight"].default_value = 0.2
    return mat


def mesh_object(name: str, vertices: list, faces: list, mat: bpy.types.Material,
                parent: bpy.types.Object | None = None) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj.parent = parent
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def empty(name: str) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    return obj


def beam(name: str, points: list, radii: list, mat: bpy.types.Material,
         parent: bpy.types.Object, sides: int = 16) -> bpy.types.Object:
    vertices, faces = [], []
    for i, (point, radius) in enumerate(zip(points, radii)):
        tangent = Vector(points[min(i + 1, len(points) - 1)]) - Vector(points[max(0, i - 1)])
        normal = tangent.normalized().cross(Vector((0, 0, 1))).normalized()
        binormal = tangent.normalized().cross(normal)
        for j in range(sides):
            angle = 2 * math.pi * j / sides
            vertex = Vector(point) + radius[0] * math.cos(angle) * normal + radius[1] * math.sin(angle) * binormal
            vertices.append(tuple(vertex))
    for i in range(len(points) - 1):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    faces.extend([tuple(reversed(range(sides))), tuple(range(len(vertices) - sides, len(vertices)))])
    return mesh_object(name, vertices, faces, mat, parent)


def rounded_box(name: str, center: tuple, half_size: tuple, mat: bpy.types.Material,
                parent: bpy.types.Object | None = None, radius: float = 0.005) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=2, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = half_size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    obj.parent = parent
    bevel = obj.modifiers.new("Soft edges", "BEVEL")
    bevel.width = radius
    bevel.segments = 3
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return obj


def dustpan(spec: dict, white: bpy.types.Material, rubber: bpy.types.Material) -> bpy.types.Object:
    root = empty("dustpan_visual")
    nx, ny = 24, 24
    vertices, faces = [], []
    for j in range(ny + 1):
        t = j / ny
        width = spec["lip_width"] * (1 - t) + spec["rear_width"] * t
        for i in range(nx + 1):
            u = 2 * i / nx - 1
            z = spec["floor_height"] + (spec["rear_floor_height"] - spec["floor_height"]) * t
            z += 0.0005 * u ** 4 * t
            vertices.append((u * width / 2, t * spec["depth"], z))
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    boundaries = [[j * (nx + 1) for j in range(ny + 1)],
                  [j * (nx + 1) + nx for j in reversed(range(ny + 1))],
                  [ny * (nx + 1) + i for i in range(nx + 1)]]
    for boundary in boundaries:
        tops = []
        for index in boundary:
            x, y, z = vertices[index]
            height = spec["wall_height"] * min(1, y / 0.05)
            tops.append(len(vertices))
            vertices.append((x, y, z + height))
        for k in range(len(boundary) - 1):
            faces.append((boundary[k], boundary[k + 1], tops[k + 1], tops[k]))
    shell = mesh_object("dustpan_shell", vertices, faces, white, root)
    solid = shell.modifiers.new("Plastic thickness", "SOLIDIFY")
    solid.thickness = spec["wall_thickness"]
    solid.offset = -1
    bevel = shell.modifiers.new("Moulded rim", "BEVEL")
    bevel.width, bevel.segments = 0.002, 3
    shell.modifiers.new("Weighted surface normals", "WEIGHTED_NORMAL")
    entry_y, entry_z = spec["floor_profile_yz"][0]
    mesh_object("dustpan_entry", [(-spec["lip_width"]/2,entry_y,entry_z),
                        (spec["lip_width"]/2,entry_y,entry_z),
                        (spec["lip_width"]/2,0,spec["floor_height"]),
                        (-spec["lip_width"]/2,0,spec["floor_height"])], [(0,1,2,3)], white, root)
    beam("dustpan_handle", [(0, .20, .038), (0, .25, .065), (0, .29, .077), tuple(spec["handle_end"])],
         [(.028,.009),(.021,.011),(.015,.010),(.017,.010)], white, root)
    rounded_box("dustpan_lip", (0, entry_y, entry_z+.0003), (spec["lip_width"] / 2, .0015, .0003), rubber, root, .0002)
    root["origin_contract"] = "Opening center, +Y points into cavity, meters, Z-up"
    return root


def brush(spec: dict, white: bpy.types.Material, bristles: bpy.types.Material) -> bpy.types.Object:
    root = empty("brush_visual")
    rounded_box("brush_head", tuple(spec["head_center"]), tuple(spec["head_half_size"]), white, root, .009)
    beam("brush_handle", [(-.085,0,.05),(-.125,0,.073),(-.17,0,.096),tuple(spec["handle_end"])],
         [(.022,.011),(.020,.010),(.014,.010),(.015,.010)], white, root)
    vertices, faces = [], []
    for i in range(52):
        for j in range(12):
            x = -.109 + i * .218 / 51
            y = -.021 + j * .042 / 11
            shift = .002 * math.sin(i * 3 + j)
            offset = len(vertices)
            for z, lean in [(0.001, .003 + shift), (.036, 0)]:
                for k in range(5):
                    angle = 2 * math.pi * k / 5
                    vertices.append((x + .0007 * math.cos(angle), y + lean + .0007 * math.sin(angle), z))
            for k in range(5):
                faces.append((offset+k,offset+(k+1)%5,offset+(k+1)%5+5,offset+k+5))
    mesh_object("brush_filaments", vertices, faces, bristles, root)
    root["origin_contract"] = "Bristle lower face center, sweep +Y, meters, Z-up"
    return root


def build_assets() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    spec_path = Path(os.environ.get("DESKCLEAN_SCENE_SPEC", str(ROOT / "agent/out/scene_spec.json")))
    spec = json.loads(spec_path.read_text())
    white = material("Ivory polypropylene", (.90,.925,.92,1), .22)
    rubber = material("Graphite lip", (.12,.16,.17,1), .7)
    fibers = material("Brush nylon", (.075,.105,.105,1), .72)
    dustpan(spec["dustpan"], white, rubber)
    brush(spec["brush"], white, fibers)
    output = ROOT / "agent/out/assets"
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "tools.blend"))
    report = {"units":"meter", "objects":len(bpy.data.objects),
              "vertices":sum(len(obj.data.vertices) for obj in bpy.data.objects if obj.type == "MESH"),
              "source":"Procedural reconstruction from deskclean.mp4; dimensions estimated"}
    (output / "asset-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    build_assets()
