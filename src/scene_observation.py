from pathlib import Path
from types import TracebackType
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from src.simulation import numbers


def camera_axes(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.r_[right, np.cross(right, forward)]


def augment_scene_cameras(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    cameras = [
        ("top", "base_link", [0.37, 0.0, 1.42], [0.64, -0.12, 0.73], 66),
        ("wrist", "brush", [-0.04, -0.14, 0.22], [0.0, 0.12, 0.0], 85),
    ]
    for name, parent, position, target, fovy in cameras:
        body = root.find(f".//body[@name='{parent}']")
        ET.SubElement(body, "camera", name=name, pos=numbers(position),
                      xyaxes=numbers(camera_axes(np.array(position), np.array(target))),
                      fovy=str(fovy))
    ET.SubElement(root.find("worldbody"), "light", name="domain_key", pos="0 -2 3",
                  directional="true", diffuse="0.7 0.7 0.7", specular="0.15 0.15 0.15",
                  dir="0 1 -1", castshadow="true")
    ET.SubElement(root.find("asset"), "texture", name="domain_sky", type="skybox",
                  builtin="gradient", rgb1="0.08 0.11 0.16", rgb2="0.2 0.24 0.3",
                  width="128", height="768")
    ET.indent(root)
    tree.write(path, encoding="unicode")


def apply_visual_domain(model: mujoco.MjModel, visual: dict) -> None:
    model.geom("table").rgba[:3] = visual["table_rgb"]
    model.geom("floor").rgba[:3] = visual["floor_rgb"]
    model.vis.headlight.ambient[:] = visual["ambient"]
    model.vis.headlight.diffuse[:] = 0.22
    model.vis.headlight.specular[:] = 0.08
    azimuth, elevation = visual["key_azimuth"], visual["key_elevation"]
    direction = np.array([np.cos(elevation) * np.cos(azimuth),
                          np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    light = model.light("domain_key")
    light.pos[:] = np.array([0.6, -0.1, 0.7]) + 3 * direction
    light.dir[:] = -direction
    light.diffuse[:] = visual["key_intensity"]
    texture = model.texture("domain_sky")
    offset = int(texture.adr[0])
    pixels = int(texture.height[0] * texture.width[0])
    sky = model.tex_data[offset:offset + pixels * 3].reshape(-1, 3)
    sky[:] = (255 * np.asarray(visual["background_rgb"])).astype(np.uint8)


class VisionRenderer:
    def __init__(self, model: mujoco.MjModel, visual: dict, size: int = 224) -> None:
        apply_visual_domain(model, visual)
        self.model = model
        self.renderer = mujoco.Renderer(model, height=size, width=size)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0

    def observe(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        mujoco.mj_camlight(self.model, data)
        observations = {}
        for camera in ["top", "wrist"]:
            self.renderer.update_scene(data, camera=camera, scene_option=self.option)
            observations[f"images_{camera}"] = self.renderer.render().copy()
        return observations

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "VisionRenderer":
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()
