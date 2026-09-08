import argparse
import copy
import json
import os
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from src.robot_control import plan_joint_targets, wrist_rotation


def numbers(values: object) -> str:
    return " ".join(f"{value:.10g}" for value in np.ravel(values))


def resolve_scene_spec(path: pathlib.Path) -> dict:
    spec = json.loads(path.read_text())
    for name in ["dustpan", "brush"]:
        spec[name]["origin"][0] -= 0.06
        spec[name]["origin"][1] -= 0.12
    spec["table"]["center"][0] -= 0.06
    spec["table"]["center"][1] -= 0.12
    for position in spec["blocks"]["positions"]:
        position[0] -= 0.06
        position[1] -= 0.12
    spec["simulation"]["workspace_translation"] = [-0.06, -0.12, 0.0]
    spec["simulation"]["duration"] = 7.2
    spec["dustpan"]["floor_profile_yz"] = [
        [-0.015, -0.001], [0.0, 0.002], [0.23, 0.009]
    ]
    spec["dustpan"]["floor_thickness_profile"] = [0.0, 0.002, 0.002]
    return spec


def add_box(
    body: ET.Element, name: str, size: list, position: list, **attributes: str
) -> ET.Element:
    return ET.SubElement(
        body, "geom", name=name, type="box", size=numbers(size),
        pos=numbers(position), mass="0", contype="1", conaffinity="1",
        friction="0.5 0.005 0.0001", solref="0.006 1", **attributes,
    )


def add_dustpan(body: ET.Element, asset: ET.Element, spec: dict) -> None:
    width, rear, depth = spec["lip_width"] / 2, spec["rear_width"] / 2, spec["depth"]
    profile = spec["floor_profile_yz"]
    vertices = [[sign * half, y, z] for (y, z), half in
                zip(profile, [width, width, rear]) for sign in [-1, 1]]
    vertices.extend([[sign * half, y, z - 0.002] for (y, z), half in
                     zip(profile[1:], [width, rear]) for sign in [-1, 1]])
    faces = [[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4],
             [0, 7, 1], [0, 6, 7], [6, 9, 7], [6, 8, 9],
             [0, 2, 6], [2, 4, 8], [2, 8, 6], [1, 7, 3],
             [3, 7, 9], [3, 9, 5], [4, 5, 9], [4, 9, 8]]
    ET.SubElement(asset, "mesh", name="dustpan_floor_mesh",
                  vertex=numbers(vertices), face=numbers(faces))
    ET.SubElement(body, "geom", name="dustpan_floor", type="mesh",
                  mesh="dustpan_floor_mesh", rgba="0.85 0.19 0.02 1",
                  contype="1", conaffinity="1", mass="0",
                  friction="0.55 0.005 0.0001", solref="0.006 1")
    height, wall = spec["wall_height"], spec["wall_thickness"]
    length = float(np.hypot(width - rear, depth))
    for sign in [-1, 1]:
        yaw = sign * np.arctan2(width - rear, depth)
        add_box(body, f"dustpan_side_{sign}", [wall / 2, length / 2, height / 2],
                [sign * (width + rear) / 2, depth / 2, height / 2],
                euler=numbers([0, 0, yaw]), rgba="0.85 0.19 0.02 1")
    add_box(body, "dustpan_back", [rear, wall / 2, height / 2],
            [0, depth, height / 2], rgba="0.85 0.19 0.02 1")
    ET.SubElement(body, "geom", name="dustpan_handle", type="capsule",
                  fromto=numbers([0, depth, 0.028, *spec["handle_end"]]),
                  size="0.012", mass="0", contype="0", conaffinity="0",
                  rgba="0.85 0.19 0.02 1")


def add_brush(body: ET.Element, spec: dict) -> None:
    add_box(body, "brush_bristles", [0.115, 0.025, spec["bristle_height"] / 2],
            [0, 0, spec["bristle_height"] / 2], rgba="0.07 0.065 0.05 1")
    add_box(body, "brush_head", spec["head_half_size"], spec["head_center"],
            rgba="0.22 0.65 0.67 1")
    ET.SubElement(body, "geom", name="brush_handle", type="capsule",
                  fromto=numbers([-0.08, 0, 0.049, *spec["handle_end"]]),
                  size="0.012", mass="0", contype="0", conaffinity="0",
                  rgba="0.20 0.62 0.65 1")


def attach_tool(root: ET.Element, side: str, name: str, spec: dict) -> None:
    wrist = root.find(f".//body[@name='arm_{side}_7_link']")
    rotation = wrist_rotation(side).T
    tcp = np.array([0, 0, -0.225 if side == "left" else 0.225])
    position = tcp - rotation @ np.array(spec["grasp_local"])
    xyzw = Rotation.from_matrix(rotation).as_quat()
    body = ET.SubElement(wrist, "body", name=name, pos=numbers(position),
                         quat=numbers(xyzw[[3, 0, 1, 2]]))
    ET.SubElement(body, "inertial", pos="0 0.08 0.02", mass=str(spec["mass"]),
                  diaginertia="0.0003 0.0003 0.0003")
    if name == "dustpan":
        add_dustpan(body, root.find("asset"), spec)
    else:
        add_brush(body, spec)
    for finger in ["left", "right"]:
        ET.SubElement(root.find("contact"), "exclude", body1=name,
                      body2=f"gripper_{side}_{finger}_finger_link")


def build_scene(spec: dict, assets: pathlib.Path, output: pathlib.Path) -> pathlib.Path:
    root = ET.parse(assets / "tiago_dual.xml").getroot()
    root.insert(0, ET.Comment(
        " Derived from MuJoCo Menagerie pal_tiago_dual, Apache-2.0, "
        "revision 8161bba264d7fa7c99ca301e91e7fb44737676ad. "
        "Modifications: fixed base, tabletop, pre-grasped tools, free blocks, "
        "and task contact parameters. Original robot force limits retained. "
    ))
    root.find("compiler").set("meshdir", os.path.relpath(assets / "assets", output))
    root.find("option").set("timestep", str(spec["simulation"]["dt"]))
    root.find("option").set("cone", "elliptic")
    position = ET.parse(assets / "tiago_dual_position.xml").getroot()
    root.append(copy.deepcopy(position.find("default")))
    root.append(copy.deepcopy(position.find("actuator")))
    base = root.find(".//body[@name='base_link']")
    base.remove(base.find("joint[@name='reference']"))
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.1",
                  rgba="0.7 0.7 0.7 1", contype="1", conaffinity="1")
    add_box(world, "table", spec["table"]["half_size"], spec["table"]["center"],
            rgba="0.20 0.40 0.33 1")
    attach_tool(root, "left", "dustpan", spec["dustpan"])
    attach_tool(root, "right", "brush", spec["brush"])
    edge = spec["blocks"]["edge"]
    for index, (xyz, color) in enumerate(zip(spec["blocks"]["positions"], spec["blocks"]["colors"])):
        body = ET.SubElement(world, "body", name=f"block_{index}", pos=numbers(xyz))
        ET.SubElement(body, "freejoint", name=f"block_{index}_free")
        ET.SubElement(body, "geom", name=f"block_{index}", type="box",
                      size=numbers([edge / 2] * 3), rgba=numbers(color),
                      mass=str(spec["blocks"]["mass"]), contype="1", conaffinity="1",
                      friction=f"{spec['blocks']['friction']} 0.005 0.0001",
                      solref="0.006 1")
    ET.indent(root)
    path = output / "scene.xml"
    temporary = output / "scene.tmp.xml"
    ET.ElementTree(root).write(temporary, encoding="unicode")
    temporary.replace(path)
    return path


def step_control(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray) -> None:
    """Apply position targets with gravity compensation and advance physics."""
    data.ctrl[:] = target
    for index in range(4, model.nu):
        joint = model.actuator_trnid[index, 0]
        dof = model.jnt_dofadr[joint]
        data.ctrl[index] += data.qfrc_bias[dof] / model.actuator_gainprm[index, 0]
    mujoco.mj_step(model, data)


def contact_record(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    records = []
    for index, contact in enumerate(data.contact):
        names = [model.body(model.geom_bodyid[g]).name for g in [contact.geom1, contact.geom2]]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        # mj_step leaves its solved contacts at the pre-integration state.
        records.append({"time": data.time - model.opt.timestep,
                        "bodies": names, "distance": float(contact.dist),
                        "geom_ids": [int(contact.geom1), int(contact.geom2)],
                        "position": contact.pos.tolist(), "force": force.tolist()})
    return records


def capture_frame(data: mujoco.MjData) -> dict[str, np.ndarray]:
    return {"body_xpos": data.xpos.copy(), "body_xquat": data.xquat.copy(),
            "geom_xpos": data.geom_xpos.copy(), "geom_xmat": data.geom_xmat.copy()}


def task_metrics(model: mujoco.MjModel, data: mujoco.MjData, spec: dict) -> dict:
    pan = model.body("dustpan").id
    rotation = data.xmat[pan].reshape(3, 3)
    points = np.array([rotation.T @ (data.body(f"block_{i}").xpos - data.xpos[pan])
                       for i in range(6)])
    width = spec["dustpan"]["lip_width"] / 2
    rear = spec["dustpan"]["rear_width"] / 2
    depth = spec["dustpan"]["depth"]
    signs = np.array([[x, y, z] for x in [-1, 1]
                      for y in [-1, 1] for z in [-1, 1]])
    corners = []
    for index, center in enumerate(points):
        body_rotation = data.body(f"block_{index}").xmat.reshape(3, 3)
        corners.append(center + signs * (spec["blocks"]["edge"] / 2)
                       @ body_rotation.T @ rotation)
    corners = np.array(corners)
    limits = width + (rear - width) * corners[:, :, 1] / depth
    floor = spec["dustpan"]["floor_height"] + (
        spec["dustpan"]["rear_floor_height"] - spec["dustpan"]["floor_height"]
    ) * corners[:, :, 1] / depth
    inside = ((abs(corners[:, :, 0]) < limits)
              & (corners[:, :, 1] > 0) & (corners[:, :, 1] < depth)
              & (corners[:, :, 2] >= floor - 0.001)
              & (corners[:, :, 2] < spec["dustpan"]["wall_height"] + 0.03)).all(axis=1)
    return {"block_count": len(points), "blocks_in_dustpan": int(inside.sum()),
            "blocks_local_positions": points.tolist(),
            "warnings": {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}}


def penetration_metrics(contacts: list[dict]) -> dict[str, float]:
    values = {"all": 0.0, "block_dustpan": 0.0, "robot_self": 0.0}
    for contact in contacts:
        penetration = max(0.0, -contact["distance"])
        names = contact["bodies"]
        values["all"] = max(values["all"], penetration)
        if "dustpan" in names and any(n.startswith("block_") for n in names):
            values["block_dustpan"] = max(values["block_dustpan"], penetration)
        if all(n.startswith(("arm_", "head_", "torso_", "gripper_")) for n in names):
            values["robot_self"] = max(values["robot_self"], penetration)
    return values


def interaction_metrics(contacts: list[dict], timestep: float) -> dict:
    result = {}
    for tool in ["brush", "dustpan"]:
        selected = [c for c in contacts if tool in c["bodies"]
                    and any(n.startswith("block_") for n in c["bodies"])]
        result[f"{tool}_block_contacts"] = len(selected)
        result[f"{tool}_block_normal_impulse_ns"] = float(
            sum(c["force"][0] for c in selected) * timestep
        )
    return result


def save_trajectory(
    output: pathlib.Path, model: mujoco.MjModel, states: dict, frames: list,
    frame_indices: list[int], controls: list[np.ndarray], contacts: list[dict],
) -> None:
    payload = {key: np.array(value) for key, value in states.items()}
    payload.update({key: np.array([frame[key] for frame in frames]) for key in frames[0]})
    payload["ctrl"] = np.array(controls)
    payload["frame_indices"] = np.array(frame_indices)
    payload["body_names"] = np.array([model.body(i).name for i in range(model.nbody)])
    payload["geom_names"] = np.array([model.geom(i).name for i in range(model.ngeom)])
    payload["actuator_names"] = np.array([model.actuator(i).name for i in range(model.nu)])
    temporary = output / "trajectory.tmp.npz"
    np.savez_compressed(temporary, **payload)
    temporary.replace(output / "trajectory.npz")
    with (output / "contacts.jsonl").open("w") as stream:
        for contact in contacts:
            stream.write(json.dumps(contact) + "\n")


def run_simulation(spec_path: pathlib.Path, assets: pathlib.Path, output: pathlib.Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    spec = resolve_scene_spec(spec_path)
    (output / "resolved-scene-spec.json").write_text(json.dumps(spec, indent=2))
    scene = build_scene(spec, assets, output)
    model = mujoco.MjModel.from_xml_path(str(scene))
    times, targets, initial, ik_errors = plan_joint_targets(model, spec)
    data = mujoco.MjData(model)
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    states = {"time": [0.0], "qpos": [data.qpos.copy()], "qvel": [data.qvel.copy()]}
    frames, indices, controls, contacts = [capture_frame(data)], [0], [], []
    steps = round(times[-1] / model.opt.timestep)
    frame_steps = set(np.rint(np.linspace(0, steps, round(times[-1] * 24) + 1)).astype(int))
    for step in range(steps):
        target = np.array([np.interp(data.time, times, targets[:, i]) for i in range(model.nu)])
        step_control(model, data, target)
        controls.append(data.ctrl.copy())
        states["time"].append(data.time)
        states["qpos"].append(data.qpos.copy())
        states["qvel"].append(data.qvel.copy())
        contacts.extend(contact_record(model, data))
        if step + 1 in frame_steps:
            mujoco.mj_forward(model, data)
            frames.append(capture_frame(data))
            indices.append(step + 1)
    metrics = task_metrics(model, data, spec)
    metrics.update({"ik_max_position_error": max(ik_errors), "duration": data.time,
                    "max_penetration_m": penetration_metrics(contacts),
                    "grasp_model": spec["simulation"]["grasp_model"],
                    "quaternion_order": "wxyz"})
    metrics.update(interaction_metrics(contacts, model.opt.timestep))
    save_trajectory(output, model, states, frames, indices, controls, contacts)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def replay_controls(scene: pathlib.Path, trajectory: pathlib.Path) -> float:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    with np.load(trajectory) as saved:
        qpos = saved["qpos"]
        data.qpos[:] = qpos[0]
        data.qvel[:] = saved["qvel"][0]
        mujoco.mj_forward(model, data)
        maximum = 0.0
        frame_steps = set(saved["frame_indices"].tolist())
        for step, control in enumerate(saved["ctrl"], 1):
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
            maximum = max(maximum, float(np.max(abs(data.qpos - qpos[step]))))
            if step in frame_steps:
                mujoco.mj_forward(model, data)
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, default=pathlib.Path("agent/out/scene_spec.json"))
    parser.add_argument("--assets", type=pathlib.Path, default=pathlib.Path("thirdparty/mujoco_menagerie/pal_tiago_dual"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("agent/out/simulation"))
    arguments = parser.parse_args()
    print(json.dumps(run_simulation(arguments.spec, arguments.assets, arguments.output), indent=2))


if __name__ == "__main__":
    main()
