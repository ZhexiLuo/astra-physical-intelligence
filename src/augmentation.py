import copy
import dataclasses
import functools
import json
import pathlib
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from src.robot_control import ArmKinematics, smooth_progress
from src.scene_observation import augment_scene_cameras
from src.simulation import (
    build_scene, capture_frame, contact_record, interaction_metrics, numbers,
    penetration_metrics, replay_controls, save_trajectory, step_control, task_metrics,
)


ARM_JOINTS = [f"arm_{side}_{i}_joint" for side in ["left", "right"] for i in range(1, 8)]
POLICY_DT = 0.05
SOURCE_PARENT = "tiago-deskclean-20260908T171409Z"


@dataclasses.dataclass
class Variant:
    candidate_id: str
    split: str
    scene_seed: int
    motion_seed: int
    visual_seed: int
    spec: dict
    base_pose: list[float]
    motion: dict
    visual: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def visual_parameters(unit: np.ndarray) -> dict:
    return {
        "ambient": float(0.12 + 0.23 * unit[0]),
        "key_azimuth": float(2 * np.pi * unit[1]),
        "key_elevation": float(np.deg2rad(35 + 40 * unit[2])),
        "key_intensity": float(0.45 + 0.5 * unit[3]),
        "table_rgb": (0.15 + 0.55 * unit[4:7]).tolist(),
        "floor_rgb": (0.05 + 0.35 * unit[7:10]).tolist(),
        "background_rgb": (0.08 + 0.35 * unit[10:13]).tolist(),
    }


def transform_layout(source: dict, values: np.ndarray, scale: float) -> tuple[dict, float]:
    spec = copy.deepcopy(source)
    delta = (2 * values - 1) * scale
    source_blocks = np.array(source["blocks"]["positions"])
    center = (source_blocks.min(axis=0) + source_blocks.max(axis=0)) / 2
    blocks = source_blocks.copy()
    blocks[:, :2] = center[:2] + (blocks[:, :2] - center[:2]) * (1 + 0.2 * delta[7:9])
    blocks[:, :2] += 0.015 * delta[9:21].reshape(6, 2)
    for name in ["dustpan", "brush"]:
        origin = np.array(source[name]["origin"])
        origin[0] += (blocks[:, 0].min() + blocks[:, 0].max()) / 2 - center[0]
        edge = np.max if name == "dustpan" else np.min
        origin[1] += edge(blocks[:, 1]) - edge(source_blocks[:, 1])
        spec[name]["origin"] = origin.tolist()
    yaw = float(np.deg2rad(8) * delta[6])
    rotation = Rotation.from_euler("z", yaw).as_matrix()
    offset = np.r_[0.05 * delta[4:6], 0.035 * delta[3]]
    blocks = (blocks - center) @ rotation.T + center + offset
    spec["blocks"]["positions"] = blocks.tolist()
    block_yaws = yaw + np.deg2rad(35) * delta[21:27]
    quats = Rotation.from_euler("z", block_yaws[:, None]).as_quat()[:, [3, 0, 1, 2]]
    spec["blocks"]["quaternions_wxyz"] = quats.tolist()
    for name in ["dustpan", "brush"]:
        origin = np.array(spec[name]["origin"])
        spec[name]["origin"] = (rotation @ (origin - center) + center + offset).tolist()
    spec["table"]["surface_z"] += float(offset[2])
    spec["table"]["center"][2] += float(offset[2])
    return spec, yaw


def motion_parameters(values: np.ndarray, scale: float, spec: dict, yaw: float) -> dict:
    delta = (2 * values - 1) * scale
    sweep_end = 0.6 + 3.6 * (1 + 0.15 * delta[3])
    retract_end = sweep_end + 0.2 + 0.8 * (1 + 0.15 * delta[4])
    pan_end = retract_end + 0.2 + 1.2 * (1 + 0.15 * delta[5])
    direction = Rotation.from_euler("z", yaw).as_matrix()[:, 1]
    separation = np.dot(np.array(spec["dustpan"]["origin"])
                        - spec["brush"]["origin"], direction)
    return {
        "yaw": yaw, "brush_curve": float(0.008 * delta[0]),
        "pan_curve_x": float(0.006 * delta[1]), "pan_curve_y": float(0.004 * delta[2]),
        "sweep_distance": float(separation + 0.075 + 0.005 * delta[8]),
        "brush_lift": float(0.09 * (1 + 0.15 * delta[6])),
        "pan_lift": float(0.05 * (1 + 0.15 * delta[7])),
        "sweep_end": float(sweep_end), "retract_end": float(retract_end),
        "pan_end": float(pan_end), "duration": float(np.ceil((pan_end + 0.6) / POLICY_DT) * POLICY_DT),
    }


def sample_variants(spec: dict, count: int, seed: int, scale: float,
                    split: str = "train") -> list[Variant]:
    seeds = np.random.SeedSequence(seed).generate_state(3).tolist()
    samples = [qmc.Sobol(d=d, scramble=True, rng=s).random_base2(int(np.log2(count)))
               for d, s in zip([27, 9, 13], seeds)]
    variants = []
    for index, (scene, motion, visual) in enumerate(zip(*samples)):
        resolved, yaw = transform_layout(spec, scene, scale)
        appearance = visual_parameters(visual)
        resolved["visual"] = appearance
        base = ((2 * scene[:3] - 1) * scale * [0.04, 0.04, np.deg2rad(8)]).tolist()
        variants.append(Variant(f"{split}-{seed}-{index:06d}", split, *seeds,
                                resolved, base, motion_parameters(motion, scale, resolved, yaw),
                                appearance))
    return variants


def build_variant(variant: Variant, assets: pathlib.Path, output: pathlib.Path) -> pathlib.Path:
    output.mkdir(parents=True, exist_ok=True)
    path = build_scene(variant.spec, assets, output)
    tree = ET.parse(path)
    root = tree.getroot()
    root.find("compiler").set("usethread", "false")
    base = root.find(".//body[@name='base_link']")
    base.set("pos", numbers([*variant.base_pose[:2], 0.0]))
    base.set("euler", numbers([0.0, 0.0, variant.base_pose[2]]))
    for index, quat in enumerate(variant.spec["blocks"]["quaternions_wxyz"]):
        root.find(f".//body[@name='block_{index}']").set("quat", numbers(quat))
    tree.write(path, encoding="unicode")
    augment_scene_cameras(path)
    (output / "resolved-scene-spec.json").write_text(json.dumps(variant.spec, indent=2))
    return path


def tool_path(time_s: float, variant: Variant) -> dict[str, np.ndarray]:
    motion = variant.motion
    progress = smooth_progress(time_s, 0.6, motion["sweep_end"])
    phase = np.clip((time_s - 0.6) / (motion["sweep_end"] - 0.6), 0, 1)
    bend = 16 * phase ** 2 * (1 - phase) ** 2
    rotation = Rotation.from_euler("z", motion["yaw"]).as_matrix()
    pan = np.array(variant.spec["dustpan"]["origin"])
    brush = np.array(variant.spec["brush"]["origin"])
    brush += rotation @ np.array([motion["brush_curve"] * bend,
                                  motion["sweep_distance"] * progress, 0.0])
    pan += rotation @ np.array([motion["pan_curve_x"] * bend,
                                motion["pan_curve_y"] * bend, 0.0])
    brush[2] += motion["brush_lift"] * smooth_progress(
        time_s, motion["sweep_end"] + 0.2, motion["retract_end"])
    pan[2] += motion["pan_lift"] * smooth_progress(
        time_s, motion["retract_end"] + 0.2, motion["pan_end"])
    return {"left": pan, "right": brush}


def initial_arm_seeds() -> dict[str, np.ndarray]:
    return {"left": np.array([0.94058, -0.72287, 1.84716, 1.48907, 0.44403, -1.09616, -0.77964]),
            "right": np.array([0.00906, -0.27955, 1.30194, 2.08326, 0.24575, -0.75370, 0.84893])}


def plan_variant(model: mujoco.MjModel, variant: Variant) -> tuple:
    data = mujoco.MjData(model)
    data.qpos[model.joint("torso_lift_joint").qposadr] = 0.18
    seeds = initial_arm_seeds()
    times = np.arange(round(variant.motion["duration"] / POLICY_DT) + 1) * POLICY_DT
    targets = np.zeros((len(times), model.nu))
    errors, orientation_errors = [], {side: [] for side in seeds}
    rotation = Rotation.from_euler("z", variant.motion["yaw"]).as_matrix()
    solvers = {side: ArmKinematics(model, data, side) for side in seeds}
    for frame, time_s in enumerate(times):
        origins = tool_path(float(time_s), variant)
        for side in seeds:
            seeds[side], error = solvers[side].solve(origins[side], rotation, seeds[side])
            errors.append(error)
            body = "dustpan" if side == "left" else "brush"
            actual = data.body(body).xmat.reshape(3, 3)
            orientation_errors[side].append(np.linalg.norm(
                Rotation.from_matrix(rotation.T @ actual).as_rotvec()))
            for index, angle in enumerate(seeds[side], 1):
                targets[frame, model.actuator(f"arm_{side}_{index}_joint_position").id] = angle
            for finger in ["left", "right"]:
                name = f"gripper_{side}_{finger}_finger_joint"
                targets[frame, model.actuator(f"{name}_position").id] = 0.012
                data.qpos[model.joint(name).qposadr] = 0.012
        targets[frame, model.actuator("torso_lift_joint_position").id] = 0.18
        if frame == 0:
            initial = data.qpos.copy()
    return times, targets, initial, {"position_m": float(max(errors)),
                                     "orientation_rad": {side: float(max(values))
                                                         for side, values in orientation_errors.items()}}


class JointTargetEnv:
    def __init__(self, model: mujoco.MjModel, initial_qpos: np.ndarray,
                 initial_qvel: np.ndarray | None = None) -> None:
        self.model, self.data = model, mujoco.MjData(model)
        self.initial_qpos = initial_qpos.copy()
        self.initial_qvel = np.zeros(model.nv) if initial_qvel is None else initial_qvel.copy()
        self.qpos_ids = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
        self.qvel_ids = np.array([model.joint(name).dofadr[0] for name in ARM_JOINTS])
        self.actuator_ids = np.array([model.actuator(f"{name}_position").id for name in ARM_JOINTS])
        self.nominal = np.zeros(model.nu)
        self.nominal[model.actuator("torso_lift_joint_position").id] = 0.18
        for side in ["left", "right"]:
            for finger in ["left", "right"]:
                self.nominal[model.actuator(f"gripper_{side}_{finger}_finger_joint_position").id] = 0.012
        self.reset()

    def state(self) -> np.ndarray:
        return np.r_[self.data.qpos[self.qpos_ids], self.data.qvel[self.qvel_ids]]

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def step(self, action: np.ndarray, record: Callable | None = None) -> np.ndarray:
        self.nominal[self.actuator_ids] = action
        for _ in range(25):
            step_control(self.model, self.data, self.nominal)
            if record is not None:
                record()
        mujoco.mj_forward(self.model, self.data)
        return self.state()


def initial_geometry(model: mujoco.MjModel, data: mujoco.MjData, spec: dict) -> dict:
    contacts = contact_record(model, data)
    overlap = max((-c["distance"] for c in contacts
                   if any(n.startswith("block_") for n in c["bodies"])), default=0.0)
    table = spec["table"]
    bounds = np.array(table["half_size"][:2]) - spec["blocks"]["edge"] / np.sqrt(2)
    positions = np.array([data.body(f"block_{i}").xpos[:2] for i in range(6)])
    inside = (np.abs(positions - table["center"][:2]) < bounds).all()
    return {"initial_nonoverlap": bool(overlap < 0.0001), "initial_on_table": bool(inside)}


def forbidden_collision_penetration(model: mujoco.MjModel, contacts: list[dict]) -> float:
    maximum = 0.0
    for contact in contacts:
        names = contact["bodies"]
        geoms = [model.geom(index).name for index in contact["geom_ids"]]
        arm = any(name.startswith(("arm_", "head_", "torso_", "gripper_")) for name in names)
        obstacle = any(name in ["table", "floor"] for name in geoms)
        tool = any(name in ["brush", "dustpan"] for name in names)
        if arm and (obstacle or tool):
            maximum = max(maximum, -contact["distance"])
    return maximum


@functools.cache
def original_robot(assets: pathlib.Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(assets / "scene_position.xml"))


def original_limits_preserved(model: mujoco.MjModel, assets: pathlib.Path) -> bool:
    original = original_robot(assets)
    for index in range(model.nu):
        actuator = model.actuator(index)
        source_actuator = original.actuator(actuator.name)
        joint = model.joint(model.actuator_trnid[index, 0])
        source_joint = original.joint(joint.name)
        for value, source in [(actuator.forcerange, source_actuator.forcerange),
                              (joint.range, source_joint.range),
                              (model.jnt_actfrcrange[joint.id], original.jnt_actfrcrange[source_joint.id]),
                              (model.jnt_actfrclimited[joint.id], original.jnt_actfrclimited[source_joint.id])]:
            if not np.array_equal(value, source):
                return False
    return True


def rollout_variant(model: mujoco.MjModel, variant: Variant, targets: np.ndarray,
                    initial: np.ndarray, output: pathlib.Path) -> dict:
    env = JointTargetEnv(model, initial)
    states = {"time": [0.0], "qpos": [env.data.qpos.copy()], "qvel": [env.data.qvel.copy()]}
    frames, indices, controls, contacts = [capture_frame(env.data)], [0], [], []
    observations, actions, next_states, collected_history = [], [], [], []
    phase_times = [0.0]

    def record() -> None:
        controls.append(env.data.ctrl.copy())
        for key in states:
            value = getattr(env.data, key)
            states[key].append(float(value) if key == "time" else value.copy())
        contacts.extend(contact_record(model, env.data))

    stable, waits = 0, 0
    gate = int(np.ceil(variant.motion["sweep_end"] / POLICY_DT))
    index = 0
    while index < len(targets) - 1:
        observations.append(env.state())
        action = targets[index, env.actuator_ids].astype(np.float32)
        actions.append(action)
        next_states.append(env.step(action, record))
        frames.append(capture_frame(env.data))
        phase_times.append((index + 1) * POLICY_DT)
        indices.append(len(controls))
        collected = task_metrics(model, env.data, variant.spec)["blocks_in_dustpan"]
        collected_history.append(collected)
        stable = stable + 1 if collected == 6 else 0
        if index == gate and stable < 3:
            waits += 1
            if waits == 12:
                break
        else:
            index += 1
    metrics = task_metrics(model, env.data, variant.spec)
    metrics.update(interaction_metrics(contacts, model.opt.timestep))
    metrics["max_penetration_m"] = penetration_metrics(contacts)
    metrics["duration"] = float(env.data.time)
    metrics["finite_states"] = bool(all(np.isfinite(states[key]).all() for key in states)
                                     and np.isfinite(controls).all())
    metrics["collection_gate_timeout"] = waits == 12
    metrics["final_hold_collected_min"] = int(min(collected_history[-8:]))
    metrics["forbidden_collision_penetration_m"] = forbidden_collision_penetration(model, contacts)
    pan_id = model.body("dustpan").id
    pan_rotations = Rotation.from_quat(np.array([frame["body_xquat"][pan_id]
                                                for frame in frames])[:, [1, 2, 3, 0]]).as_matrix()
    metrics["max_pan_tilt_rad"] = float(np.arccos(np.clip(pan_rotations[:, 2, 2], -1, 1)).max())
    pan_positions = np.array([frame["body_xpos"][pan_id] for frame in frames])
    lip = np.array([[-0.14, -0.015, -0.001], [0.14, -0.015, -0.001]])
    lip_z = np.einsum("fij,kj->fki", pan_rotations, lip)[:, :, 2] + pan_positions[:, None, 2]
    contact_phase = np.array(phase_times) <= variant.motion["retract_end"] + 0.2
    metrics["pan_lip_table_max_gap_m"] = float(np.abs(
        lip_z[contact_phase] - variant.spec["table"]["surface_z"]).max())
    metrics["final_pan_height_m"] = float(env.data.body("dustpan").xpos[2]
                                          - variant.spec["table"]["surface_z"])
    metrics["final_block_speed_m_s"] = float(max(np.linalg.norm(env.data.qvel[
        model.joint(f"block_{i}_free").dofadr[0]:model.joint(f"block_{i}_free").dofadr[0] + 3])
        for i in range(6)))
    save_trajectory(output, model, states, frames, indices, controls, contacts)
    with np.load(output / "trajectory.npz") as saved:
        payload = {key: saved[key] for key in saved.files}
    payload.update(state=np.array(observations, dtype=np.float32), action=np.array(actions),
                   next_state=np.array(next_states, dtype=np.float32), timestamps=np.arange(len(actions)) * POLICY_DT,
                   initial_qpos=initial, initial_qvel=np.zeros(model.nv))
    np.savez_compressed(output / "trajectory.npz", **payload)
    return metrics


def generate_episode(variant: Variant, assets: pathlib.Path, output: pathlib.Path) -> dict:
    started = time.perf_counter()
    scene = build_variant(variant, assets, output)
    model = mujoco.MjModel.from_xml_path(str(scene))
    compiled = time.perf_counter()
    _, targets, initial, ik = plan_variant(model, variant)
    planned = time.perf_counter()
    env = JointTargetEnv(model, initial)
    checks = initial_geometry(model, env.data, variant.spec)
    metrics = rollout_variant(model, variant, targets, initial, output)
    rolled = time.perf_counter()
    replay_error = replay_controls(scene, output / "trajectory.npz")
    checks.update({
        "collected_all": metrics["blocks_in_dustpan"] == 6,
        "final_hold_collected": metrics["final_hold_collected_min"] == 6,
        "collection_gate": not metrics["collection_gate_timeout"],
        "lifted_pan": metrics["final_pan_height_m"] > 0.03,
        "settled_blocks": metrics["final_block_speed_m_s"] < 0.03,
        "physical_contacts": metrics["brush_block_contacts"] > 0 and metrics["dustpan_block_contacts"] > 0,
        "no_warnings": not metrics["warnings"],
        "finite_states": metrics["finite_states"],
        "ik_position": ik["position_m"] < 0.001,
        "pan_target_orientation": ik["orientation_rad"]["left"] < 0.025,
        "pan_plane_tilt": metrics["max_pan_tilt_rad"] < 0.025,
        "pan_lip_table_gap": metrics["pan_lip_table_max_gap_m"] < 0.0025,
        "pan_penetration": metrics["max_penetration_m"]["block_dustpan"] < 0.0015,
        "robot_self_penetration": metrics["max_penetration_m"]["robot_self"] < 0.001,
        "forbidden_collision": metrics["forbidden_collision_penetration_m"] < 0.001,
        "global_penetration": metrics["max_penetration_m"]["all"] < 0.0015,
        "independent_replay": replay_error < 1e-9,
        "original_robot_limits": original_limits_preserved(model, assets),
    })
    metrics.update(ik=ik, replay_max_abs_qpos_error=replay_error)
    result = {"candidate_id": variant.candidate_id, "split": variant.split,
              "source_parent": SOURCE_PARENT, "variant": variant.to_dict(),
              "scene_seed": variant.scene_seed, "motion_seed": variant.motion_seed,
              "visual_seed": variant.visual_seed, "scene_xml": "scene.xml",
              "trajectory": "trajectory.npz", "policy_dt": POLICY_DT,
              "state_joint_names": ARM_JOINTS, "action_semantics": "nominal_absolute_joint_position",
              "success": all(checks.values()), "validity_checks": checks, "metrics": metrics,
              "wall_seconds": {"compile": compiled - started, "ik": planned - compiled,
                               "rollout": rolled - planned, "replay": time.perf_counter() - rolled}}
    (output / "episode.json").write_text(json.dumps(result, indent=2))
    return result
