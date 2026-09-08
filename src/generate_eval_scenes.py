import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
import concurrent.futures
import copy
import functools
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from src.augmentation import (
    ARM_JOINTS, POLICY_DT, SOURCE_PARENT, Variant, build_variant, initial_arm_seeds, sample_variants,
)
from src.robot_control import ArmKinematics
from src.simulation import capture_frame, resolve_scene_spec


def geometry_checks(spec: dict) -> dict[str, bool]:
    positions = np.array(spec["blocks"]["positions"])
    half = spec["blocks"]["edge"] / 2
    rotations = Rotation.from_quat(np.array(spec["blocks"]["quaternions_wxyz"])[:, [1, 2, 3, 0]]).as_matrix()
    square = half * np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    corners = np.einsum("nij,kj->nki", rotations[:, :2, :2], square) + positions[:, None, :2]
    table = spec["table"]
    inside = (np.abs(corners - table["center"][:2])
              < np.array(table["half_size"][:2]) - 0.002).all()
    separated = True
    for left in range(6):
        for right in range(left):
            axes = np.r_[rotations[left, :2, :2].T, rotations[right, :2, :2].T]
            a, b = corners[left] @ axes.T, corners[right] @ axes.T
            gaps = np.maximum(a.min(axis=0) - b.max(axis=0), b.min(axis=0) - a.max(axis=0))
            separated &= bool(np.any(gaps > 0.001))
    height = positions[:, 2] - half - table["surface_z"]
    return {"table_bounds": bool(inside), "nonoverlapping_blocks": bool(separated),
            "table_support_height": bool(((height >= 0) & (height <= 0.002)).all())}


def shell(value: float, limits: list[float]) -> float:
    magnitude = limits[0] + (limits[1] - limits[0]) * ((2 * value) % 1)
    return float(magnitude if value >= 0.5 else -magnitude)


def shift_table_height(spec: dict, height: float) -> None:
    delta = height - spec["table"]["surface_z"]
    spec["table"]["surface_z"] = height
    spec["table"]["center"][2] += delta
    for name in ["dustpan", "brush"]:
        spec[name]["origin"][2] += delta
    for position in spec["blocks"]["positions"]:
        position[2] += delta


def apply_domain(variant: Variant, unit: np.ndarray, group: dict, source: dict) -> dict:
    domain = group["domain"]
    detail = {"domain": domain}
    if domain == "layout_shell":
        axis = group["axes"][int(unit[0] * len(group["axes"]))]
        if axis in ["base_x", "base_y"]:
            variant.base_pose[["base_x", "base_y"].index(axis)] = shell(unit[1], group["base_xy_abs_m"])
        elif axis == "base_yaw":
            variant.base_pose[2] = float(np.deg2rad(shell(unit[1], group["base_yaw_abs_deg"])))
        else:
            shift_table_height(variant.spec, source["table"]["surface_z"]
                               + shell(unit[1], group["table_height_abs_offset_m"]))
        detail["ood_axis"] = axis
    elif domain == "visual_shell":
        for key, coordinate in [("key_intensity", 0), ("ambient", 1)]:
            low, high = group[key]
            variant.visual[key] = float(low + (high - low) * unit[coordinate])
        low, high = group["table_red"]
        variant.visual["table_rgb"][0] = float(low + (high - low) * unit[2])
        variant.spec["visual"] = variant.visual
    elif domain == "dynamics_shell":
        for spec_key, key, coordinate in [("mass", "block_mass_kg", 0),
                                           ("friction", "block_sliding_friction", 1)]:
            low, high = group[key]
            variant.spec["blocks"][spec_key] = float(low + (high - low) * unit[coordinate])
    return detail


def sample_group(source: dict, name: str, group: dict) -> list[tuple[Variant, dict]]:
    count, seed = group["candidates"], group["seed"]
    variants = sample_variants(source, count, seed, 1.0, name)
    units = qmc.Sobol(d=3, scramble=True, rng=seed + 1000000).random_base2(int(np.log2(count)))
    fixed = sample_variants(source, 1, seed, 0.0, name)[0]
    proposals = []
    for variant, unit in zip(variants, units, strict=True):
        detail = apply_domain(variant, unit, group, source)
        if group["domain"] == "counterfactual":
            blocks = copy.deepcopy(variant.spec["blocks"])
            for position in blocks["positions"]:
                position[2] = source["blocks"]["positions"][0][2]
            variant.spec = copy.deepcopy(fixed.spec)
            variant.spec["blocks"] = blocks
            variant.base_pose = fixed.base_pose.copy()
            variant.motion, variant.visual = copy.deepcopy(fixed.motion), copy.deepcopy(fixed.visual)
            detail["family_id"] = group["family_id"]
        else:
            detail["family_id"] = variant.candidate_id
        proposals.append((variant, detail))
    return proposals


@functools.cache
def reference_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as saved:
        return saved["qpos"][0].copy(), saved["qvel"][0].copy()


def initialize_robot(model: mujoco.MjModel, variant: Variant,
                     reference: Path, counterfactual: bool) -> tuple[mujoco.MjData, dict]:
    data = mujoco.MjData(model)
    original_qpos, original_qvel = reference_state(reference)
    robot_joints = np.flatnonzero(model.jnt_type != mujoco.mjtJoint.mjJNT_FREE)
    qpos_ids, qvel_ids = model.jnt_qposadr[robot_joints], model.jnt_dofadr[robot_joints]
    data.qpos[qpos_ids] = original_qpos[qpos_ids]
    data.qvel[qvel_ids] = original_qvel[qvel_ids]
    residuals = {}
    if not counterfactual:
        target_rotation = Rotation.from_euler("z", variant.motion["yaw"]).as_matrix()
        for side, seed in initial_arm_seeds().items():
            tool = "dustpan" if side == "left" else "brush"
            q, error = ArmKinematics(model, data, side).solve(
                np.array(variant.spec[tool]["origin"]), target_rotation, seed)
            ids = [model.joint(f"arm_{side}_{i}_joint").qposadr[0] for i in range(1, 8)]
            data.qpos[ids] = q
            residuals[side] = error
    mujoco.mj_forward(model, data)
    report = {"strategy": "fixed_robot_reference" if counterfactual else "training_initial_tool_pose_ik_only",
              "initial_ik_position_error_m": residuals, "ik_used_as_filter": False,
              "robot_joint_names": [model.joint(index).name for index in robot_joints],
              "robot_qpos_sha256": hashlib.sha256(data.qpos[qpos_ids].tobytes()).hexdigest(),
              "robot_qvel_sha256": hashlib.sha256(data.qvel[qvel_ids].tobytes()).hexdigest()}
    return data, report


def save_scene(job: tuple) -> dict:
    variant, detail, geometry, assets, output, reference = job
    scene = build_variant(variant, assets, output)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data, initialization = initialize_robot(model, variant, reference,
                                             detail["domain"] == "counterfactual")
    frame = capture_frame(data)
    payload = {key: value[None] for key, value in frame.items()}
    payload.update(initial_qpos=data.qpos.copy(), initial_qvel=data.qvel.copy(),
                   qpos=data.qpos[None].copy(), qvel=data.qvel[None].copy(), time=np.array([0.0]),
                   frame_indices=np.array([0]), ctrl=np.empty((0, model.nu)),
                   body_names=np.array([model.body(i).name for i in range(model.nbody)]),
                   geom_names=np.array([model.geom(i).name for i in range(model.ngeom)]),
                   actuator_names=np.array([model.actuator(i).name for i in range(model.nu)]))
    np.savez_compressed(output / "initial_state.npz", **payload)
    report = {"role": "evaluation_initial_state", "candidate_id": variant.candidate_id,
              "split": variant.split, "source_parent": SOURCE_PARENT,
              "family_id": detail["family_id"], "domain": detail,
              "variant": variant.to_dict(), "scene_xml": scene.name,
              "trajectory": "initial_state.npz", "policy_dt": POLICY_DT,
              "state_joint_names": ARM_JOINTS, "geometry_checks": geometry,
              "initialization": initialization, "expert_success": None,
              "expert_rollout_executed": False, "success": None,
              "selection": "initial block geometry only, before robot IK"}
    (output / "episode.json").write_text(json.dumps(report, indent=2))
    return {"episode_dir": output.name, "candidate_id": variant.candidate_id,
            "family_id": detail["family_id"], "initialization": initialization}


def generate_group(source: dict, name: str, group: dict, assets: Path,
                   output: Path, reference: Path, workers: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    proposals = sample_group(source, name, group)
    valid = []
    with (output / "proposed.jsonl").open("w") as stream:
        for variant, detail in proposals:
            checks = geometry_checks(variant.spec)
            stream.write(json.dumps({"variant": variant.to_dict(), "domain": detail,
                                     "geometry_checks": checks}) + "\n")
            if all(checks.values()):
                valid.append((variant, detail, checks))
    selected = [valid[index] for index in range(group["episodes"])]
    jobs = [(variant, detail, checks, assets, output / f"episode-{index:06d}", reference)
            for index, (variant, detail, checks) in enumerate(selected)]
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        with (output / "initializations.jsonl").open("w") as stream:
            for record in pool.map(save_scene, jobs):
                records.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
    (output / "episodes.json").write_text(json.dumps([record["episode_dir"] for record in records], indent=2))
    summary = {"group": name, "seed": group["seed"], "proposed": len(proposals),
               "geometrically_valid": len(valid), "selected": len(selected), "saved": len(records),
               "selection_uses_ik": False, "teacher_rollouts": 0}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", type=Path, default=Path("agent/doc/plan/eval-domains.json"))
    parser.add_argument("--spec", type=Path, default=Path("agent/out/scene_spec.json"))
    parser.add_argument("--assets", type=Path, default=Path("thirdparty/mujoco_menagerie/pal_tiago_dual"))
    parser.add_argument("--reference", type=Path, default=Path("agent/out/simulation/trajectory.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--groups", nargs="+")
    args = parser.parse_args()
    domains = json.loads(args.domains.read_text())
    source = resolve_scene_spec(args.spec)
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"pid": os.getpid(), "output": str(args.output),
                      "workers": args.workers}), flush=True)
    (args.output / "registered-domains.json").write_text(json.dumps(domains, indent=2))
    for name in args.groups or list(domains["groups"]):
        report = generate_group(source, name, domains["groups"][name], args.assets,
                                args.output / name, args.reference, args.workers)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
