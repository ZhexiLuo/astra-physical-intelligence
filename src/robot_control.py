import dataclasses

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclasses.dataclass
class ArmKinematics:
    """Solve tool poses on a separate MuJoCo state."""
    model: mujoco.MjModel
    data: mujoco.MjData
    side: str

    def solve(
        self, position: np.ndarray, rotation: np.ndarray, seed: np.ndarray
    ) -> tuple[np.ndarray, float]:
        joints = [
            self.model.joint(f"arm_{self.side}_{i}_joint").id
            for i in range(1, 8)
        ]
        addresses = self.model.jnt_qposadr[joints]
        bounds = self.model.jnt_range[joints].copy()
        # Keep the wrist shells clear without changing the physical limits.
        bounds[5] = [-1.18, 1.18]
        tool = "dustpan" if self.side == "left" else "brush"
        body = self.model.body(tool).id

        def residual(q: np.ndarray) -> np.ndarray:
            self.data.qpos[addresses] = q
            mujoco.mj_forward(self.model, self.data)
            current = self.data.xmat[body].reshape(3, 3)
            delta = Rotation.from_matrix(rotation.T @ current).as_rotvec()
            return np.r_[
                self.data.xpos[body] - position,
                0.05 * delta,
                0.00001 * (q - seed),
            ]

        result = least_squares(
            residual,
            seed,
            bounds=(bounds[:, 0] + 1e-6, bounds[:, 1] - 1e-6),
            max_nfev=200,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        return result.x, float(np.linalg.norm(result.fun[:3]))


def wrist_rotation(side: str) -> np.ndarray:
    if side == "left":
        return (
            Rotation.from_euler("y", -25, degrees=True).as_matrix()
            @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
        )
    return (
        Rotation.from_euler("x", 30, degrees=True).as_matrix()
        @ Rotation.from_euler("y", -45, degrees=True).as_matrix()
        @ Rotation.from_euler("x", 180, degrees=True).as_matrix()
    )


def smooth_progress(time: float, start: float, end: float) -> float:
    phase = np.clip((time - start) / (end - start), 0.0, 1.0)
    return float(phase * phase * (3.0 - 2.0 * phase))


def tool_targets(time: float, spec: dict) -> dict[str, np.ndarray]:
    pan = np.array(spec["dustpan"]["origin"], dtype=float)
    brush = np.array(spec["brush"]["origin"], dtype=float)
    brush[1] += 0.345 * smooth_progress(time, 0.6, 4.2)
    brush[2] += 0.09 * smooth_progress(time, 4.4, 5.2)
    pan[2] += 0.05 * smooth_progress(time, 5.4, 6.6)
    return {"left": pan, "right": brush}


def plan_joint_targets(
    model: mujoco.MjModel, spec: dict, duration: float = 7.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    data = mujoco.MjData(model)
    data.qpos[model.joint("torso_lift_joint").qposadr] = 0.18
    seeds = {
        "left": np.array([0.94058, -0.72287, 1.84716, 1.48907,
                          0.44403, -1.09616, -0.77964]),
        "right": np.array([0.00906, -0.27955, 1.30194, 2.08326,
                           0.24575, -0.75370, 0.84893]),
    }
    times = np.linspace(0.0, duration, round(duration * 24) + 1)
    targets = np.zeros((len(times), model.nu))
    errors = []
    solvers = {side: ArmKinematics(model, data, side) for side in seeds}
    initial = data.qpos.copy()
    for frame, time in enumerate(times):
        origins = tool_targets(float(time), spec)
        for side in ["left", "right"]:
            seeds[side], error = solvers[side].solve(
                origins[side], np.eye(3), seeds[side]
            )
            errors.append(error)
            for index, angle in enumerate(seeds[side], 1):
                name = f"arm_{side}_{index}_joint"
                targets[frame, model.actuator(f"{name}_position").id] = angle
            for finger in ["left", "right"]:
                name = f"gripper_{side}_{finger}_finger_joint"
                targets[frame, model.actuator(f"{name}_position").id] = 0.012
                data.qpos[model.joint(name).qposadr] = 0.012
        targets[frame, model.actuator("torso_lift_joint_position").id] = 0.18
        if frame == 0:
            initial = data.qpos.copy()
    return times, targets, initial, errors
