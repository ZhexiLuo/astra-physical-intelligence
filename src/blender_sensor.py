import json
from io import BytesIO
import os
import socket
from types import TracebackType

import mujoco
import numpy as np
from PIL import Image

from src.sensor_protocol import receive_packet, send_packet


class BlenderRenderClient:
    def __init__(self, model: mujoco.MjModel, visual: dict, scene_spec: dict) -> None:
        self.model = model
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(os.environ["BLENDER_SENSOR_SOCKET"])
        self.stream = self.socket.makefile("rwb")
        self.cameras = [model.camera(name).id for name in ["top", "wrist"]]
        config = {"command": "configure", "spec": scene_spec, "visual": visual,
                  "camera_fovy": model.cam_fovy[self.cameras].tolist(),
                  "geom_names": [model.geom(i).name for i in range(model.ngeom)],
                  "body_names": [model.body(i).name for i in range(model.nbody)]}
        send_packet(self.stream, json.dumps(config).encode())
        self.config_audit = json.loads(receive_packet(self.stream))
        self.last_audit = {}

    def observe(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        mujoco.mj_camlight(self.model, data)
        request = {"command": "render", "time": float(data.time),
                   "geom_xpos": data.geom_xpos.tolist(), "geom_xmat": data.geom_xmat.tolist(),
                   "body_xpos": data.xpos.tolist(), "body_xquat": data.xquat.tolist(),
                   "camera_xpos": data.cam_xpos[self.cameras].tolist(),
                   "camera_xmat": data.cam_xmat[self.cameras].tolist()}
        send_packet(self.stream, json.dumps(request).encode())
        self.last_audit = json.loads(receive_packet(self.stream))
        images = {}
        for name in ["top", "wrist"]:
            with Image.open(BytesIO(receive_packet(self.stream))) as frame:
                images[f"images_{name}"] = np.array(frame.convert("RGB"))
        return images

    def close(self) -> None:
        self.stream.close()
        self.socket.close()

    def __enter__(self) -> "BlenderRenderClient":
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()
