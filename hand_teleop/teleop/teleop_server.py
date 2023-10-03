# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import dataclasses
import threading
import time
from copy import deepcopy
from typing import List, Optional

import numpy as np
import zmq


@dataclasses.dataclass
class InitializationConfig:
    init_ee_pose: np.ndarray
    robot_base_pose: np.ndarray
    init_qpos: np.ndarray
    joint_names: List[str]

    def validate(self, dof: int):
        if len(self.init_ee_pose) != 7:
            raise ValueError(f"init_ee_pose should be a 7d vector")
        if len(self.robot_base_pose) != 7:
            raise ValueError(f"robot_base_pose should be a 7d vector")
        if len(self.init_qpos) != dof:
            raise ValueError(f"init_qpos should be a {dof}d vector, the same dim as dof")
        if len(self.joint_names) != dof:
            raise ValueError(f"joint_names should be a {dof}d vector, the same dim as dof")

    def get_joint_index_mapping(self, server_side_joint_names) -> np.ndarray:
        # Build index mapping.
        # Note that retargeting, lula, and simulation of the same robot may have different joint order,
        # Especially for multi-finger robot which is a kinematic tree rather than a single kinematics chain
        # lula_qpos[index_lula2optimizer] = retargeted_qpos
        # server_side_qpos[index_server2client] = client_side_qpos
        index_client2server = [self.joint_names.index(name) for name in server_side_joint_names]
        index_client2server = np.array(index_client2server, dtype=int)
        return index_client2server

    def to_dict(self):
        return dict(
            init_ee_pose=self.init_ee_pose.tolist(),
            robot_base_pose=self.robot_base_pose.tolist(),
            init_qpos=self.init_qpos.tolist(),
            joint_names=self.joint_names,
        )

    @classmethod
    def from_dict(cls, config):
        return InitializationConfig(
            init_ee_pose=np.array(config["init_ee_pose"]),
            robot_base_pose=np.array(config["robot_base_pose"]),
            init_qpos=np.array(config["init_qpos"]),
            joint_names=config["joint_names"]
        )


class TeleopServer:
    def __init__(self, port: int, host="localhost"):
        # Create socket
        self.ctx = zmq.Context()
        if host == "localhost":
            pub_bind_to = f"tcp://*:{port}"
        else:
            pub_bind_to = f"tcp://{host}:{port}"
        self.pub_socket: Optional[zmq.Socket] = None
        self.init_bind_to = ':'.join(pub_bind_to.split(':')[:-1] + [str(int(pub_bind_to.split(':')[-1]) + 1)])
        self.pub_bind_to = pub_bind_to

        print(f"TeleopServer: Waiting for connection to {pub_bind_to}")

        # Monitor re-initialization with a different thread
        self._lock = threading.Lock()
        self._shared_initialized = False
        self._shared_last_init_config: Optional[InitializationConfig] = None
        self._thread = threading.Thread(target=self.handle_init_config_request)
        self._thread.start()

    def send_teleop_cmd(self, target_qpos: np.ndarray, ee_pose: np.ndarray):
        self.pub_socket.send_pyobj(dict(target_qpos=target_qpos, ee_pose=ee_pose))

    def handle_init_config_request(self):
        try:
            while True:
                init_socket = self.ctx.socket(zmq.REP)
                init_socket.bind(self.init_bind_to)
                print("Starting new handling cycle for initialization config.")

                init_config_dict = init_socket.recv_json()
                init_socket.close()
                if self.pub_socket is not None:
                    self.pub_socket.close()
                    self.pub_socket = None

                try:
                    init_config = InitializationConfig.from_dict(init_config_dict)
                except TypeError as e:
                    raise ValueError(
                        f"Teleop Server: Invalid initialization config. "
                        f"It must be an instance of InitializationConfig.\n" f"{e}")
                print(f"TeleopServer: receive initialization config")

                with self._lock:
                    self._shared_last_init_config = deepcopy(init_config)
                    self._shared_initialized = False

                while not self._shared_initialized:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            return

    def wait_for_init_config(self):
        print("Teleop Server: Waiting for initialization config from teleop client.")
        while True:
            with self._lock:
                if self._shared_last_init_config is not None:
                    config = deepcopy(self._shared_last_init_config)
                    return config
            time.sleep(0.1)

    @property
    def initialized(self):
        with self._lock:
            return self._shared_initialized

    @property
    def last_init_config(self):
        with self._lock:
            return self._shared_last_init_config

    def set_initialized(self):
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(self.pub_bind_to)
        with self._lock:
            self._shared_initialized = True


def teleop_server_test():
    import time
    cmd_shape = 3
    server = TeleopServer(2345)
    config = server.wait_for_init_config()
    print(f"Pose: {config}")
    time.sleep(1)
    server.set_initialized()
    scale = 1
    for i in range(100):
        if server.initialized:
            time.sleep(0.1)
            print(i)
            server.send_teleop_cmd(np.ones(cmd_shape) * i * scale, np.ones(7))
        else:
            time.sleep(1)
            scale *= 10
            server.set_initialized()


if __name__ == '__main__':
    teleop_server_test()
