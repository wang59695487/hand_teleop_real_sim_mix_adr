# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import pickle
import threading
import time
from typing import List

import numpy as np
import zmq
from tornado import ioloop
from zmq.eventloop import zmqstream

from hand_teleop.teleop.teleop_server import InitializationConfig


class TeleopClient:
    def __init__(self, port: int, cmd_dim, host="localhost"):
        # Create socket
        self.ctx = zmq.Context()
        if host == "localhost":
            sub_bind_to = f"tcp://localhost:{port}"
        else:
            sub_bind_to = f"tcp://{host}:{port}"
        self.init_bind_to = ':'.join(sub_bind_to.split(':')[:-1] + [str(int(sub_bind_to.split(':')[-1]) + 1)])
        self.sub_bind_to = sub_bind_to
        self.sub_socket = None

        self.cmd_dim = cmd_dim

        # Setup background IO loop
        self._loop = None
        self._started = threading.Event()
        self._stream = None
        self._thread = threading.Thread(target=self.run)
        self._thread.daemon = True
        self._thread.start()

        # Multi-thread variable
        self._lock = threading.Lock()
        self._shared_most_recent_teleop_cmd = np.zeros(cmd_dim)
        self._shared_most_recent_ee_pose = np.zeros(7)
        self._shared_server_started = False

    def send_init_config(self, *, init_ee_pose: np.ndarray, robot_base_pose: np.ndarray, init_qpos: np.ndarray,
                         joint_names: List[str]):
        init_socket = self.ctx.socket(zmq.REQ)
        init_socket.connect(self.init_bind_to)
        init_config = InitializationConfig(
            init_ee_pose=init_ee_pose.astype(np.float64),
            robot_base_pose=robot_base_pose,
            init_qpos=init_qpos.astype(np.float64),
            joint_names=joint_names,
        )
        init_socket.send_json(init_config.to_dict())
        with self._lock:
            self._shared_server_started = False
        init_socket.close()

    def update_teleop_cmd(self, message):
        cmd = pickle.loads(message[0])
        target_qpos = cmd["target_qpos"]
        ee_pose = cmd["ee_pose"]
        if not self.started:
            print(f"Teleop Client: Teleop Server start, begin teleoperation now.")
            with self._lock:
                self._shared_server_started = True
                self._shared_most_recent_teleop_cmd[:] = target_qpos
                self._shared_most_recent_ee_pose[:] = ee_pose
        else:
            if not isinstance(target_qpos, np.ndarray) or target_qpos.shape != (self.cmd_dim,):
                raise ValueError(
                    f"Teleop client: Invalid command: qpos dim: {target_qpos.shape}, cmd dim: {self.cmd_dim}")
            with self._lock:
                self._shared_most_recent_teleop_cmd[:] = target_qpos
                self._shared_most_recent_ee_pose[:] = ee_pose

    def get_teleop_cmd(self):
        with self._lock:
            return self._shared_most_recent_teleop_cmd

    def get_ee_pose(self):
        with self._lock:
            return self._shared_most_recent_ee_pose

    @property
    def started(self):
        with self._lock:
            return self._shared_server_started

    def wait_for_server_start(self):
        try:
            while not self.started:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print('Keyboard interrupt, shutting down.\n')

    def run(self):
        self._loop = ioloop.IOLoop()
        self._loop.initialize()
        self._loop.make_current()
        self.sub_socket = self.ctx.socket(zmq.SUB)
        self._stream = zmqstream.ZMQStream(self.sub_socket, io_loop=ioloop.IOLoop.current())

        # Wait for server start
        self.sub_socket.connect(self.sub_bind_to)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b'')

        self._stream.on_recv(self.update_teleop_cmd)
        self._started.set()
        self._loop.start()


def teleop_client_test():
    cmd_shape = 3
    client = TeleopClient(2345, cmd_dim=cmd_shape)
    client.send_init_config(
        init_ee_pose=np.array([0.5, 0, 0.3, 1, 0, 0, 0]),
        init_qpos=np.zeros(cmd_shape),
        joint_names=[f"{i}_joint" for i in range(cmd_shape)],
    )

    client.wait_for_server_start()
    for _ in range(10):
        time.sleep(0.2)
        print(client.get_teleop_cmd())

    client.send_init_config(
        init_ee_pose=np.array([0.5, 0, 0.3, 1, 0, 0, 0]),
        init_qpos=np.zeros(cmd_shape),
        joint_names=[f"{i}_joint" for i in range(cmd_shape)],
    )
    client.wait_for_server_start()

    for _ in range(10):
        time.sleep(0.2)
        print(client.get_teleop_cmd())


if __name__ == '__main__':
    teleop_client_test()
