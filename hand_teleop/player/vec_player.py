# Modified from ManiSkill2: https://github.com/haosulab/ManiSkill2/blob/main/mani_skill2/vector/vec_env.py
import multiprocessing as mp
from functools import partial
from multiprocessing.connection import Connection
from typing import Callable, List, Optional

import numpy as np
import sapien.core as sapien
import torch
import tqdm

from hand_teleop.render.render_player import RenderPlayer


def find_available_port():
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
        server_address = f"localhost:{port}"
    return server_address


def _worker(
        rank: int,
        remote: Connection,
        player_fn: Callable[..., RenderPlayer],
):
    player = None
    try:
        player = player_fn()
        while True:
            cmd, data = remote.recv()
            if cmd == "set_player_data":
                player.meta_data = data["meta_data"]
                player.data = data["data"]
                remote.send(len(data["data"]))
            elif cmd == "set_sim_data":
                player.set_sim_data(data)
                remote.send(None)
            elif cmd == "take_picture":
                cameras = [x for x in player.cameras.values()]
                player.scene._update_render_and_take_pictures(cameras)
                remote.send(None)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "handshake":
                remote.send(None)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
    except KeyboardInterrupt:
        print("Worker KeyboardInterrupt")
    except EOFError:
        print("Worker EOF")
    except Exception as err:
        print(err)
    finally:
        if player is not None:
            player.scene = None
            player.scene = None


class VecPlayer:
    device: torch.device
    remotes: List[Connection] = []
    work_remotes: List[Connection] = []
    processes: List[mp.Process] = []

    def __init__(
            self,
            player_fns: List[Callable[[], RenderPlayer]],
            start_method: Optional[str] = None,
            server_address: str = "auto",
            server_kwargs: dict = None,
            texture_names=("Color",),
            seed=None,
    ):
        self.waiting = False
        self.closed = False

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        n_players = len(player_fns)
        self.num_players = n_players

        # Start RenderServer
        if server_address == "auto":
            server_address = find_available_port()
        self.server_address = server_address
        server_kwargs = {} if server_kwargs is None else server_kwargs
        self.server = sapien.RenderServer(**server_kwargs)
        self.server.start(self.server_address)
        print(f"RenderServer is running at: {server_address}")

        # Wrap player_fn
        for i, player_fn in enumerate(player_fns):
            client_kwargs = {"address": self.server_address, "process_index": i, "renderer": "client"}
            player_fns[i] = partial(
                player_fn, **client_kwargs
            )

        # Initialize workers
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_players)])
        self.processes = []
        for rank in tqdm.trange(n_players):
            work_remote = self.work_remotes[rank]
            player_fn = player_fns[rank]
            args = (rank, work_remote, player_fn)
            process: mp.Process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        # To make sure environments are initialized in all workers
        for remote in self.remotes:
            remote.send(("handshake", None))
        for remote in self.remotes:
            remote.recv()

        # Infer texture names
        self.texture_names = texture_names

        # Allocate torch buffers
        # A list of [n_players, n_cams, H, W, C] tensors
        self._obs_torch_buffer: List[torch.Tensor] = self.server.auto_allocate_torch_tensors(
            list(set(self.texture_names)))
        self.device = self._obs_torch_buffer[0].device

        self.data_lengths = np.zeros(self.num_players)
        self._seed = 0 if seed is None else seed
        self._random_state = np.random.RandomState(self._seed)

    def set_sim_data_async(self, idx_list: np.ndarray):
        for remote, idx in zip(self.remotes, idx_list):
            remote.send(("set_sim_data", idx))

    def set_sim_data_random_async(self):
        indices = self._random_state.randint(0, self.data_lengths)
        self.set_sim_data_async(indices)
        return indices

    def set_sim_data_wait(self):
        results = [remote.recv() for remote in self.remotes]
        return

    def render_async(self):
        for remote in self.remotes:
            remote.send(("take_picture", None))

    def render_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.server.wait_all()
        tensor_dict = {}
        for i, name in enumerate(self.texture_names):
            tensor_dict[name] = self._obs_torch_buffer[i]

        return tensor_dict

    def load_player_data(self, demos):
        if len(demos) != self.num_players:
            raise ValueError(f"The number of demo should match the number of players.")
        for i, remote in enumerate(self.remotes):
            remote.send(("set_player_data", demos[i]))
        num_frames = [remote.recv() for remote in self.remotes]
        self.data_lengths[:] = num_frames[:]
