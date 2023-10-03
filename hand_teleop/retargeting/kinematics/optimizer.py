from abc import abstractmethod
from typing import List

import nlopt
import numpy as np
import sapien.core as sapien
import torch

from hand_teleop.retargeting.kinematics.optimizer_utils import SAPIENKinematicsModelStandalone


class Optimizer:
    def __init__(self, robot: sapien.Articulation, target_joint_names: List[str]):
        self.robot = robot
        self.robot_dof = robot.dof
        self.model = robot.create_pinocchio_model()

        joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        target_joint_index = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(f"Joint {target_joint_name} given does not appear to be in robot XML.")
            target_joint_index.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.target_joint_indices = np.array(target_joint_index)
        self.fixed_joint_indices = np.array([i for i in range(robot.dof) if i not in target_joint_index], dtype=int)
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(target_joint_index))
        self.dof = len(target_joint_index)

    def set_joint_limit(self, joint_limits: np.ndarray):
        if joint_limits.shape != (self.dof, 2):
            raise ValueError(f"Expect joint limits have shape: {(self.dof, 2)}, but get {joint_limits.shape}")
        self.opt.set_lower_bounds(joint_limits[:, 0].tolist())
        self.opt.set_upper_bounds(joint_limits[:, 1].tolist())

    def get_last_result(self):
        return self.opt.last_optimize_result()

    def get_link_names(self):
        return [link.get_name() for link in self.robot.get_links()]

    def get_link_indices(self, target_link_names):
        target_link_index = []
        for target_link_name in target_link_names:
            if target_link_name not in self.get_link_names():
                raise ValueError(f"Body {target_link_name} given does not appear to be in robot XML.")
            target_link_index.append(self.get_link_names().index(target_link_name))
        return target_link_index

    @abstractmethod
    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        pass

    def optimize(self, objective_fn, last_qpos):
        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
        except RuntimeError as e:
            print(e)
            return np.array(last_qpos)
        return qpos


class PositionOptimizer(Optimizer):
    def __init__(self, robot: sapien.Articulation, target_joint_names: List[str], target_link_names: List[str],
                 huber_delta=0.02, norm_delta=4e-3):
        super().__init__(robot, target_joint_names)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        self.target_link_indices = self.get_link_indices(target_link_names)

        # Use local jacobian if target link name <= 2, otherwise first cache all jacobian and then get all
        # This is only for the speed but will not affect the performance
        if len(target_link_names) <= 40:
            self.use_sparse_jacobian = True
        else:
            self.use_sparse_jacobian = False
        self.opt.set_ftol_abs(1e-5)

    def _get_objective_function(self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.robot_dof)
        qpos[self.fixed_joint_indices] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.target_joint_indices] = x
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [self.model.get_link_pose(index) for index in self.target_link_indices]
            body_pos = np.array([pose.p for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            # huber_distance = torch.norm(torch_body_pos - torch_target_pos, dim=1).mean()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    jacobians = []
                    for i, index in enumerate(self.target_link_indices):
                        link_spatial_jacobian = self.model.compute_single_link_local_jacobian(qpos, index)[:3,
                                                self.target_joint_indices]
                        link_rot = self.model.get_link_pose(index).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [self.model.get_link_jacobian(index, local=True)[:3, self.target_joint_indices] for
                                 index in self.target_link_indices]

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given")
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        if isinstance(last_qpos, np.ndarray):
            last_qpos = last_qpos.astype(np.float32)
        last_qpos = list(last_qpos)
        objective_fn = self._get_objective_function(ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32))
        return self.optimize(objective_fn, last_qpos)


class VectorOptimizer(Optimizer):
    def __init__(self, robot: sapien.Articulation, target_joint_names: List[str], origin_link_names: List[str],
                 task_link_names: List[str], huber_delta=0.02, norm_delta=4e-3, scaling=1.0):
        super().__init__(robot, target_joint_names)
        self.origin_link_names = origin_link_names
        self.task_link_names = task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(set(origin_link_names).union(set(task_link_names)))
        self.origin_link_indices = torch.tensor([self.computed_link_names.index(name) for name in origin_link_names])
        self.task_link_indices = torch.tensor([self.computed_link_names.index(name) for name in task_link_names])

        # Sanity check and cache link indices
        self.robot_link_indices = self.get_link_indices(self.computed_link_names)

        # Use local jacobian if target link name <= 2, otherwise first cache all jacobian and then get all
        # This is only for the speed but will not affect the performance
        if len(self.computed_link_names) <= 40:
            self.use_sparse_jacobian = True
        else:
            self.use_sparse_jacobian = False
        self.opt.set_ftol_abs(1e-6)

    def _get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.robot_dof)
        qpos[self.fixed_joint_indices] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.target_joint_indices] = x
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [self.model.get_link_pose(index) for index in self.robot_link_indices]
            body_pos = np.array([pose.p for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(robot_vec, torch_target_vec)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    jacobians = []
                    for i, index in enumerate(self.robot_link_indices):
                        link_spatial_jacobian = self.model.compute_single_link_local_jacobian(qpos, index)[:3,
                                                self.target_joint_indices]
                        link_rot = self.model.get_link_pose(index).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [self.model.get_link_jacobian(index, local=True)[:3, self.target_joint_indices] for
                                 index in self.robot_link_indices]

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given")
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        last_qpos = list(last_qpos)
        objective_fn = self._get_objective_function(ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32))
        return self.optimize(objective_fn, last_qpos)


class DexPilotAllegroIOptimizer(VectorOptimizer):
    MANO_INDEX = np.array([
        [8, 12, 16, 12, 16, 16, 0, 0, 0, 0],
        [4, 4, 4, 8, 8, 12, 4, 8, 12, 16]
    ])

    def __init__(self, robot: sapien.Articulation, target_joint_names: List[str],
                 # DexPilot parameters
                 gamma=2.5e-3,
                 project_dist=0.03,
                 escape_dist=0.05,
                 eta1=1e-4,
                 eta2=3e-2,
                 ):
        self.origin_link_names = [
            "index_link_3_tip", "middle_link_3_tip", "ring_link_3_tip",  # S1
            "middle_link_3_tip", "ring_link_3_tip", "ring_link_3_tip",  # S2
            "wrist", "wrist", "wrist", "wrist"  # root
        ]
        self.task_link_names = [
            "thumb_link_3_tip", "thumb_link_3_tip", "thumb_link_3_tip",
            "index_link_3_tip", "index_link_3_tip", "middle_link_3_tip",
            "thumb_link_3_tip", "index_link_3_tip", "middle_link_3_tip", "ring_link_3_tip"
        ]

        # Huber_delta and norm_delta has no effect, just assign a 0 value
        super().__init__(robot, target_joint_names, origin_link_names=self.origin_link_names,
                         task_link_names=self.task_link_names, huber_delta=0, norm_delta=0, scaling=1.6)

        # DexPilot parameters
        self.gamma = gamma
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # DexPilot cache
        self.projected = np.zeros(6, dtype=bool)
        self.s2_project_index_origin = np.array([1, 2, 2], dtype=int)
        self.s2_project_index_task = np.array([0, 0, 1], dtype=int)
        self.projected_dist = np.array([eta1] * 3 + [eta2] * 3)

    def _get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        target_vector = target_vector.astype(np.float32)
        qpos = np.zeros(self.robot_dof)
        qpos[self.fixed_joint_indices] = fixed_qpos

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:6], axis=1)
        self.projected[:3][target_vec_dist[0:3] < self.project_dist] = True
        self.projected[:3][target_vec_dist[0:3] > self.escape_dist] = False
        self.projected[3:6] = np.logical_and(
            self.projected[:3][self.s2_project_index_origin],
            self.projected[:3][self.s2_project_index_task]
        )

        # Update weight vector
        normal_weight = np.ones(6, dtype=np.float32)
        high_weight = np.array([200] * 3 + [400] * 3, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)
        weight = torch.from_numpy(np.concatenate([weight, np.ones(4, dtype=np.float32) * 10]))

        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (10, 3)
        dir_vec = target_vector[:6] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

        # Compute final reference vector
        reference_vec = np.where(self.projected[:, None], projected_vec, normal_vec[:6])  # (6, 3)
        reference_vec = np.concatenate([reference_vec, normal_vec[6:10]], axis=0)  # (10, 3)
        torch_ref_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_ref_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.target_joint_indices] = x
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [self.model.get_link_pose(index) for index in self.robot_link_indices]
            body_pos = np.array([pose.p for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            error = robot_vec - torch_ref_vec
            mse_loss = torch.sum(error * error, dim=1)  # (10)
            weighted_mse_loss = torch.sum(mse_loss * weight)
            result = weighted_mse_loss.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    jacobians = []
                    for i, index in enumerate(self.robot_link_indices):
                        link_spatial_jacobian = self.model.compute_single_link_local_jacobian(qpos, index)[:3,
                                                self.target_joint_indices]
                        link_rot = self.model.get_link_pose(index).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [self.model.get_link_jacobian(index, local=True)[:3, self.target_joint_indices] for
                                 index in self.robot_link_indices]

                weighted_mse_loss.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                #  Finally, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                #  (equivalent to fully opened the hand)
                grad_qpos += 2 * self.gamma * x

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given")
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        objective_fn = self._get_objective_function(ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32))
        return self.optimize(objective_fn, last_qpos)


def main_position_optimizer():
    import time
    from pathlib import Path
    np.set_printoptions(precision=4)
    np.random.seed(1)

    # SAPIEN Scene
    urdf_path = Path(__file__).parent.parent.parent / "assets/urdf/allegro_hand.urdf"
    sapien_model = SAPIENKinematicsModelStandalone(str(urdf_path))
    robot = sapien_model.robot
    print(robot.dof)

    # Optimizer
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    link_names = ["thumb_link_3", "index_link_3", "middle_link_3", "ring_link_3"]
    joint_limit = robot.get_qlimits()
    optimizer = PositionOptimizer(robot, joint_names, link_names)
    optimizer.set_joint_limit(joint_limit[:])

    tic = time.time()
    for i in range(1000):
        random_qpos = np.random.rand(robot.dof)
        random_qpos = random_qpos.clip(joint_limit[:, 0], joint_limit[:, 1])
        robot.set_qpos(random_qpos)
        random_target = np.array([robot.get_links()[i].get_pose().p for i in optimizer.target_link_indices])
        init_qpos = np.clip(random_qpos + np.random.randn(robot.dof) * 0.2, joint_limit[:, 0], joint_limit[:, 1])
        computed_qpos = optimizer.retarget(random_target, fixed_qpos=[], last_qpos=init_qpos[:])
        print(np.mean(np.abs(computed_qpos - random_qpos[:])))

        qpos = np.copy(random_qpos)
        qpos[:] = computed_qpos
        robot.set_qpos(qpos)
        computed_target = np.array([robot.get_links()[i].get_pose().p for i in optimizer.target_link_indices])
        # print("distance from target", np.mean(np.linalg.norm(computed_target - random_target, axis=1)))
        robot.set_qpos(init_qpos)
        init_target = np.array([robot.get_links()[i].get_pose().p for i in optimizer.target_link_indices])
        print("distance from init", np.mean(np.linalg.norm(computed_target - init_target, axis=1)))

    print(f"Kinematics Retargeting computation takes {time.time() - tic}s")


def main_vector_optimizer():
    import time
    from pathlib import Path
    np.set_printoptions(precision=4)
    np.random.seed(1)

    # SAPIEN Scene
    asset_path = Path(__file__).parent.parent.parent / "assets"
    urdf_path = asset_path / "urdf/kuka_allegro_description/allegro_hand_retargeting.urdf"
    sapien_model = SAPIENKinematicsModelStandalone(str(urdf_path), add_dummy_rotation=True)
    robot = sapien_model.robot
    print(robot.dof)

    # Optimizer
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    origin_link_names = ["palm_retargeting_link"] * 4
    task_link_names = ["thumb_link_3", "index_link_3", "middle_link_3", "ring_link_3"]
    joint_limit = robot.get_qlimits()
    optimizer = VectorOptimizer(robot, joint_names, origin_link_names, task_link_names)
    optimizer.set_joint_limit(joint_limit[:])

    tic = time.time()
    for i in range(100):
        random_qpos = np.random.rand(robot.dof)
        random_qpos = random_qpos.clip(joint_limit[:, 0], joint_limit[:, 1])
        robot.set_qpos(random_qpos)
        random_pos = np.array([robot.get_links()[i].get_pose().p for i in optimizer.robot_link_indices])
        origin_pos = random_pos[optimizer.origin_link_indices]
        task_pos = random_pos[optimizer.task_link_indices]
        random_target_vector = task_pos - origin_pos
        init_qpos = np.clip(random_qpos + np.random.randn(robot.dof) * 0.5, joint_limit[:, 0], joint_limit[:, 1])
        computed_qpos = optimizer.retarget(random_target_vector, fixed_qpos=[], last_qpos=init_qpos[:])
        print(np.mean(np.abs(computed_qpos - random_qpos[:])))

        qpos = np.copy(random_qpos)
        qpos[:] = computed_qpos
        robot.set_qpos(qpos)
        computed_pos = np.array([robot.get_links()[i].get_pose().p for i in optimizer.robot_link_indices])
        computed_origin_pos = computed_pos[optimizer.origin_link_indices]
        computed_task_pos = computed_pos[optimizer.task_link_indices]
        computed_target_vector = computed_task_pos - computed_origin_pos
        print("distance from target", np.mean(np.linalg.norm(computed_target_vector - random_target_vector, axis=1)))
        robot.set_qpos(init_qpos)

    print(f"Kinematics Retargeting computation takes {time.time() - tic}s for 100 times")


def main_dexpilot_optimizer():
    import time
    from pathlib import Path
    np.set_printoptions(precision=4)
    np.random.seed(1)

    # SAPIEN Scene
    urdf_path = Path(__file__).parent.parent.parent / "assets/urdf/allegro_hand.urdf"
    sapien_model = SAPIENKinematicsModelStandalone(str(urdf_path))
    robot = sapien_model.robot
    print(robot.dof)

    # Optimizer
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    joint_limit = robot.get_qlimits()
    optimizer = DexPilotAllegroIOptimizer(robot, joint_names)
    optimizer.set_joint_limit(joint_limit[:])

    tic = time.time()
    for i in range(100):
        random_qpos = np.random.rand(robot.dof)
        random_qpos = random_qpos.clip(joint_limit[:, 0], joint_limit[:, 1])
        robot.set_qpos(random_qpos)
        random_pos = np.array([robot.get_links()[i].get_pose().p for i in optimizer.robot_link_indices])
        origin_pos = random_pos[optimizer.origin_link_indices]
        task_pos = random_pos[optimizer.task_link_indices]
        random_target_vector = task_pos - origin_pos
        init_qpos = np.clip(random_qpos + np.random.randn(robot.dof) * 0.4, joint_limit[:, 0], joint_limit[:, 1])
        computed_qpos = optimizer.retarget(random_target_vector, fixed_qpos=[], last_qpos=init_qpos[:])
        print(np.mean(np.abs(computed_qpos - random_qpos[:])))

        qpos = np.copy(random_qpos)
        qpos[:] = computed_qpos
        robot.set_qpos(qpos)
        computed_pos = np.array([robot.get_links()[i].get_pose().p for i in optimizer.robot_link_indices])
        computed_origin_pos = computed_pos[optimizer.origin_link_indices]
        computed_task_pos = computed_pos[optimizer.task_link_indices]
        computed_target_vector = computed_task_pos - computed_origin_pos
        print("distance from target", np.mean(np.linalg.norm(computed_target_vector - random_target_vector, axis=1)))
        robot.set_qpos(init_qpos)

    print(f"Kinematics Retargeting computation takes {time.time() - tic}s")


if __name__ == '__main__':
    main_vector_optimizer()
