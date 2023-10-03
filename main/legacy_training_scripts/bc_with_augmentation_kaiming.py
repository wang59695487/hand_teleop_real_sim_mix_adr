import imageio
import enum
import os
import pickle
import random
from itertools import chain
from multiprocessing import Pool
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hand_teleop.player.play_multiple_demonstrations import \
    play_one_visual_demo
from tqdm import tqdm

from main.behavior_cloning import BehaviorCloning
from main.feature_extractor import generate_feature_extraction_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self):
		super().__init__()
		self.pad = int(224/21)

	def forward(self, x):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


def temporal_stack_4_frames(features):
    """
    Take a series of temporal features T x ... x C and stack the last 4 frames
    together.

    features : T x ... x C
    """
    stacked_shape = features.shape + (4, )
    # structure of stacked_frames
    # stacked frame index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
    # -------------------------------------------------
    # 0 in stack          | 0 | 0 | 0 | 0 | 1 | 2 | 3 |
    # 1 in stack          | 0 | 1 | 1 | 1 | 2 | 3 | 4 |
    # 2 in stack          | 0 | 1 | 2 | 2 | 3 | 4 | 5 |
    # 3 in stack          | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
    stacked_frames = np.zeros(stacked_shape, dtype=np.float32)
    # put in stacked frames of t >= 3
    stacked_frames[3:, ..., 0] = features[:-3, ...]
    stacked_frames[3:, ..., 1] = features[1:-2, ...]
    stacked_frames[3:, ..., 2] = features[2:-1, ...]
    stacked_frames[3:, ..., 3] = features[3:, ...]
    # put in the first frame in the stack, for t <= 2
    stacked_frames[:3, ..., 0] = features[0]
    # put in the second to fourth frame in the stack, for t = 0
    stacked_frames[0, ..., 1:] = features[0][..., np.newaxis]
    # put in the second frame in the stack, for 1 <= t <= 2
    stacked_frames[1:3, ..., 1] = features[1]
    # put in the third and fourth frame in the stack, for t = 1
    stacked_frames[1, ..., 2:] = features[1][..., np.newaxis]
    # put in the third and fourth frame in the stack, for t = 2
    stacked_frames[2, ..., 2:] = features[2][..., np.newaxis]

    return stacked_frames


@torch.no_grad()
def single_stack_and_augment(demo, model, augmenter):
    rgb_imgs = [obs["relocate_view-rgb"] for obs in demo["obs"]]
    robot_states = [obs["state"] for obs in demo["obs"]]
    target_actions = demo["action"]
    rgb_imgs = np.stack(rgb_imgs, axis=0)
    n, h, w, c = rgb_imgs.shape

    stacked_frames = temporal_stack_4_frames(rgb_imgs)
    print(stacked_frames.shape)
    stacked_frames = stacked_frames.transpose((0, 1, 2, 4, 3))
    stacked_frames = stacked_frames.reshape((n, h, w, 4 * c))
    stacked_frames = stacked_frames.transpose((0, 3, 1, 2))

    stacked_frames = torch.from_numpy(stacked_frames).to(device)
    print(stacked_frames.shape)
    augmented_stacked_frames = augmenter(stacked_frames)
    print(augmented_stacked_frames.shape)
    assert len(rgb_imgs) == len(augmented_stacked_frames)

    augmented_stacked_frames = augmented_stacked_frames.reshape((n, 4, 3, h, w))
    augmented_stacked_frames = augmented_stacked_frames.reshape((4 * n, 3, h, w))
    print(len(rgb_imgs))
    print(augmented_stacked_frames.shape)

    features = model(augmented_stacked_frames).detach().cpu().numpy().squeeze(axis=(-1, -2))

    features = np.stack(features, axis=0)
    vis_c = features.shape[-1]
    features = features.reshape((-1, 4, vis_c))
    features = features.reshape((n, -1))
    robot_states = np.stack(robot_states, axis=0)
    stacked_robot_states = temporal_stack_4_frames(robot_states)
    stacked_robot_states = stacked_robot_states.transpose((0, 2, 1))
    stacked_robot_states = stacked_robot_states.reshape((n, -1))
    print(stacked_robot_states.shape)
    concatenated_obs = np.concatenate([features, stacked_robot_states],
        axis=-1)
    concatenated_obs = [concatenated_obs[i] for i in range(n)]
    print(np.array(concatenated_obs).shape)
    return concatenated_obs, target_actions


def stack_and_augment(visual_baked_demos, backbone_type):
    data_augmentor = RandomShiftsAug()
    visual_training_set = dict(obs=[], action=[])
    model = generate_feature_extraction_model(backbone_type=backbone_type)
    model.to(device)
    model.eval()
    data_augmentor.to(device)
    data_augmentor.eval()

    for visual_baked in visual_baked_demos:
        obs, act = single_stack_and_augment(demo=visual_baked, model=model, augmenter=data_augmentor)
        visual_training_set["obs"].extend(obs)
        visual_training_set["action"].extend(act)

    return visual_training_set


def train_augmented_bc(backbone_type, demo_folder):
    demo_folder = demo_folder
    model_path = './trained_models/latest_model_with_augmentation_{}_complete_random_new_long.pt'.format(backbone_type.lower())
    robot_name = 'mano'
    demo_files = []
    for file_name in os.listdir(demo_folder):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(demo_folder, file_name))
    visual_baked_demos = []
    for i, demo in enumerate(demo_files):
        print(demo)
        with open(demo, 'rb') as file:
            demo = pickle.load(file)
            visual_baked, meta_data = play_one_visual_demo(demo, robot_name)
            visual_baked_demos.append(visual_baked)

    if backbone_type == "ResNet34":
        bc_model = BehaviorCloning(2280,51)
    else:
        bc_model = BehaviorCloning(8424,51)
    epochs = 400
    learning_rate = 0.001
    params = list(bc_model.parameters())
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(params=params, lr=learning_rate, weight_decay=0.01)
    bc_model = bc_model.to(device).float()
    criterion = criterion.to(device)

    for t in tqdm(range(epochs)):
        visual_training_set = stack_and_augment(visual_baked_demos, backbone_type)
        random_order = np.random.permutation(len(visual_training_set["obs"]))
        # NOTE: add validation set, you may need to divide directly the demos for that rather than the training set.
        obs = np.array(visual_training_set["obs"])
        obs = obs[random_order]
        obs = torch.from_numpy(obs).float()
        obs = obs.to(device)
        targets = np.array(visual_training_set["action"])
        targets = targets[random_order]
        targets = torch.from_numpy(targets).float()
        targets = targets.to(device)

        losses = 0
        bc_model.train()
        optimizer.zero_grad()
        outputs = bc_model(obs)
        loss = criterion(outputs, targets)
        losses += loss.item()
        loss.backward()
        optimizer.step()
        print("Epoch: {}    Loss: {}".format(t, losses/len(visual_training_set)))

    model_dict = bc_model.state_dict()
    state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state_dict, model_path)
    print("Training Complete")
    print("Behavior Cloning Model is saved to: {}".format(model_path))

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_augmented_bc(backbone_type = "ResNet34", demo_folder='./sim/raw_small_random_data')
