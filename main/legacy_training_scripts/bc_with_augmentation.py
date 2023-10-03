import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hand_teleop.player.play_multiple_demonstrations import play_one_visual_demo
from main.feature_extractor import generate_feature_extraction_model
from main.behavior_cloning import BehaviorCloning

import pickle
import os
import numpy as np
from tqdm import tqdm
import imageio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def stack_and_augment(visual_baked_demos, backbone_type):
    data_augmentor = RandomShiftsAug()
    data_augmentor = data_augmentor.to(device)
    visual_training_set = dict(obs=[], action=[])
    model = generate_feature_extraction_model(backbone_type=backbone_type)
    model = model.to(device)
    model.eval()
    for visual_baked in visual_baked_demos:
        rgb_imgs = []
        robot_states = []
        target_actions = []
        for act, obs in zip(visual_baked["action"], visual_baked["obs"]):
            rgb_imgs.append(obs["relocate_view-rgb"])
            robot_states.append(obs["state"])
            target_actions.append(act)
        stacked_frames = []
        for i in range(len(rgb_imgs)):
            if i==0:
                stacked_frames.append(np.concatenate((rgb_imgs[i],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1))
            elif i==1:
                stacked_frames.append(np.concatenate((rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1))
            elif i==2:
                stacked_frames.append(np.concatenate((rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i]), axis=-1))
            else:
                stacked_frames.append(np.concatenate((rgb_imgs[i-3],rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i]), axis=-1))        
        stacked_frames = np.asarray(stacked_frames)
        stacked_frames = np.moveaxis(stacked_frames, 3, 1)
        stacked_frames = torch.from_numpy(stacked_frames).to(device)
        augmented_stacked_frames = data_augmentor(stacked_frames)
        augmented_stacked_frames = augmented_stacked_frames.cpu().detach().numpy()
        assert len(rgb_imgs) == len(augmented_stacked_frames)

        augmented_expanded_frames = []
        for img in augmented_stacked_frames:
            augmented_expanded_frames.append(img[:3,:,:])
            augmented_expanded_frames.append(img[3:6,:,:])
            augmented_expanded_frames.append(img[6:9,:,:])
            augmented_expanded_frames.append(img[9:,:,:])
        augmented_expanded_frames = np.array(augmented_expanded_frames)

        features = []
        for img in augmented_expanded_frames:
            img = torch.from_numpy(img)
            img = img.reshape(1, 3, 224, 224)
            img = img.to(device)
            with torch.no_grad():
                feature = model(img)
            features.append(feature.cpu().detach().numpy().reshape(-1))
        concatenated_obs = []
        for i,j in zip(range(3,len(features),4), range(len(robot_states))):
            if j==0:
                concatenated_obs.append(np.concatenate((features[i-3],features[i-2],features[i-1],features[i],robot_states[j],robot_states[j],robot_states[j],robot_states[j])))
            elif j==1:
                concatenated_obs.append(np.concatenate((features[i-3],features[i-2],features[i-1],features[i],robot_states[j-1],robot_states[j],robot_states[j],robot_states[j])))
            elif j==2:
                concatenated_obs.append(np.concatenate((features[i-3],features[i-2],features[i-1],features[i],robot_states[j-2],robot_states[j-1],robot_states[j],robot_states[j])))
            else:
                concatenated_obs.append(np.concatenate((features[i-3],features[i-2],features[i-1],features[i],robot_states[j-3],robot_states[j-2],robot_states[j-1],robot_states[j])))        
        visual_training_set['obs'].extend(concatenated_obs)
        visual_training_set['action'].extend(target_actions)
        
    return visual_training_set


def train_augmented_bc(backbone_type, demo_folder):
    demo_folder = demo_folder
    model_path = './trained_models/latest_model_with_augmentation_{}_small_random.pt'.format(backbone_type.lower())
    robot_name = 'mano'
    demo_files = []
    for file_name in os.listdir(demo_folder):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(demo_folder, file_name))
    visual_baked_demos = []
    for demo in demo_files:
        print(demo)
        with open(demo, 'rb') as file:
            demo = pickle.load(file)
            visual_baked, meta_data = play_one_visual_demo(demo, robot_name)
            visual_baked_demos.append(visual_baked)

    if backbone_type == "ResNet34" or backbone_type == 'MoCo18':
        bc_model = BehaviorCloning(2280,51)
    else:
        bc_model = BehaviorCloning(8424,51)
    epochs = 1000
    learning_rate = 0.001
    params = list(bc_model.parameters())
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(params=params, lr=learning_rate, weight_decay=0.01)
    bc_model = bc_model.to(device).float()
    criterion = criterion.to(device)

    for t in tqdm(range(epochs)):
        visual_training_set = stack_and_augment(visual_baked_demos, backbone_type)
        random_order = np.random.permutation(len(visual_training_set["obs"]))
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
    train_augmented_bc(backbone_type = "MoCo50", demo_folder='./sim/raw_small_random_data')



