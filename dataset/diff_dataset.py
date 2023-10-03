import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import pickle
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


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


class BCDataset(Dataset):
    def __init__(self, data_paths=None, data=None):
        super().__init__()
        self.paths = None
        if data_paths != None:
            assert data == None
            self.paths = data_paths
            with open(self.paths[0], 'rb') as dummy_file:
                self.dummy_data = pickle.load(dummy_file)
        elif data != None:
            self.keys = data.keys()
            self.obs = torch.from_numpy(np.array(data['obs']))
            self.next_obs = torch.from_numpy(np.array(data['next_obs']))
            self.robot_qpos = torch.from_numpy(np.array(data['robot_qpos']))
            self.state = None
            self.next_state = None
            if 'state' in data.keys():
                self.state = torch.from_numpy(np.array(data['state']))
                self.next_state = torch.from_numpy(np.array(data['next_state']))
            self.action = torch.from_numpy(np.array(data['action']))
            self.dummy_data = {}
            for key in data.keys():
                self.dummy_data[key] = data[key][0]                
        else:
            raise NotImplementedError

    def __len__(self):
        if self.paths != None:
            return len(self.paths)
        else:
            return len(self.action)
    
    def __getitem__(self, idx):
        if self.paths != None:
            with open(self.paths[idx],'rb') as file:
                data = pickle.load(file)
            obs = torch.from_numpy(np.array(data['obs']))
            next_obs = torch.from_numpy(np.array(data['next_obs']))
            action = torch.from_numpy(np.array(data['action']))
            robot_qpos = torch.from_numpy(np.array(data['robot_qpos']))
            if 'state' in data.keys():
                state = torch.from_numpy(np.array(data['state']))
                next_state = torch.from_numpy(np.array(data['next_state']))
                return obs, next_obs, state, next_state, action
        else:
            obs = self.obs[idx]
            next_obs = self.next_obs[idx]
            state = None
            next_state = None
            action = self.action[idx]
            robot_qpos = self.robot_qpos[idx]
            if 'state' in self.keys:
                state = self.state[idx]
                next_state = self.next_state[idx]
                return obs, next_obs, state, next_state, action
        return obs, next_obs, robot_qpos, action



def prepare_data(dataset_folder, backbone_type, batch_size , val_ratio = 0.02, seed = 0):
    '''
    dataset_folder = Can be the folder where individual data are stored or the .pickle file of whole dataset
    (concatenated_obs = n x dim)
    obs = n x fs*3 x 224 x 224
    next_obs = n x fs*3 x 224 x 224
    state = n x fs*dim
    next_state = n x fs*dim
    action = n x dim
    '''
    # dataset_folder = './sim/baked_data/{}'.format(dataset_folder)
    print('=== Loading trajectories ===')
    with open('{}/{}_dataset.pickle'.format(dataset_folder, backbone_type.replace("/", "")),'rb') as file:
        data = pickle.load(file)
    random_order = np.random.permutation(len(data["action"]))
    np.random.seed(seed)
    if 'state' not in data.keys():
        obs = np.array(data['obs'])
        obs = obs[random_order]
        next_obs = np.array(data['next_obs'])
        next_obs = next_obs[random_order]
        robot_qpos = np.array(data['robot_qpos'])
        robot_qpos = robot_qpos[random_order]
        targets = np.array(data["action"])
        targets = targets[random_order]
        cutoff = int(len(obs) * val_ratio)
        train_data = dict(obs=obs[cutoff:], next_obs=next_obs[cutoff:], action=targets[cutoff:], robot_qpos=robot_qpos[cutoff:])
        validation_data = dict(obs=obs[:cutoff], next_obs=next_obs[:cutoff], action=targets[:cutoff], robot_qpos=robot_qpos[:cutoff])
    else:
        obs = np.array(data['obs'])
        obs = obs[random_order]
        next_obs = np.array(data['next_obs'])
        next_obs = next_obs[random_order]
        states = np.array(data["state"])
        states = states[random_order]
        next_states = np.array(data["next_state"])
        next_states = next_states[random_order]
        targets = np.array(data["action"])
        targets = targets[random_order]
        cutoff = int(len(obs) * val_ratio)
        #cutoff = int(len(obs) % batch_size)
        train_obs = obs[cutoff:]
        valid_obs = obs[:cutoff]
        train_next_obs = next_obs[cutoff:]
        valid_next_obs = next_obs[:cutoff]
        train_states = states[cutoff:]
        valid_states = states[:cutoff]
        train_next_states = next_states[cutoff:]
        valid_next_states = next_states[:cutoff]
        train_targets = targets[cutoff:]
        valid_targets = targets[:cutoff]
        train_data = dict(obs=train_obs, next_obs=train_next_obs, state=train_states, next_state=train_next_states, action=train_targets)
        validation_data = dict(obs=valid_obs, next_obs=valid_next_obs, state=valid_states, next_state=valid_next_states, action=valid_targets)
    bc_train_set = BCDataset(data_paths=None, data=train_data)
    bc_validation_set = BCDataset(data_paths=None, data=validation_data)
    bc_train_dataloader = DataLoader(bc_train_set, batch_size=batch_size, shuffle=True)
    bc_validation_dataloader = DataLoader(bc_validation_set, batch_size=len(bc_validation_set), shuffle=False)
    it_per_epoch = len(bc_train_set) // batch_size
    print('  ', 'total number of training samples', len(bc_train_set))
    print('  ', 'total number of validation samples', len(bc_validation_set))
    print('  ', 'number of iters per epoch', it_per_epoch)
    return it_per_epoch, bc_train_set, bc_train_dataloader, bc_validation_set, bc_validation_dataloader

def get_stacked_data_from_obs(rgb_imgs, stacked_robot_qpos, robot_qpos, robot_states, obs, i, concatenate=False, feature_extractor=None, preprocess=None):
    if concatenate:
        assert feature_extractor != None
        assert preprocess != None
        robot_state = obs["state"]
        img = obs["relocate_view-rgb"]
        img = torch.from_numpy(np.moveaxis(img,-1,0)[None, ...])
        img = preprocess(img)
        img = img.to(device)
        with torch.no_grad():
            feature = feature_extractor(img)
        feature = feature.cpu().detach().numpy().reshape(-1)
        ################rgb_imgs here is actually features################
        rgb_imgs.append(feature)
        robot_states.append(robot_state)
        stacked_robot_qpos.append(robot_qpos)
        if i == 0:
            obs = np.concatenate((rgb_imgs[i],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]))
            concatenate_robot_qpos = np.concatenate((stacked_robot_qpos[i],stacked_robot_qpos[i],stacked_robot_qpos[i],stacked_robot_qpos[i]))
        elif i == 1:
            obs = np.concatenate((rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]))
            concatenate_robot_qpos = np.concatenate((stacked_robot_qpos[i-1],stacked_robot_qpos[i],stacked_robot_qpos[i],stacked_robot_qpos[i]))
        elif i == 2:
            obs = np.concatenate((rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i]))
            concatenate_robot_qpos = np.concatenate((stacked_robot_qpos[i-2],stacked_robot_qpos[i-1],stacked_robot_qpos[i],stacked_robot_qpos[i]))
        else:
            obs = np.concatenate((rgb_imgs[i-3],rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i]))
            concatenate_robot_qpos = np.concatenate((stacked_robot_qpos[i-3],stacked_robot_qpos[i-2],stacked_robot_qpos[i-1],stacked_robot_qpos[i]))
        obs = torch.from_numpy(obs).to(device) 
        concatenate_robot_qpos = torch.from_numpy(concatenate_robot_qpos).to(device)
        return rgb_imgs, robot_states, stacked_robot_qpos, obs, concatenate_robot_qpos
    else:
        robot_states.append(obs["state"])
        rgb_imgs.append(obs["relocate_view-rgb"])

        # Save the frames - for debugging
        # rgb_pic = (obs["relocate_view-rgb"] * 255).astype(np.uint8)
        # imageio.imsave("./temp/eval/relocate-rgb_{}.png".format(i), rgb_pic)

        if i==0:
            stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0)
            stacked_states = np.concatenate((robot_states[i],robot_states[i],robot_states[i],robot_states[i]))        
        elif i==1:
            stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0)
            stacked_states = np.concatenate((robot_states[i-1],robot_states[i],robot_states[i],robot_states[i]))         
        elif i==2:
            stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0)
            stacked_states = np.concatenate((robot_states[i-2],robot_states[i-1],robot_states[i],robot_states[i]))       
        else:
            stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i-3],rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i]), axis=-1), -1, 0)       
            stacked_states = np.concatenate((robot_states[i-3],robot_states[i-2],robot_states[i-1],robot_states[i]))        

        stacked_imgs = torch.from_numpy(stacked_imgs).to(device)
        stacked_imgs = torch.unsqueeze(stacked_imgs, 0)
        stacked_states = torch.from_numpy(stacked_states).to(device)
        stacked_states = torch.unsqueeze(stacked_states, 0)        
        return rgb_imgs, robot_states, stacked_imgs, stacked_states

def argument_dependecy_checker(args):
    if args['resume']:
        assert args['load_model_from'] != None
    if args['use_visual_backbone']:
        assert args['backbone_type'] != None
    else:
        assert args['embedding_name'] != None
    if args['embedding_name'] == 'flexible':
        assert args['num_layers'] > 0
    if args['use_ss']:
        assert args['use_visual_encoder'] == True
        assert args['train_visual_encoder'] == True
        assert args['train_inv'] == True
        assert args['num_shared_layers'] > 0
    if args['use_state_encoder']:
        assert args['state_hidden_dim'] != None
        
    if args['use_visual_backbone']:
        args['model_name'] = '{}'.format(args['backbone_type'])
    elif args['use_visual_encoder']:
        args['model_name'] = '{}'.format(args['embedding_name'])
    else:
        args['model_name'] = 'policy_only'
    if args['use_state_encoder']:
        args['model_name'] = '{}_using_se'.format(args['model_name'])
    if args['use_ss']:
        args['model_name'] = '{}_with_pad'.format(args['model_name'])
    args['model_name'] = '{}_{}'.format(args['model_name'], args['dataset_folder'])

    return args
