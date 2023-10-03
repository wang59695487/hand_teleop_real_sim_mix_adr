import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import h5py
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

class EpisodicDataset(Dataset):
    def __init__(self, episode_ids, data_path=None, aug_before=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.path = data_path
        self.aug_before = aug_before
        data_path = self.path[0] if self.aug_before is not None else self.path
        with h5py.File(data_path, 'r') as root:
            self.demo_data = root[f"episode_0"]
            self.chunk_size = len(self.demo_data['obs'])
            self.dummy_data = {'obs': self.demo_data['obs'][0],'robot_qpos': self.demo_data['robot_qpos'][0], 'action': self.demo_data['action'][0], 'sim_real_label': self.demo_data['sim_real_label'][0]}
    
    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode for now
        episode_id = self.episode_ids[index]
        if self.aug_before is not None:
            if episode_id < self.aug_before:
                data_path = self.path[0]  
            else:
                episode_id = episode_id - self.aug_before
                data_path = self.path[1]
        else:
            data_path = self.path
        with h5py.File(data_path, 'r') as root:
            self.episode_data  = root[f"episode_{episode_id}"]  
            self.obs = self.episode_data['obs']
            self.robot_qpos = self.episode_data['robot_qpos']
            self.action = self.episode_data['action']
            self.label = self.episode_data['sim_real_label']

            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(self.chunk_size)

            # get observation at start_ts only
            obs = self.obs[start_ts]
            robot_qpos = self.robot_qpos[start_ts]
            label = self.label[start_ts]
            
            # get all actions after and including start_ts
            action_len = self.chunk_size - start_ts
            action = self.action[start_ts:]
            padded_action =np.zeros((self.chunk_size,len(action[0])), dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(self.chunk_size)
            is_pad[action_len:] = 1

        obs = torch.from_numpy(obs).float()
        robot_qpos = torch.from_numpy(robot_qpos).float()
        padded_action = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

            
        return obs, robot_qpos, padded_action, label, is_pad


def prepare_real_sim_data(real_dataset_folder=None,sim_dataset_folder=None, backbone_type=None, real_batch_size=None, sim_batch_size=None, val_ratio = 0.1, seed = 0, chunk_size = 50):
    real_demo_length = 0
    sim_demo_length = 0

    if real_dataset_folder != None:
        print('=== Loading real trajectories ===')
        real_demo_file = os.path.join(real_dataset_folder, "dataset.h5")
        with open('{}/meta_data.pickle'.format(real_dataset_folder),'rb') as file:
            real_meta_data = pickle.load(file)
            real_img_data_aug = real_meta_data['num_img_aug']
            real_demo_length = real_meta_data['total_episodes']
        
    if sim_dataset_folder != None:
        print('=== Loading Sim trajectories ===')
        sim_demo_file = os.path.join(sim_dataset_folder, "dataset.h5")
        with open('{}/meta_data.pickle'.format(sim_dataset_folder),'rb') as file:
            sim_meta_data = pickle.load(file)
            sim_img_data_aug = sim_meta_data['num_img_aug']
            sim_demo_length = sim_meta_data['total_episodes']

    if sim_demo_length > 0 and real_demo_length > 0:

        sim_real_ratio = sim_demo_length/real_demo_length
        data_type = "real_sim"

        print("=== preparing real data: ===")
        it_per_epoch_real, bc_train_set_real, bc_train_dataloader_real, bc_validation_dataloader_real = prepare_data(real_demo_file, real_demo_length, real_batch_size, val_ratio, seed, real_img_data_aug, chunk_size)
        print("=== preparing sim data: ===")
        it_per_epoch_sim, bc_train_set_sim, bc_train_dataloader_sim, bc_validation_dataloader_sim = prepare_data(sim_demo_file, sim_demo_length, sim_batch_size, val_ratio, seed, sim_img_data_aug, chunk_size)
        Prepared_Data = {"it_per_epoch_real": it_per_epoch_real, "bc_train_set_real": bc_train_set_real, 
                     "bc_train_dataloader_real": bc_train_dataloader_real, "bc_validation_dataloader_real": bc_validation_dataloader_real,
                     "it_per_epoch_sim": it_per_epoch_sim, "bc_train_set_sim": bc_train_set_sim, "bc_train_dataloader_sim": bc_train_dataloader_sim, 
                     "bc_validation_dataloader_sim": bc_validation_dataloader_sim, "data_type": data_type, "sim_real_ratio": sim_real_ratio}
    else:
        if sim_demo_length != 0:
            print("=== preparing only sim data: ===")
            print("=== demo_length: ===")
            print(sim_demo_length)
            data_type = "sim"
            it_per_epoch, bc_train_set, bc_train_dataloader, bc_validation_dataloader = prepare_data(sim_demo_file, sim_demo_length, sim_batch_size, val_ratio, seed, sim_img_data_aug, chunk_size)
            
        elif real_demo_length != 0:
            print("=== preparing only real data: ===")
            print("=== demo_length: ===")
            print(real_demo_length)
            data_type = "real"
            it_per_epoch, bc_train_set, bc_train_dataloader, bc_validation_dataloader = prepare_data(real_demo_file, real_demo_length, real_batch_size, val_ratio, seed, real_img_data_aug, chunk_size)
            
        Prepared_Data = {"it_per_epoch": it_per_epoch, "bc_train_set": bc_train_set, "bc_train_dataloader": bc_train_dataloader, 
                     "bc_validation_dataloader": bc_validation_dataloader, "data_type": data_type}

    return Prepared_Data
    
def prepare_sim_aug_data(sim_dataset_folder=None,sim_dataset_aug_folder=None,sim_aug_demo_length=None, sim_batch_size=None, val_ratio = 0.1, seed = 0, chunk_size = 50):
    data_type = "sim"
    print('=== Loading Sim trajectories ===')
    sim_demo_file = os.path.join(sim_dataset_folder, "dataset.h5")
    sim_img_data_aug = 1
    with open('{}/meta_data.pickle'.format(sim_dataset_folder),'rb') as file:
        sim_meta_data = pickle.load(file)
        sim_demo_length = sim_meta_data['total_episodes'] 
    
    if sim_dataset_aug_folder != None:
        sim_aug_demo_file = os.path.join(sim_dataset_aug_folder, "dataset.h5")
        sim_demo_file = [sim_demo_file, sim_aug_demo_file]
        sim_demo_length = [sim_demo_length,sim_aug_demo_length]
    
    it_per_epoch, bc_train_set, bc_train_dataloader, bc_validation_dataloader = prepare_data(sim_demo_file, sim_demo_length, sim_batch_size, val_ratio, seed, sim_img_data_aug, chunk_size)
            
    Prepared_Data = {"it_per_epoch": it_per_epoch, "bc_train_set": bc_train_set, "bc_train_dataloader": bc_train_dataloader, 
                    "bc_validation_dataloader": bc_validation_dataloader, "data_type": data_type}

    return Prepared_Data

def prepare_data(data_path, total_episodes, batch_size, val_ratio = 0.1, seed = 0, img_data_aug = 1, action_length = 50):
    
    set_seed(seed)
    
    if len(data_path) == 2:
        aug_before = total_episodes[0]
        total_episodes = total_episodes[0] + total_episodes[1]
    else:
        aug_before = None
        
    random_demo_id = np.random.permutation(total_episodes)
    train_demo_idx = random_demo_id[:int(len(random_demo_id)*(1-val_ratio))]
    validation_demo_idx = random_demo_id[int(len(random_demo_id)*(1-val_ratio)):]
        
    bc_train_set = EpisodicDataset(episode_ids=train_demo_idx, data_path=data_path,aug_before=aug_before)
    bc_validation_set = EpisodicDataset(episode_ids=validation_demo_idx, data_path=data_path,aug_before=aug_before)
   
    bc_train_dataloader = DataLoader(bc_train_set, batch_size=batch_size, shuffle=True)
    val_batch_size = batch_size if batch_size < len(bc_validation_set) else len(bc_validation_set)
    bc_validation_dataloader = DataLoader(bc_validation_set, batch_size=val_batch_size, shuffle=False)
    
    it_per_epoch = len(bc_train_set) // batch_size
    print('  ', 'total number of training samples', len(bc_train_set))
    print('  ', 'total number of validation samples', len(bc_validation_set))
    print('  ', 'number of iters per epoch', it_per_epoch)
    return it_per_epoch, bc_train_set, bc_train_dataloader, bc_validation_dataloader

def argument_dependecy_checker(args):
    
    if args['sim_dataset_folder'] != None:
        args['model_name'] = '{}_sim_{}'.format(args['model_name'], args['sim_dataset_folder'])
    else:
        args['model_name'] = '{}_mix_{}'.format(args['model_name'], args['sim_dataset_folder'])

    return args
