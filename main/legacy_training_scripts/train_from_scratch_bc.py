import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from hand_teleop.player.play_multiple_demonstrations import play_one_visual_demo
from main.feature_extractor import generate_feature_extraction_model
from main.behavior_cloning import BehaviorCloning
from main.bc_with_augmentation import RandomShiftsAug

import pickle
import numpy as np
from tqdm import tqdm
import imageio
import time
import os
import torchvision.transforms as T
from termcolor import colored
import itertools
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

class BCDataset(Dataset):
    def __init__(self, data, args):
        super().__init__()
        self.args = args            
        self.obs = []
        self.states = []
        self.actions = []
        self.obs = torch.from_numpy(np.array(data['obs']))
        self.states = torch.from_numpy(np.array(data['state']))
        self.actions = torch.from_numpy(np.array(data['action']))
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, index):
        obs = self.obs[index]
        state = self.states[index]
        action = self.actions[index]
        return obs, state, action


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class EmbeddingNet(nn.Module):
    """
    Input shape must be (N, H, W, 3), where N is the number of frames.
    The class will then take care of transforming and normalizing frames.
    The output shape will be (N, O), where O is the embedding size.
    """
    def __init__(self, embedding_name, in_channels=3, pretrained=True, train=False, disable_cuda=False):
        super(EmbeddingNet, self).__init__()

#        assert not train, 'Training the embedding is not supported.'

        self.embedding_name = embedding_name

        if self.embedding_name == 'true_state':
            return

        self.in_channels = in_channels
        self.embedding, self.transforms = \
            _get_embedding(embedding_name, in_channels, pretrained, train)
        dummy_in = torch.zeros(1, in_channels, 224, 224)
        dummy_in = self.transforms(dummy_in)
        self.in_shape = dummy_in.shape[1:]
        dummy_out = self._forward(dummy_in)
        self.out_size = np.prod(dummy_out.shape)
        print(colored(f"Embedding dim: {self.out_size}", 'green'))

        # Always use CUDA, it is much faster for these models
        # Disable it only for debugging
        if torch.cuda.is_available() and not disable_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.embedding = self.embedding.to(device=self.device)
        self.training = self.embedding.training

    def _forward(self, observation):
        out = self.embedding(observation)
        return out

    def forward(self, observation):
        if self.embedding_name == 'true_state':
            return observation.squeeze().cpu().numpy()
        # observation.shape -> (N, H, W, 3)
        observation = observation.to(device)
        observation = observation.transpose(1, 2).transpose(1, 3).contiguous()
        observation = self.transforms(observation)

        if self.embedding.training:
            out = self._forward(observation)
            return out.view(-1, self.out_size).squeeze()
        else:
            with torch.no_grad():
                out = self._forward(observation)
                return out.view(-1, self.out_size).squeeze().cpu().numpy()
            


def _get_embedding(embedding_name='random', in_channels=3, pretrained=True, train=False):
    """
    See https://pytorch.org/vision/stable/models.html

    Args:
        embedding_name (str, 'random'): the name of the convolution model,
        in_channels (int, 3): number of channels of the input image,
        pretrained (bool, True): if True, the model's weights will be downloaded
            from torchvision (if possible),
        train (bool, False): if True the model will be trained during learning,
            if False its parameters will not change.

    """

    # Default transforms: https://pytorch.org/vision/stable/models.html
    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    # where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then
    # normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    transforms = nn.Sequential(
        T.Resize(256, interpolation=3) if 'mae' in embedding_name else T.Resize(256),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )

    assert in_channels == 3, 'Current models accept 3-channel inputs only.'

    # FIXED 5-LAYER CONV
    if embedding_name == 'random':
        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        # original model
        model = nn.Sequential(
            init_(nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
        )
    elif embedding_name == 'ours':

        # model = nn.Sequential(
        #     init_(nn.Conv2d(in_channels, 32, kernel_size=(5,5), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        # )
        model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1),
            # nn.ReLU(),
        )

        # # DrQ like model
        # model = nn.Sequential(
        #     init_(nn.Conv2d(in_channels, 32, kernel_size=(7,7), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(5,5), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        # )
        # remove imgnet normalization and resize to 84x84
        transforms = nn.Sequential(
            T.Resize(84),
        )

    if train:
        model.train()
        for p in model.parameters():
            p.requires_grad = True
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model, transforms

def train(args):
    # Set seeds
    # set_seed(args.seed)
    model_save_path = 'trained_models/{}.pt'.format(args['model_name'])
    
    # Init models, env, optimizer, ...    
    embedding_model = EmbeddingNet(args['embedding_name'],
                                   in_channels=3,
                                   pretrained=True,
                                   train=args['train_embedding'])
    embedding_model = embedding_model.to(device)
    actor_model = BehaviorCloning(3432, 51)
    actor_model = actor_model.to(device)
    writer = SummaryWriter(log_dir='runs/{}'.format(args['model_name']))

    if args['train_embedding']:
        embedding_model.train()
    else:
        embedding_model.eval()
    optimizer = torch.optim.Adam(
        itertools.chain(embedding_model.parameters(), actor_model.parameters()) if args['train_embedding'] else actor_model.parameters(),
        lr=args['learning_rate'])
    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    # Resume old run
    if args['resume']:
        checkpoint = torch.load(model_save_path)
        embedding_model.load_state_dict(checkpoint["embedding_model_state_dict"])
        actor_model.load_state_dict(checkpoint["actor_model_state_dict"])
        optimizer.load_state_dict(checkpoint["actor_model_optimizer_state_dict"])

    print('=== BC run ===')
    print('  ', 'embedding:', args['embedding_name'])
    print('  ', 'train_embedding:', args['train_embedding'])
    print('  ', 'use augmentation:', args['use_augmentation'])
    print('  ', 'lr:', args['learning_rate'])

    # Read and prepare data
    # If multiple environments are used, we read one environment at the time
    # and immediately pass frames through the embedding, in order to save space
    # (frames are not kept in memory).
    print('=== Loading trajectories ===')
    data_path = './sim/baked_data/{}.pickle'.format(args['dataset_name'])
    with open(data_path,'rb') as file:
        data = pickle.load(file)
    random_order = np.random.permutation(len(data["obs"]))
    obs = np.array(data["obs"])
    obs = obs[random_order]
    states = np.array(data["state"])
    states = states[random_order]
    targets = np.array(data["action"])
    targets = targets[random_order]
    cutoff = int(len(obs) % args['batch_size'])
    train_obs = obs[cutoff:]
    valid_obs = obs[:cutoff]
    train_states = states[cutoff:]
    valid_states = states[:cutoff]
    train_targets = targets[cutoff:]
    valid_targets = targets[:cutoff]
    train_data = dict(obs=train_obs, state=train_states, action=train_targets)
    valid_obs = torch.from_numpy(np.array(valid_obs)).to(device)
    valid_states = torch.from_numpy(np.array(valid_states)).to(device)
    valid_targets = torch.from_numpy(np.array(valid_targets)).to(device)
    bc_dataset = BCDataset(train_data, args)
    bc_dataloader = DataLoader(bc_dataset, batch_size=args['batch_size'], shuffle=True)
    it_per_epoch = len(bc_dataset) // args['batch_size']
    print('  ', 'total number of training samples', len(bc_dataset))
    print('  ', 'total number of validation samples', len(valid_obs))
    print('  ', 'number of iters per epoch', it_per_epoch)


    print('=== Training policy ===')
    # prepare some stuff
    it = 0
    start_time = time.time()
    aug = RandomShiftsAug()
    aug = aug.to(device)
    
    success_epoch_record = []
    return_epoch_record = []

    # begin training
    actor_model.train()
    if args['train_embedding']:
        embedding_model.train()
    
    # wall time counter
    time_train_iter_list = []

    # training loop
    for epoch in range(args['num_epochs']):
        loss_epoch = 0
        print('  ','Epoch: ', epoch)
        for _ in tqdm(range(it_per_epoch)):
            optimizer.zero_grad()
            it += 1
            time_train_iter = time.time()
            obs_batch, state_batch, action_batch = next(iter(bc_dataloader))
            obs_batch = obs_batch.to(device)
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            
            if args['train_embedding']: # use image and train embedding
                embedding_batch = []
                for frame_id in range(args['frame_stack']):
                    if args['use_augmentation']:
                        obs_batch[:, frame_id*3:(frame_id+1)*3] = aug(obs_batch[:, frame_id*3:(frame_id+1)*3])
                    embedding_i = embedding_model(obs_batch[:, frame_id*3:(frame_id+1)*3].permute(0, 2, 3, 1)) # obs: bxfsx256x256 -> bx256x256xfs
                    embedding_batch.append(embedding_i)
                embedding_batch = torch.stack(embedding_batch, dim=1)
                embedding_batch = embedding_batch.view(args['batch_size'], -1) # concat frames

            # TODO: WHAT IS THIS PART????
            else: # use image and do not train embedding. the obs is already the embedding
                embedding_batch = obs_batch.cuda()
                embedding_batch = [ obs_batch[:, frame_id] for frame_id in range(args['frame_stack']) ]
                embedding_batch = embedding_batch.view(args['batch_size'], -1) # concat embedding frames
            
            final_obs_batch = torch.cat((embedding_batch,state_batch), dim=1)
            # Prediction with agent state at step t
            actor_output = actor_model(final_obs_batch)

            # Loss
            loss_batch = criterion(actor_output, action_batch)
            loss_epoch += loss_batch.item()
            loss_batch.backward()
            optimizer.step()
            
            # # log walltime
            # time_train_iter_list.append(time.time() - time_train_iter)

            # if it % args.log_interval == 0:
            #     # compute grad norm
            #     embed_grad_norm = 0
            #     actor_grad_norm = 0
            #     for p in list(filter(lambda p: p.grad is not None, embedding_model.parameters())):
            #         embed_grad_norm += p.grad.data.norm(2).item()
            #     for p in list(filter(lambda p: p.grad is not None, actor_model.parameters())):
            #         actor_grad_norm += p.grad.data.norm(2).item()

            #     train_metrics = { 'bc_loss': loss_batch.item(), \
            #             'embed_grad_norm': embed_grad_norm, 'actor_grad_norm': actor_grad_norm,\
            #             'iterations': it ,  'total_time': time.time() - start_time,}
            #     L.log(train_metrics, category='train')
            
            # if it % 100 == 0:
            #     print('train iter (avg over 100):', np.mean(time_train_iter_list))
            #     time_train_iter_list = []

        writer.add_scalar('Loss/train', loss_epoch/len(bc_dataset), epoch)

        # Validation
        if epoch%10==0 or epoch==args['num_epochs']-1:
            embedding_model.eval()
            actor_model.eval()
            with torch.no_grad():
                valid_embedding = []
                for frame_id in range(args['frame_stack']):
                    embedding_i = embedding_model(valid_obs[:, frame_id*3:(frame_id+1)*3].permute(0, 2, 3, 1)) # obs: bxfsx256x256 -> bx256x256xfs
                    valid_embedding.append(torch.from_numpy(embedding_i).to(device))
                valid_embedding = torch.stack(valid_embedding, dim=1)
                valid_embedding = valid_embedding.view(len(valid_obs), -1) # concat frames
                final_valid_obs = torch.cat((valid_embedding,valid_states), dim=1)

                valid_actor_output = actor_model(final_valid_obs)
                valid_loss = criterion(valid_actor_output, valid_targets)
                writer.add_scalar('Loss/test', valid_loss.item()/len(valid_targets), epoch)
            actor_model.train()
            if args['train_embedding']:
                    embedding_model.train()

        print('  ', 'Loss: {}'.format(loss_epoch/len(bc_dataset)))
        if args['save_model']:
            torch.save({
                'embedding_model_state_dict': embedding_model.state_dict(),
                'actor_model_state_dict': actor_model.state_dict(),
                'actor_model_optimizer_state_dict': optimizer.state_dict(),
                'args': args,
            }, model_save_path)    


if __name__ == '__main__':
    args = {
        'model_name' : 'train_from_scratch_trial1400',
        'dataset_name' : 'dataset_for_train_from_scratch',
        'embedding_name' : 'ours',
        'train_embedding' : True,
        'use_augmentation' : False,
        'learning_rate' : 0.001,
        'batch_size' : 1750,
        'num_epochs': 1400,
        'frame_stack': 4,
        'resume' : False,
        'save_model' : True
    }
    train(args)