import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import pickle
import os

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BehaviorCloning(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BehaviorCloning, self).__init__()
        self.fc1 = nn.Linear(in_dim, 2000)
        self.bn1 = nn.BatchNorm1d(2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, 200)
        self.bn3 = nn.BatchNorm1d(200)
        self.fc4 = nn.Linear(200, out_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x) 
        x = self.fc4(x)        
        return x


class BehaviorCloningMultiModal(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(BehaviorCloningMultiModal, self).__init__()
        self.image_encoder = nn.Sequential(
            nn.Linear(in_dim-(58*4),1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim*4)
        )        
        self.robot_state_encoder = nn.Sequential(
            nn.Linear(58*4, out_dim*4),
            nn.BatchNorm1d(out_dim*4),
            nn.ReLU(),
            nn.Linear(out_dim*4, out_dim*4)
        )
        self.combinator = nn.Sequential(
            nn.Linear(out_dim*4, out_dim*2),
            nn.BatchNorm1d(out_dim*2),
            nn.ReLU(),
            nn.Linear(out_dim*2, out_dim*2),
            nn.BatchNorm1d(out_dim*2),
            nn.ReLU(),
            nn.Linear(out_dim*2, out_dim)
        )
        self.bn = nn.BatchNorm1d(out_dim*4)
    def forward(self, img_features, robot_states):
        encoded_img_features  = self.image_encoder(img_features)
        encoded_robot_states = self.robot_state_encoder(robot_states)
        output = self.combinator(nn.functional.relu(encoded_img_features + encoded_robot_states))
        return output


def train_bc(args):

    DATAPATH = './sim/baked_data/{}.pickle'.format(args['dataset_name'])
    model_path = './trained_models/{}_{}.pt'.format(args['task_props'], args['model_backbone'])
    if args['use_visual_obs']:
        if args['stack_frames']:
            if args['model_backbone'] == 'ResNet34' or args['model_backbone'] == 'MoCo18':
                model = BehaviorCloning(2280,51) #9320
            else:
                model = BehaviorCloning(8424,51)
        else:
            if args['encode']:
                model = BehaviorCloning(2280,51) #2280
            else:
                model = BehaviorCloning(2106,51)
    else:
        model = BehaviorCloning(63, 51) #69,51 

    writer = SummaryWriter(log_dir='runs/{}_{}'.format(args['task_props'], args['model_backbone']))

    # Set up data
    file = open(DATAPATH,'rb')
    data = pickle.load(file)
    random_order = np.random.permutation(len(data["obs"]))
    obs = np.array(data["obs"])
    obs = obs[random_order]
    obs = torch.from_numpy(obs).float()
    validation_obs = obs[:int(len(obs)/5)]
    obs = obs[int(len(obs)/5):]
    obs = obs.to(device)
    validation_obs = validation_obs.to(device)
    targets = np.array(data["action"])
    targets = targets[random_order]
    targets = torch.from_numpy(targets).float()
    validation_targets = targets[:int(len(targets)/5)]
    targets = targets[int(len(targets)/5):]
    targets = targets.to(device)
    validation_targets = validation_targets.to(device)
    print('=== Loading trajectories ===')
    print('  ', 'total number of samples', len(obs))

    # Training Specs
    num_epochs = args['num_epochs']
    learning_rate = args['learning_rate']
    params = list(model.parameters())
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(params=params, lr=learning_rate, weight_decay=0.01)
    # lambda1 = lambda epoch: epoch**0.9999
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda1)
    model = model.to(device).float()
    criterion = criterion.to(device)
    print('=== BC run ===')
    print('  ', 'epochs:', args['num_epochs'])
    print('  ', 'lr:', args['learning_rate'])

    print('=== Training policy ===')
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(obs)
        loss = criterion(outputs, targets)
        training_loss = loss.item()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        print("Epoch: {}    Loss: {}".format(epoch, training_loss/len(data)))

        # Validation
        model.eval()
        with torch.no_grad():
            validation_outputs = model(validation_obs)
            validation_loss = criterion(validation_outputs,validation_targets)

        # TensorBoard
        writer.add_scalar('Loss/train', training_loss/len(data), epoch)
        writer.add_scalar('Loss/test', validation_loss.item()/len(validation_targets), epoch)

    model_dict = model.state_dict()
    state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state_dict, model_path)
    print("Training Complete")
    print("Behavior Cloning Model is saved to: {}".format(model_path))


if __name__ == '__main__':
    args = {
        'num_epochs': 2000,
        'learning_rate': 0.001,
        'dataset_name': 'dataset_lessrandom_pick_place',
        'task_props': 'lessrandom_pick_place',
        'model_backbone': 'MoCo50',
        'use_visual_obs': True,
        'stack_frames': True,
        'encode': False
    }
    train_bc(args)


