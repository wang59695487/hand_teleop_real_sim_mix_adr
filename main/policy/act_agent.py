# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.act.detr_vae import build_ACT_model
from dataset.act_dataset import set_seed

import IPython
e = IPython.embed


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)  # will be overridden
    parser.add_argument("--weight_decay", default=1e-2,
                        type=float)  # will be overridden
    parser.add_argument("--kl_weight", default=10, type=int)
    # Model parameters

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str,  # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # parser.add_argument('--camera_names', default=[], type=list, # will be overridden
    #                     help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,  # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,  # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,  # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,  # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,  # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=50, type=int,  # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Not used
    parser.add_argument("--real-demo-folder", default=None, type=str)
    parser.add_argument("--sim-demo-folder", default=None, type=str)
    parser.add_argument("--sim-dataset-folder", default=None)
    parser.add_argument("--sim-aug-dataset-folder", default=None, type=str)
    parser.add_argument("--backbone-type", default="regnet_3_2gf")
    parser.add_argument("--eval-freq", default=100, type=int)
    parser.add_argument("--eval-start-epoch", default=400, type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--num-epochs", default=2000, type=int)
    parser.add_argument("--real-batch-size", default=32678, type=int)
    parser.add_argument("--sim-batch-size", default=32678, type=int)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--eval-randomness-scale", default=0, type=int)
    parser.add_argument("--randomness-rank", default=1, type=int)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--task-name", default="pick_place", type=str)
    parser.add_argument("--dann", action="store_true")
    parser.add_argument("--is-feature", default=False, type=bool)
    parser.add_argument("--domain_weight", default=20, type=float)
    parser.add_argument("--object-name", default="mustard_bottle", type=str)

    return parser


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.domain_weight = args_override['domain_weight']
        self.dann = args_override['dann']
        self.is_feature = args_override['is_feature']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, obs, qpos, actions=None, is_pad=None, sim_real_label=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        obs = normalize(obs) if not self.is_feature else obs

        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            if self.dann:
                a_hat, is_pad_hat, (mu, logvar), domain_logits = self.model(
                    obs, qpos, actions, is_pad)
                domain_loss = F.binary_cross_entropy(
                    domain_logits, sim_real_label)
            else:
                a_hat, is_pad_hat, (mu, logvar) = self.model(
                    obs, qpos, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            if self.dann:
                loss_dict['domain'] = domain_loss
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * \
                    self.kl_weight + loss_dict['domain'] * self.domain_weight
            else:
                loss_dict['loss'] = loss_dict['l1'] + \
                    loss_dict['kl'] * self.kl_weight
            return loss_dict

        else:  # inference time
            # no action, sample from prior
            a_hat, _, (_, _) = self.model(obs, qpos)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class ActAgent(object):
    def __init__(self, args):

        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'weight_decay': args['weight_decay'],
                         'num_queries': args['num_queries'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': 1e-5,
                         'backbone': 'resnet18',
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'dann': args['dann'],
                         'domain_weight': args['domain_weight'],
                         'is_feature': args['is_feature'],
                         }

        set_seed(args['seed'])
        self.policy = ACTPolicy(policy_config)
        self.policy.cuda()
        self.optimizer = self.policy.configure_optimizers()

    def compute_loss(self, obs, qpos, actions, is_pad, sim_real_label):
        loss_dict = self.policy(obs, qpos, actions, is_pad, sim_real_label)
        return loss_dict

    def update_policy(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def evaluate(self, obs, qpos, action=None, is_pad=None):
        pred_action = self.policy(obs, qpos)

        if action is not None:
            assert is_pad is not None
            all_l1 = F.l1_loss(pred_action, action, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            return l1.detach().cpu().item()

        return pred_action

    def save(self, weight_path, args):
        torch.save({
            'act_network_state_dict': self.policy.state_dict(),
            'args': args,
        }, weight_path
        )

    def load(self, weight_path):
        act_network_checkpoint = torch.load(weight_path)
        self.policy.load_state_dict(
            act_network_checkpoint['act_network_state_dict'])
        args = act_network_checkpoint['args']
        return args

    def finetune(self, args):
        act_network_checkpoint = torch.load(args["ckpt"])
        self.policy.load_state_dict(
            act_network_checkpoint['act_network_state_dict'])

        # for name, param in self.policy.model.named_parameters():
        #     param.requires_grad = False

        # # for param in self.policy.model.encoder.parameters():
        # #     if param.dim() > 1:
        # #         # nn.init.xavier_uniform_(param)
        # #         param.requires_grad = True

        # for param in self.policy.model.backbones.parameters():
        #     if param.dim() > 1:
        #         # nn.init.xavier_uniform_(param)
        #         param.requires_grad = True

        # for name, param in self.policy.model.named_parameters():
        #     if "transformer.encoder.layers.3" in name:
        #         if param.dim() > 1:
        #             # nn.init.xavier_uniform_(param)
        #             param.requires_grad = True
        #     elif "transformer.decoder.layers.5" in name:
        #         if param.dim() > 1:
        #             # nn.init.xavier_uniform_(param)
        #             param.requires_grad = True
        #     elif "transformer.decoder.layers.6" in name:
        #         if param.dim() > 1:
        #             # nn.init.xavier_uniform_(param)
        #             param.requires_grad = True
        #     elif "encoder.layers.3" in name:
        #         if param.dim() > 1:
        #             # nn.init.xavier_uniform_(param)
        #             param.requires_grad = True

        ############## Initialize optimizer and BatchNorm##################
        pg = [p for _, p in self.policy.model.named_parameters()
              if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            pg, lr=args['lr'], weight_decay=args['weight_decay'])
        args = act_network_checkpoint['args']
        return args
