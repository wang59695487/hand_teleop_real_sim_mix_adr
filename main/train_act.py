import os
import pickle
import shutil
import time
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import imageio
import numpy as np
import sapien.core as sapien
import torch
import wandb

from main.policy.act_agent import ActAgent

from eval_act import eval_in_env
from logger import Logger
from tqdm import tqdm
from dataset.act_dataset import argument_dependecy_checker, prepare_real_sim_data, set_seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(agent, validation_loader, L, epoch):
    loss_val = 0
    with torch.inference_mode():
        agent.policy.eval()
        for iter, data_batch in enumerate(validation_loader):
            obs, robot_qpos, action, label, is_pad, _ = data_batch
            obs, robot_qpos, action, label, is_pad = obs.cuda(), robot_qpos.cuda(), action.cuda(), label.cuda(), is_pad.cuda()
            loss = agent.evaluate(obs, robot_qpos, action, is_pad)
            loss_val += loss

    loss_val /= len(validation_loader)

    return loss_val

################## compute loss in one iteration #######################
def compute_loss(agent, bc_train_dataloader, L, epoch):

    data_batch = next(iter(bc_train_dataloader))
    obs, robot_qpos, action, label, is_pad, sim_real_label = data_batch
    obs, robot_qpos, action, label, is_pad, sim_real_label = obs.cuda(), robot_qpos.cuda(), action.cuda(), label.cuda(), is_pad.cuda(), sim_real_label.cuda()
    loss_dict = agent.compute_loss(obs, robot_qpos, action, is_pad, sim_real_label)
    l1_loss = loss_dict['l1']
    kl_loss = loss_dict['kl']
    domain_loss = loss_dict['domain']
    loss = loss_dict['loss']

    return l1_loss,kl_loss,domain_loss,loss

def train_real_sim_in_one_epoch(agent, sim_real_ratio, it_per_epoch_real, it_per_epoch_sim, bc_train_dataloader_real, 
                                bc_validation_dataloader_real, bc_train_dataloader_sim, bc_validation_dataloader_sim, L, epoch):
    
    loss_train_sim = 0
    loss_l1_sim = 0
    loss_kl_sim = 0
    for _ in tqdm(range(it_per_epoch_sim)):

        l1_loss, kl_loss, act_loss = compute_loss(agent,bc_validation_dataloader_sim,L,epoch)
        loss = agent.update_policy(act_loss)
        loss_train_sim += loss
        loss_l1_sim += l1_loss.detach().cpu().item()
        loss_kl_sim += kl_loss.detach().cpu().item()

    loss_val_sim = evaluate(agent, bc_train_dataloader_sim, L, epoch)
    agent.policy.train()

    loss_train_real = 0
    loss_l1_real = 0
    loss_kl_real = 0
    for _ in tqdm(range(it_per_epoch_real)):

        l1_loss, kl_loss, act_loss = compute_loss(agent,bc_validation_dataloader_real,L,epoch)
        loss = agent.update_policy(act_loss*sim_real_ratio)
        loss_train_real += loss
        loss_l1_real += l1_loss.detach().cpu().item()
        loss_kl_real += kl_loss.detach().cpu().item()

    loss_val_real = evaluate(agent, bc_train_dataloader_real, L, epoch)
    agent.policy.train()
   
    loss_real_sim_mix = {'loss_train_real':loss_train_real,'loss_train_sim':loss_train_sim,
                     'loss_val_real':loss_val_real,'loss_val_sim':loss_val_sim, 
                     'loss_l1_real':loss_l1_real,'loss_l1_sim':loss_l1_sim,
                     'loss_kl_real':loss_kl_real,'loss_kl_sim':loss_kl_sim
                     }
    
    return loss_real_sim_mix

def train_in_one_epoch(agent, it_per_epoch, bc_train_dataloader, bc_validation_dataloader, L, epoch, sim_real_ratio=1):
    
    loss_train = 0
    loss_train_l1 = 0
    loss_train_kl = 0
    loss_train_domain = 0
    for _ in tqdm(range(it_per_epoch)):

        l1_loss, kl_loss,domain_loss, act_loss = compute_loss(agent,bc_train_dataloader,L,epoch)
        loss = agent.update_policy(act_loss*sim_real_ratio)
        loss_train += loss
        loss_train_l1 += l1_loss.detach().cpu().item()
        loss_train_kl += kl_loss.detach().cpu().item()
        loss_train_domain += domain_loss.detach().cpu().item()

    loss_val = evaluate(agent, bc_validation_dataloader, L, epoch)
    agent.policy.train()

    return loss_train/(it_per_epoch),loss_train_l1/(it_per_epoch), loss_train_kl/(it_per_epoch),loss_train_domain/(it_per_epoch), loss_val

def main(args):
    # read and prepare data
    set_seed(args["seed"])
    Prepared_Data = prepare_real_sim_data(args['real_dataset_folder'],args['sim_dataset_folder'],args["backbone_type"],args['real_batch_size'],args['sim_batch_size'],
                                 args['val_ratio'], seed = 20230920, chunk_size=args['num_queries'])
    print('Data prepared')
    if Prepared_Data['data_type'] == "real_sim":
        print("##########################Training Sim and Real##################################")
        sim_real_ratio = Prepared_Data['sim_real_ratio']
        print('Sim Data : Real Data = ', sim_real_ratio)
        bc_train_set = Prepared_Data['bc_train_set_real']
    else:
        bc_train_set = Prepared_Data['bc_train_set']
    
    concatenated_obs_shape = len(bc_train_set.dummy_data['obs'])
    print("Concatenated Observation (State + Visual Obs) Shape: {}".format(concatenated_obs_shape))
    action_shape = len(bc_train_set.dummy_data['action'])
    robot_qpos_shape = len(bc_train_set.dummy_data['robot_qpos'])
    print("Action shape: {}".format(action_shape))
    print("robot_qpos shape: {}".format(robot_qpos_shape))
    # make agent
    agent = ActAgent(args)

    L = Logger("{}_{}".format(args['model_name'],args['num_epochs']))
    if not args["finetune"]:
        if not args["eval_only"]:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            if Prepared_Data['data_type'] == "real":
                log_dir = os.path.join("logs", f"{args['real_dataset_folder'].split('/')[-1]}_{Prepared_Data['data_type']}_{args['backbone_type']}_{cur_time}")
            else:
                log_dir = os.path.join("logs", f"{args['sim_dataset_folder'].split('/')[-1]}_{Prepared_Data['data_type']}_{args['backbone_type']}_{cur_time}")
            wandb.init(
                project="hand-teleop",
                name=os.path.basename(log_dir),
                config=args
            )
            os.makedirs(log_dir, exist_ok=True)

            best_success = 0

            for epoch in range(args['num_epochs']):
                print('  ','Epoch: ', epoch)
                agent.policy.train()
                if Prepared_Data['data_type'] == "real_sim":

                    loss_real_sim = train_real_sim_in_one_epoch(agent,sim_real_ratio,
                                                    Prepared_Data['it_per_epoch_real'],Prepared_Data['it_per_epoch_sim'],
                                                    Prepared_Data['bc_train_dataloader_real'], Prepared_Data['bc_validation_dataloader_real'], 
                                                    Prepared_Data['bc_train_dataloader_sim'], Prepared_Data['bc_validation_dataloader_sim'], L, epoch)

                    metrics = {
                        "loss/train_real": loss_real_sim['loss_train_real'],
                        "loss/train_sim": loss_real_sim['loss_train_sim'],
                        "loss/val_real": loss_real_sim['loss_val_real'],
                        "loss/val_sim": loss_real_sim['loss_val_sim'],
                        "loss/train_l1_real": loss_real_sim['loss_l1_real'],
                        "loss/train_l1_sim": loss_real_sim['loss_l1_sim'],
                        "loss/train_kl_real": loss_real_sim['loss_kl_real'],
                        "loss/train_kl_sim": loss_real_sim['loss_kl_sim'],
                        "epoch": epoch
                    }

                else:

                    loss_train,loss_l1,loss_kl, loss_val = train_in_one_epoch(agent, Prepared_Data['it_per_epoch'], Prepared_Data['bc_train_dataloader'], 
                                                            Prepared_Data['bc_validation_dataloader'], L, epoch)
                    metrics = {
                        "loss/train": loss_train,
                        "loss/train_l1": loss_l1,
                        "loss/train_kl": loss_kl,
                        "loss/val": loss_val,
                        "epoch": epoch
                    }
                
                if (epoch + 1) % args["eval_freq"] == 0 and (epoch+1) >= args["eval_start_epoch"]:
                    ##total_steps = x_steps * y_steps = 4 * 5 = 20
                    if Prepared_Data['data_type'] == "sim":
                        with torch.inference_mode():
                            agent.policy.eval()
                            avg_success = eval_in_env(args, agent, log_dir, epoch + 1, 4, 5)
                        metrics["avg_success"] = avg_success
                        if avg_success > best_success:
                            agent.save(os.path.join(log_dir, f"epoch_best.pt"), args)
                            
                    agent.save(os.path.join(log_dir, f"epoch_{epoch + 1}.pt"), args)
                    

                wandb.log(metrics)
            
            if Prepared_Data['data_type'] == "sim":
                agent.load(os.path.join(log_dir, "epoch_best.pt"))
                with torch.inference_mode():
                    agent.policy.eval()
                    final_success = eval_in_env(args, agent, log_dir, "best", 10, 10)
                wandb.log({"final_success": final_success})
                print(f"Final success rate: {final_success:.4f}")
                
            wandb.finish()

        else:
            path = args["ckpt"].split("/")[-2]+"_eval_randomness_rank_"+str(args["randomness_rank"])
            log_dir = os.path.dirname("./logs/"+path+"/")
            wandb.init(
                project="hand-teleop_eval",
                name=path,
                config=args
            )
            os.makedirs(log_dir, exist_ok=True)
            agent.load(args["ckpt"])
            with torch.inference_mode():
                agent.policy.eval()
                final_success = eval_in_env(args, agent, log_dir, "best", 10, 10)
            wandb.log({"final_success": final_success})
            print(f"Final success rate: {final_success:.4f}")
            wandb.finish()

    else:
        best_loss = 1
        agent.finetune(args)
        print("##########################Finetune##################################")
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        assert Prepared_Data['data_type'] == "real"
        log_dir = os.path.join("logs", f"{args['ckpt'].split('/')[-2]}_Finetuning_{args['backbone_type']}_{cur_time}")
        
        wandb.init(
            project="hand-teleop-adr-rank",
            name=os.path.basename(log_dir),
            config=args
        )
        
        os.makedirs(log_dir, exist_ok=True)

        for epoch in range(args['num_epochs']):
            print('  ','Epoch: ', epoch)
            agent.policy.train()

            loss_train,loss_l1,loss_kl, loss_val = train_in_one_epoch(agent, Prepared_Data['it_per_epoch'], Prepared_Data['bc_train_dataloader'], 
                                                    Prepared_Data['bc_validation_dataloader'], L, epoch)
            metrics = {
                "loss/train": loss_train,
                "loss/train_l1": loss_l1,
                "loss/train_kl": loss_kl*args['kl_weight'],
                "loss/val": loss_val,
                "epoch": epoch
            }
            
            wandb.log(metrics)
            if (epoch+1) >= args["eval_start_epoch"]:
                if loss_l1 < best_loss:
                    best_loss = loss_l1
                    agent.save(os.path.join(log_dir, f"epoch_best.pt"), args)

        wandb.finish()
        

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--real-demo-folder", default=None, type=str)
    parser.add_argument("--sim-demo-folder", default=None, type=str)
    parser.add_argument("--backbone-type", default="regnet_3_2gf")
    parser.add_argument("--eval-freq", default=100, type=int)
    parser.add_argument("--eval-start-epoch", default=400, type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--kl_weight", default=20, type=int)
    parser.add_argument("--dim_feedforward", default=3200, type=int)
    parser.add_argument("--num-epochs", default=2000, type=int)
    parser.add_argument("--real-batch-size", default=64, type=int)
    parser.add_argument("--sim-batch-size", default=64, type=int)
    parser.add_argument("--num_queries", default=50, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--randomness-rank", default=1, type=int)
    parser.add_argument("--dann", action="store_true")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    args = {
        'real_dataset_folder': args.real_demo_folder,
        'sim_dataset_folder': args.sim_demo_folder,
        'num_queries': args.num_queries,
         # 8192 16384 32678 65536
        'real_batch_size': args.real_batch_size,
        'sim_batch_size': args.sim_batch_size,
        'val_ratio': args.val_ratio,
        'lr': args.lr,
        'num_epochs': args.num_epochs,        
        'weight_decay': args.weight_decay,
        'kl_weight': args.kl_weight,
        'dim_feedforward': args.dim_feedforward,
        'model_name': '',
        'backbone_type' : args.backbone_type,
        'hidden_dim': args.hidden_dim,
        'robot_name': 'xarm6_allegro_modified_finger',
        'use_visual_obs': True,
        'adapt': False,
        "eval_freq": args.eval_freq,
        "eval_start_epoch": args.eval_start_epoch,
        "eval_only": args.eval_only,
        "finetune": args.finetune,
        "randomness_rank": args.randomness_rank,
        "ckpt": args.ckpt,
        "seed": 20230915
    }
    args = argument_dependecy_checker(args)

    main(args)