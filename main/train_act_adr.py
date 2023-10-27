import os
import pickle
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import numpy as np
import sapien.core as sapien
import torch
import wandb
import copy
import h5py
import time

import multiprocessing as mp

from main.policy.act_agent import ActAgent

from eval_act import Eval_player
from adr import adr, aug_in_adr
from logger import Logger
from dataset.act_dataset import argument_dependecy_checker, prepare_sim_aug_data, prepare_real_data, set_seed
from main.train_act import train_in_one_epoch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_and_aug(args, demo_files, log_dir, current_rank):

    set_seed(args["seed"])
    # Augment the data
    sim_aug_demo_length, sim_aug_path, adr_dict = aug_in_adr(
        args, current_rank, demo_files)
    # Prepare the data
    Prepared_Data_Sim = prepare_sim_aug_data(sim_dataset_folder=args['sim_dataset_folder'], sim_dataset_aug_folder=sim_aug_path, sim_aug_demo_length=sim_aug_demo_length, sim_batch_size=args['sim_batch_size'],
                                             val_ratio=args['val_ratio'], seed=20230920, chunk_size=args['num_queries'])

    if args['real_dataset_folder'] is not None:
        Prepared_Data_Real = prepare_real_data(
            args['real_dataset_folder'], args['real_batch_size'])
        sim_real_ratio = Prepared_Data_Sim['total_episodes'] / \
            Prepared_Data_Real['total_episodes']
        print("Sim_Real_Ratio: {}".format(sim_real_ratio))

    print('Data prepared')
    print('---------------------')
    print("Concatenated Observation (State + Visual Obs) Shape: {}".format(
        len(Prepared_Data_Sim['bc_train_set'].dummy_data['obs'])))
    print("Action shape: {}".format(
        len(Prepared_Data_Sim['bc_train_set'].dummy_data['action'])))
    print("robot_qpos shape: {}".format(
        len(Prepared_Data_Sim['bc_train_set'].dummy_data['robot_qpos'])))

    # make agent
    agent = ActAgent(args)
    if current_rank == 1:
        epochs = args['num_epochs']
        eval_freq = args['eval_freq']
    elif current_rank > 1 and not adr_dict['is_stop']:
        agent.load(os.path.join(
            args['sim_aug_dataset_folder'], f"epoch_best.pt"))
        epochs = 300  # 100, 200
        eval_freq = 100  # 25, 50
    elif adr_dict['is_stop']:
        agent.load(os.path.join(
            args['sim_aug_dataset_folder'], f"epoch_best.pt"))
        epochs = 2000  # 100, 200
        eval_freq = 100  # 25, 50

    L = Logger("{}_{}".format(args['model_name'], epochs))

    # make evaluation environment
    eval_player = Eval_player(num_workers=10, args=args, policy=agent.policy)
    best_success = 0
    min_loss = 1

    for epoch in range(epochs):
        print('  ', 'Epoch: ', epoch)
        agent.policy.train()
        loss_train_dict_sim = train_in_one_epoch(agent, Prepared_Data_Sim['it_per_epoch'], Prepared_Data_Sim['bc_train_dataloader'],
                                                 Prepared_Data_Sim['bc_validation_dataloader'], L, epoch)
        metrics = {
            "loss/train_sim": loss_train_dict_sim["train"],
            "loss/train_l1_sim": loss_train_dict_sim["train_l1"],
            "loss/train_kl_sim": loss_train_dict_sim["train_kl"]*args['kl_weight'],
            "loss/val_sim": loss_train_dict_sim["val"],
            "epoch": epoch,
            "current_rank": current_rank
        }
        if args['real_dataset_folder'] is not None:
            loss_train_dict_real = train_in_one_epoch(agent, Prepared_Data_Real['it_per_epoch'], Prepared_Data_Real['bc_train_dataloader'],
                                                      Prepared_Data_Real['bc_validation_dataloader'], L, epoch, sim_real_ratio)
            real_metrics = {
                "loss/train_real": loss_train_dict_real["train"],
                "loss/train_l1_real": loss_train_dict_real["train_l1"],
                "loss/train_kl_real": loss_train_dict_real["train_kl"]*args['kl_weight'],
                "loss/val_real": loss_train_dict_real["val"]
            }
            metrics.update(real_metrics)

        if args['dann']:
            if loss_train_dict_real["train_l1"] < min_loss:
                min_loss = loss_train_dict_real["train_l1"]
                agent.save(os.path.join(log_dir, f"epoch_best.pt"), args)
            metrics["loss/train_domain_sim"] = loss_train_dict_sim["train_domain"]
            metrics["loss/train_domain_real"] = loss_train_dict_real["train_domain"]

        elif (epoch + 1) % eval_freq == 0 and ((current_rank > 1 and (epoch + 1) >= 100) or (current_rank == 1 and (epoch + 1) >= 300)):
            # total_steps = x_steps * y_steps = 4 * 5 = 20
            torch.cuda.empty_cache()
            with torch.inference_mode():
                agent.policy.eval()
                eval_player.eval_init()
                avg_success = 0
                for rank in range(1, args['randomness_rank']+1):
                    if args['task_name'] in ['pick_place', 'pour']:
                        var_object = [0, 0] if rank < 4 else [
                            0.05, 0.08]  # 0.05, 0.1
                        # -0.08 0.08 /// -0.05 0
                        x = np.linspace(-0.08 -
                                        var_object[0], 0.12+var_object[1], 5)
                        # 0.12 0.18 /// 0.12 0.32
                        if args['task_name'] == 'pick_place':
                            y = np.linspace(
                                0.2-var_object[1], 0.3+var_object[0], 4)
                        elif args['task_name'] == 'pour':
                            y = np.linspace(
                                -0.18-var_object[1]*2, -0.08+var_object[0]/2, 4)
                        for i in range(20):
                            eval_player.eval_start(
                                log_dir, epoch+1, i+1, x[int(i/4)], y[i % 4], rank)
                    elif args['task_name'] == 'dclaw':
                        if rank < 3:
                            var_object = [0, 0]
                        elif rank == 3:
                            var_object = [0.05, 0.1]
                        elif rank >= 4:
                            var_object = [0.1, 0.2]
                        # -0.08 0.08 /// -0.05 0
                        x = np.linspace(-var_object[0], var_object[1], 4)
                        # 0.12 0.18 /// 0.12 0.32
                        y = np.linspace(var_object[1], var_object[1], 5)
                        for i in range(20):
                            eval_player.eval_start(
                                log_dir, epoch+1, i+1, x[int(i/5)], y[i % 5], rank)

                timeout_in_seconds = 80*args['randomness_rank']
                start = time.time()
                rank_list = [i for i in range(1, args['randomness_rank']+1)]
                avg_success_chunk = eval_player.eval_get_result()

                ################# Wait for all the processes to finish#################
                while time.time() - start <= timeout_in_seconds:
                    if len(rank_list) == 0:
                        metrics['eval_time'] = time.time() - start
                        break
                    time.sleep(.1)
                    for rank in rank_list:
                        if len(avg_success_chunk[rank]) == 20:
                            metrics[f"avg_success_{rank}"] = np.mean(
                                avg_success_chunk[rank])
                            avg_success += metrics[f"avg_success_{rank}"]
                            rank_list.remove(rank)
                eval_player.eval_terminate()

                ################# Calculate the average success#################
                avg_success = avg_success/args['randomness_rank']
                metrics["avg_success"] = avg_success
                if avg_success > best_success:
                    best_success = avg_success
                    if best_success > adr_dict['last_best_success']:
                        agent.save(os.path.join(
                            args['sim_aug_dataset_folder'], f"epoch_best.pt"), args)
                metrics["best_success"] = best_success

        if current_rank > 1:

            metrics["var_adr_light"] = adr_dict['var_adr_light']
            metrics["var_adr_object"] = adr_dict['var_adr_object']

            if args['task_name'] in ['pick_place', 'pour']:
                metrics["var_adr_target"] = adr_dict['var_adr_target']

        wandb.log(metrics)

    adr_dict["best_success"] = best_success

    return adr(args, current_rank, adr_dict)


def main(args):

    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        "logs", f"{args['sim_dataset_folder'].split('/')[-1]}_sim_{cur_time}")

    wandb.init(
        project="hand-teleop-adr-rank",
        name=os.path.basename(log_dir),
        config=args
    )
    os.makedirs(log_dir, exist_ok=True)

    demo_files = []
    for file_name in os.listdir(args['sim_demo_folder']):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(args['sim_demo_folder'], file_name))

    ########## Initialize the ADR parameters##########
    current_rank = 1
    for iteration in range(500):
        torch.cuda.empty_cache()
        current_rank, is_stop = train_and_aug(
            args, demo_files, log_dir, current_rank)

        if is_stop:
            ############## Final Train#################
            torch.cuda.empty_cache()
            current_rank, is_stop = train_and_aug(
                args, demo_files, log_dir, current_rank)
            print(
                '#################################Stop training##############################')
            break

    wandb.finish()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--real-demo-folder", default=None, type=str)
    parser.add_argument("--sim-folder", default=None)
    parser.add_argument("--sim-demo-folder", default=None, type=str)
    parser.add_argument("--sim-aug-dataset-folder", default=None, type=str)
    parser.add_argument("--backbone-type", default="regnet_3_2gf")
    parser.add_argument("--eval-freq", default=100, type=int)
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
    parser.add_argument("--randomness-rank", default=2, type=int)
    parser.add_argument("--task-name", default="pick_place", type=str)
    parser.add_argument("--dann", action="store_true")
    parser.add_argument("--domain_weight", default=20, type=float)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    args = {
        'real_dataset_folder': args.real_demo_folder,
        'sim_dataset_folder': args.sim_demo_folder,
        "sim_demo_folder": args.sim_folder,
        'sim_aug_dataset_folder': args.sim_aug_dataset_folder,
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
        'backbone_type': args.backbone_type,
        'hidden_dim': args.hidden_dim,
        'robot_name': 'xarm6_allegro_modified_finger',
        'use_visual_obs': True,
        'adapt': False,
        "eval_freq": args.eval_freq,
        "eval_only": args.eval_only,
        "finetune": args.finetune,
        "dann": args.dann,
        'domain_weight': args.domain_weight,
        "randomness_rank": args.randomness_rank,
        "ckpt": args.ckpt,
        "task_name": args.task_name,
        "seed": 20230930
    }
    args = argument_dependecy_checker(args)

    main(args)
