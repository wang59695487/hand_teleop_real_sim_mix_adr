import os
import pickle
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import numpy as np
import sapien.core as sapien
import torch, wandb, copy, h5py, time

import multiprocessing as mp

from main.policy.act_agent import ActAgent

from eval_act import Eval_player
from logger import Logger
from dataset.act_dataset import argument_dependecy_checker, prepare_sim_aug_data, set_seed
from main.train_act import train_in_one_epoch
from hand_teleop.player.player_augmentation import generate_sim_aug_in_play_demo
from hand_teleop.player.play_multiple_demonstrations_act import stack_and_save_frames_aug

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_aug(args, demo_files, log_dir, current_rank):
    # read and prepare data
    set_seed(args["seed"])
       
    if current_rank > 1: 

        for file_name in os.listdir(args['sim_aug_dataset_folder']):
            if "meta_data.pickle" in file_name:
                meta_data_path = f"{args['sim_aug_dataset_folder']}/meta_data.pickle" 
                with open(meta_data_path,'rb') as file:
                    all_meta_data = pickle.load(file)
                last_total_episodes = all_meta_data['total_episodes']
                total_episodes = last_total_episodes
                init_obj_poses = all_meta_data['init_obj_poses']
                last_best_success = all_meta_data['best_success']
                var_adr_light = all_meta_data['var_adr_light']
                var_adr_plate = all_meta_data['var_adr_plate']
                var_adr_object = all_meta_data['var_adr_object']
                is_var_adr = all_meta_data['is_var_adr']
                is_stop = all_meta_data['is_stop']
                break
            else:
                last_total_episodes = 0
                total_episodes = 0
                init_obj_poses = []
                var_adr_light = 1
                var_adr_plate = 0.02
                var_adr_object = 0.02
                is_var_adr = True
                is_stop = False
                meta_data_path = f"{args['sim_dataset_folder']}/meta_data.pickle" 
                with open(meta_data_path,'rb') as file:
                    all_meta_data = pickle.load(file)
                last_best_success = all_meta_data['best_success']

        print('Replaying the sim demos and augmenting the dataset:')
        print('---------------------')
        aug = {2:10,3:5,4:10,5:100}
        ########### Add new sim demos to the original dataset ###########
        file1 = h5py.File(f"{args['sim_aug_dataset_folder']}/dataset.h5", 'a')
        for i in range(400):
            for _ , file_name in enumerate(demo_files):
                print(file_name)
                if args['task_name'] == 'pick_place':
                    var_obj = var_adr_object if current_rank >= 4 else 0
                    var_plate = var_adr_plate if current_rank >= 3 else 0
                    x2 = np.random.uniform(-0.02-var_plate, 0.02 + var_plate)
                    y2 = np.random.uniform(-0.02-var_plate*2, 0.02)
                    if np.fabs(x2) <= 0.01 and np.fabs(y2) <= 0.01:
                        continue
                    init_pose_aug_plate = sapien.Pose([x2, y2, 0], [1, 0, 0, 0])
                    
                elif args['task_name'] == 'dclaw':
                    var_obj = var_adr_object if current_rank >= 3 else 0
                    init_pose_aug_plate = None
        
                x1, y1 = np.random.uniform(-0.02-var_obj,0.02+var_obj,2)        
                if np.fabs(x1) <= 0.01 and np.fabs(y1) <= 0.01:
                    continue
                init_pose_aug_obj = sapien.Pose([x1, y1, 0], [1, 0, 0, 0])
               
                with open(file_name, 'rb') as file:
                    demo = pickle.load(file)
                all_data = copy.deepcopy(demo)

                visual_baked, meta_data, info_success = generate_sim_aug_in_play_demo(args, demo=all_data, init_pose_aug_plate=init_pose_aug_plate , 
                                                                                      init_pose_aug_obj=init_pose_aug_obj, var_adr_light=var_adr_light)
                if not info_success:
                    continue
                aug[current_rank] -= 1
                init_obj_poses.append(meta_data['env_kwargs']['init_obj_pos'])
                total_episodes, _, _ , _ = stack_and_save_frames_aug(visual_baked, total_episodes, args, file1)
            
                if aug[current_rank] <= 0:
                    break

            if aug[current_rank] <= 0:
                    break  
                
        file1.close()
        sim_aug_demo_length = total_episodes
        sim_aug_path = args['sim_aug_dataset_folder']
    else:
        os.makedirs(args['sim_aug_dataset_folder'])
        last_best_success = 0
        sim_aug_demo_length = None
        sim_aug_path = None
        is_stop = False

    Prepared_Data = prepare_sim_aug_data(sim_dataset_folder=args['sim_dataset_folder'],sim_dataset_aug_folder=sim_aug_path, sim_aug_demo_length=sim_aug_demo_length, sim_batch_size=args['sim_batch_size'],
                                 val_ratio=args['val_ratio'], seed = 20230920, chunk_size=args['num_queries'])
    
    print('Data prepared')
    print('---------------------')
    print("Concatenated Observation (State + Visual Obs) Shape: {}".format(len(Prepared_Data['bc_train_set'].dummy_data['obs'])))
    print("Action shape: {}".format(len(Prepared_Data['bc_train_set'].dummy_data['action'])))
    print("robot_qpos shape: {}".format(len(Prepared_Data['bc_train_set'].dummy_data['robot_qpos'])))
    
    # make agent
    agent = ActAgent(args)
    if current_rank == 1:
        epochs = args['num_epochs']
        eval_freq = args['eval_freq']
    elif current_rank > 1 and not is_stop:
        agent.load(os.path.join(args['sim_aug_dataset_folder'], f"epoch_best.pt"))
        epochs = 300  # 100, 200            
        eval_freq = 100 # 25, 50
    elif is_stop:
        agent.load(os.path.join(args['sim_aug_dataset_folder'], f"epoch_best.pt"))
        epochs = 2500  # 100, 200
        eval_freq = 100 # 25, 50
    
    L = Logger("{}_{}".format(args['model_name'],epochs))
    
    # make evaluation environment
    num_workers = 10
    eval_player = Eval_player(num_workers, args, agent.policy)
    best_success = 0
    for epoch in range(epochs):
        print('  ','Epoch: ', epoch)
        agent.policy.train()
        loss_train,loss_l1,loss_kl, loss_val = train_in_one_epoch(agent, Prepared_Data['it_per_epoch'], Prepared_Data['bc_train_dataloader'], 
                                                Prepared_Data['bc_validation_dataloader'], L, epoch)
        metrics = {
            "loss/train": loss_train,
            "loss/train_l1": loss_l1,
            "loss/train_kl": loss_kl*args['kl_weight'],
            "loss/val": loss_val,
            "epoch": epoch,
            "current_rank": current_rank
        }
        
        if (epoch + 1) % eval_freq == 0 and ((current_rank > 1 and (epoch + 1) >= 100) or (current_rank == 1 and (epoch + 1) >= 300)):
            ##total_steps = x_steps * y_steps = 4 * 5 = 20
            torch.cuda.empty_cache()
            with torch.inference_mode():
                agent.policy.eval()
                eval_player.eval_init()
                avg_success = 0
                for rank in range(1,args['randomness_rank']+1):
                    var_object = [0,0] if rank < 4 else [0.05,0.08]  # 0.05, 0.1
                    x = np.linspace(-0.08-var_object[0], 0.12+var_object[1], 5)   # -0.08 0.08 /// -0.05 0
                    y = np.linspace(0.2-var_object[1], 0.3+var_object[1], 4)  # 0.12 0.18 /// 0.12 0.32
                    for i in range(20):
                        eval_player.eval_start(log_dir, epoch+1, i+1, x[int(i/4)], y[i%4], rank)
               
                timeout_in_seconds = 80*args['randomness_rank']
                start = time.time()
                rank_list = [i for i in range(1,args['randomness_rank']+1)]
                avg_success_chunk = eval_player.eval_get_result()
                
                #################Wait for all the processes to finish#################
                while time.time() - start <= timeout_in_seconds:
                    if len(rank_list) == 0:
                        metrics['eval_time'] = time.time() - start
                        break
                    time.sleep(.1)
                    for rank in rank_list:
                        if len(avg_success_chunk[rank]) == 20:
                            metrics[f"avg_success_{rank}"] = np.mean(avg_success_chunk[rank])
                            avg_success += metrics[f"avg_success_{rank}"]
                            rank_list.remove(rank)
                eval_player.eval_terminate()
                    
                #################Calculate the average success#################
                avg_success = avg_success/args['randomness_rank']
                metrics["avg_success"] = avg_success
                if avg_success > best_success:
                    best_success = avg_success
                    if best_success > last_best_success:
                        agent.save(os.path.join(args['sim_aug_dataset_folder'], f"epoch_best.pt"), args)
                metrics["best_success"] = best_success

        if current_rank > 1:
            
            metrics["var_adr_light"]=var_adr_light
            metrics["var_adr_object"]=var_adr_object
            
            if args['task_name'] == 'pick_place':
                 metrics["var_adr_plate"]=var_adr_plate

        wandb.log(metrics)
    
    ##################ADR##################
    if best_success <= last_best_success - 0.1:
        total_episodes = last_total_episodes
        best_success = last_best_success
        ################Cancel the augment in environment domain and continue sample in the original domain#################
        if is_var_adr:
            is_var_adr = False
        ################Cancel sample in the original domain and Exit#################
        else:
            current_rank += 1
            is_var_adr = True
    
    ##############Update the ADR parameters for different ranks#################
    if current_rank == 1:
        meta_data_path = f"{args['sim_dataset_folder']}/meta_data.pickle" 
        with open(meta_data_path,'rb') as file:
            all_meta_data = pickle.load(file)
        all_meta_data["best_success"] = best_success
        current_rank += 1
    else:
        meta_data_path = f"{args['sim_aug_dataset_folder']}/meta_data.pickle" 
        if current_rank == 2: 
            var_adr_light = var_adr_light + 0.2 if is_var_adr else var_adr_light
            if var_adr_light > 2:
                var_adr_light = 2
                current_rank += 1
            
        elif current_rank == 3 and args['task_name'] == 'pick_place':
            var_adr_plate = var_adr_plate + 0.02 if is_var_adr else var_adr_plate
            if var_adr_plate > 0.12:
                var_adr_plate = 0.12
                current_rank += 1
            
        elif current_rank == 4 and args['task_name'] == 'pick_place':
            var_adr_object = var_adr_object + 0.02 if is_var_adr else var_adr_object
            if var_adr_object > 0.12:
                var_adr_object = 0.12
                current_rank += 1
        
        elif current_rank == 3 and args['task_name'] == 'dclaw':
            var_adr_object = var_adr_object + 0.01 if is_var_adr else var_adr_object
            if var_adr_object > 0.1:
                var_adr_object = 0.1
                current_rank += 1
        
        elif current_rank == 4 and args['task_name'] == 'dclaw':
            var_adr_object = var_adr_object + 0.01 if is_var_adr else var_adr_object
            if var_adr_object > 0.2:
                var_adr_object = 0.1
                current_rank += 1

        ##################Finish ADR##################
        if current_rank == args['randomness_rank']+1:
            is_stop = True
           
        all_meta_data = {'init_obj_poses': init_obj_poses, 'total_episodes': total_episodes, "best_success": best_success, "var_adr_light": var_adr_light,
                         "var_adr_plate": var_adr_plate, "var_adr_object": var_adr_object, "is_var_adr": is_var_adr,  "is_stop": is_stop}

    with open(meta_data_path,'wb') as file:
        pickle.dump(all_meta_data, file)

    return current_rank, is_stop

def main(args):
    
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", f"{args['sim_dataset_folder'].split('/')[-1]}_sim_{cur_time}")

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
    
    ##########Initialize the ADR parameters##########
    current_rank = 1
    for iteration in range(500):
        torch.cuda.empty_cache()
        current_rank, is_stop = train_and_aug(args, demo_files, log_dir, current_rank)

        if is_stop:
            ##############Final Train#################
            torch.cuda.empty_cache()
            current_rank, is_stop = train_and_aug(args, demo_files, log_dir, current_rank)
            print('#################################Stop training##############################')
            break
    
    wandb.finish()
       
        

def parse_args():
    parser = ArgumentParser()
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
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    args = {
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
        'backbone_type' : args.backbone_type,
        'hidden_dim': args.hidden_dim,
        'robot_name': 'xarm6_allegro_modified_finger',
        'use_visual_obs': True,
        'adapt': False,
        "eval_freq": args.eval_freq,
        "eval_only": args.eval_only,
        "finetune": args.finetune,
        "randomness_rank": args.randomness_rank,
        "ckpt": args.ckpt,
        "task_name": args.task_name,
        "seed": 20230930
    }
    args = argument_dependecy_checker(args)

    main(args)