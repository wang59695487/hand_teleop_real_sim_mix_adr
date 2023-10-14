import os
import pickle
import copy
import h5py
import numpy as np
import sapien.core as sapien
from hand_teleop.player.player_augmentation import generate_sim_aug_in_play_demo
from hand_teleop.player.play_multiple_demonstrations_act import stack_and_save_frames_aug

def aug_in_adr(args, current_rank, demo_files):
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
                ##############Initialize the ADR parameters#################
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
        aug = {2:5,3:10,4:15,5:15}
        ########### Add new sim demos to the original dataset ###########
        file1 = h5py.File(f"{args['sim_aug_dataset_folder']}/dataset.h5", 'a')
        for i in range(400):
            for _ , file_name in enumerate(demo_files):
                print(file_name)
                if args['task_name'] == 'pick_place':
                    var_obj = var_adr_object if current_rank >= 4 else 0
                    x1, y1 = np.random.uniform(-0.02-var_obj,0.02+var_obj,2)        
                    if np.fabs(x1) <= 0.01 and np.fabs(y1) <= 0.01:
                        continue
                    init_pose_aug_obj = sapien.Pose([x1, y1, 0], [1, 0, 0, 0])
                    
                    var_plate = var_adr_plate if current_rank >= 3 else 0
                    x2 = np.random.uniform(-0.02-var_plate, 0.02 + var_plate)
                    y2 = np.random.uniform(-0.02-var_plate*2, 0.02)
                    if np.fabs(x2) <= 0.01 and np.fabs(y2) <= 0.01:
                        continue
                    init_pose_aug_plate = sapien.Pose([x2, y2, 0], [1, 0, 0, 0])
                    
                elif args['task_name'] == 'dclaw':
                    var_obj = var_adr_object if current_rank >= 3 else 0
                    x1 = np.random.uniform(-var_obj/2,var_obj/2)
                    y1 = np.random.uniform(-var_obj,var_obj)
                    init_pose_aug_obj = sapien.Pose([x1, y1, 0], [1, 0, 0, 0])
                    init_pose_aug_plate = None
        
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
        adr_dict = {'init_obj_poses': init_obj_poses, 'total_episodes': total_episodes, "last_total_episodes":last_total_episodes, "last_best_success": last_best_success, 
                "var_adr_light": var_adr_light, "var_adr_plate": var_adr_plate, "var_adr_object": var_adr_object, "is_var_adr": is_var_adr, "is_stop": is_stop}
    else:
        os.makedirs(args['sim_aug_dataset_folder'])
        sim_aug_demo_length = None
        sim_aug_path = None
        adr_dict = {"last_best_success": 0 , "is_stop": False}
   

    return sim_aug_demo_length, sim_aug_path, adr_dict
        

def adr(args, current_rank, adr_dict):
    # ##################ADR##################
    # if adr_dict['best_success'] <= adr_dict['last_best_success'] - 0.1:
    #     adr_dict['total_episodes'] = adr_dict['last_total_episodes']
    #     adr_dict['best_success'] = adr_dict['last_best_success']
    #     ################Cancel the augment in environment domain and continue sample in the original domain#################
    #     if adr_dict['is_var_adr']:
    #         adr_dict['is_var_adr'] = False
    #     ################Cancel sample in the original domain and Exit#################
    #     else:
    #         current_rank += 1
    #         adr_dict['is_var_adr'] = True
    
    ##############Update the ADR parameters for different ranks#################
    if current_rank == 1:
        meta_data_path = f"{args['sim_dataset_folder']}/meta_data.pickle" 
        with open(meta_data_path,'rb') as file:
            all_meta_data = pickle.load(file)
        all_meta_data["best_success"] = adr_dict['best_success']
        current_rank += 1
    else:
        meta_data_path = f"{args['sim_aug_dataset_folder']}/meta_data.pickle" 
        if current_rank == 2: 
            adr_dict['var_adr_light'] = adr_dict['var_adr_light'] + 0.2 if adr_dict['is_var_adr'] else adr_dict['var_adr_light']
            if adr_dict['var_adr_light'] > 2:
                adr_dict['var_adr_light'] = 2
                current_rank += 1
            
        elif current_rank == 3 and args['task_name'] == 'pick_place':
            adr_dict['var_adr_plate'] = adr_dict['var_adr_plate'] + 0.02 if adr_dict['is_var_adr'] else adr_dict['var_adr_plate']
            if adr_dict['var_adr_plate'] > 0.1:
                adr_dict['var_adr_plate'] = 0.1
                current_rank += 1
            
        elif current_rank == 4 and args['task_name'] == 'pick_place':
            adr_dict['var_adr_object'] = adr_dict['var_adr_object'] + 0.02 if adr_dict['is_var_adr'] else adr_dict['var_adr_object']
            if adr_dict['var_adr_object'] > 0.12:
                adr_dict['var_adr_object'] = 0.12
                current_rank += 1
        
        elif current_rank == 3 and args['task_name'] == 'dclaw':
            adr_dict['var_adr_object'] = adr_dict['var_adr_object'] + 0.02 if adr_dict['is_var_adr'] else adr_dict['var_adr_object']
            if adr_dict['var_adr_object'] > 0.1:
                adr_dict['var_adr_object'] = 0.1
                current_rank += 1
        
        elif current_rank == 4 and args['task_name'] == 'dclaw':
            adr_dict['var_adr_object'] = adr_dict['var_adr_object'] + 0.02 if adr_dict['is_var_adr'] else adr_dict['var_adr_object']
            if adr_dict['var_adr_object'] > 0.2:
                adr_dict['var_adr_object'] = 0.2
                current_rank += 1

        ##################Finish ADR##################
        if current_rank == args['randomness_rank']+1:
            adr_dict['is_stop'] = True
        
        all_meta_data = adr_dict

    with open(meta_data_path,'wb') as file:
        pickle.dump(all_meta_data, file)
        
    return current_rank, adr_dict['is_stop']