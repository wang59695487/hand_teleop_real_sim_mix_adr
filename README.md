# Usage
## Demo replay
export PYTHONPATH=.

## Training and evaluation
```bash

#batch-size: 8, 16, 32
#######################################ACT Test######################################
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_sugar_box --out-folder=sim/baked_data/pick_place_sg_wo_light_test_act --task-name=pick_place --object-name=sugar_box --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=default --img-data-aug=1  --kinematic-aug=0  > logs/play_sim 2>&1 & 

#kl-weight: 10, 20, 30
nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_sg_wo_light_test_act \
    --backbone-type=regnet_y_3_2gf \
    --sim-batch-size=32 \
    --lr=1e-5 \
    --kl_weight=10 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=1600 \
    --eval-start-epoch=300 \
    --eval-freq=100 > logs/train_sim_act 2>&1 &

################################Mustard Bottle##########################################
#sim demo: 37670  kinematic aug: 0
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_mustard_bottle --out-folder=sim/baked_data/pick_place_mb_wo_light --task-name=pick_place --object-name=mustard_bottle --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --img-data-aug=1 --chunk-size=50  > logs/play_sim 2>&1 & 

#sim demo: 75340  kinematic aug: 0
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_mustard_bottle --out-folder=sim/baked_data/pick_place_mb_w_light --task-name=pick_place --object-name=mustard_bottle --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=random --img-data-aug=2  --kinematic-aug=0  > logs/play_sim 2>&1 & 

#sim demo: 162254  kinematic aug: 50
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_mustard_bottle --out-folder=sim/baked_data/pick_place_mb_wo_light_km_50 --task-name=pick_place --object-name=mustard_bottle --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=default --img-data-aug=2  --kinematic-aug=50  > logs/play_sim 2>&1 & 

#sim demo: 159472  kinematic aug: 50
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_mustard_bottle --out-folder=sim/baked_data/pick_place_mb_w_light_km_50 --task-name=pick_place --object-name=mustard_bottle --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=random --img-data-aug=2  --kinematic-aug=50  > logs/play_sim2 2>&1 & 

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_mb_w_light \
    --backbone-type=regnet_y_3_2gf \
    --sim-batch-size=32 \
    --lr=1e-5 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=4000 \
    --eval-start-epoch=300 \
    --eval-freq=100 > logs/train_sim_act_0 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_mb_w_light_km_50 \
    --backbone-type=regnet_y_3_2gf \
    --sim-batch-size=32 \
    --lr=1e-5 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=4000 \
    --eval-start-epoch=300 \
    --eval-freq=100 > logs/train_sim_act 2>&1 &

################################Tomato_soup_can##########################################
#sim demo: 36808  kinematic aug: 0
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_tomato_soup_can --out-folder=sim/baked_data/pick_place_tms_wo_light --task-name=pick_place --object-name=tomato_soup_can --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=default --img-data-aug=1  --kinematic-aug=0  > logs/play_sim 2>&1 & 

#sim demo: 72336  kinematic aug: 0
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_tomato_soup_can --out-folder=sim/baked_data/pick_place_tms_w_light --task-name=pick_place --object-name=tomato_soup_can --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=random --img-data-aug=2  --kinematic-aug=0  > logs/play_sim 2>&1 & 

#sim demo: 149858  kinematic aug: 50
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_tomato_soup_can --out-folder=sim/baked_data/pick_place_tms_wo_light_km_50 --task-name=pick_place --object-name=tomato_soup_can --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=default --img-data-aug=2  --kinematic-aug=50  > logs/play_sim 2>&1 & 

#sim demo: 146770  kinematic aug: 50
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_tomato_soup_can --out-folder=sim/baked_data/pick_place_tms_w_light_km_50 --task-name=pick_place --object-name=tomato_soup_can --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=random --img-data-aug=2  --kinematic-aug=50  > logs/play_sim2 2>&1 & 

#8 16 32
nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_tms_w_light \
    --backbone-type=regnet_y_3_2gf \
    --sim-batch-size=32 \
    --lr=1e-5 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=4000 \
    --eval-start-epoch=300 \
    --eval-freq=100 > logs/train_sim_act_0 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_tms_w_light_km_50 \
    --backbone-type=regnet_y_3_2gf \
    --sim-batch-size=32 \
    --lr=1e-5 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=4000 \
    --eval-start-epoch=300 \
    --eval-freq=100 > logs/train_sim_act 2>&1 &

################################Sugar Box##########################################
#sim demo: 31253 kinematic aug: 0
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_sugar_box --out-folder=sim/baked_data/pick_place_sg_wo_light --task-name=pick_place --object-name=sugar_box --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --img-data-aug=1  --chunk-size=50  > logs/play_sim0 2>&1 & 

#sim demo: 62506 kinematic aug: 0
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_sugar_box --out-folder=sim/baked_data/pick_place_sg_w_light --task-name=pick_place --object-name=sugar_box --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=random --img-data-aug=2  --kinematic-aug=0 > logs/play_sim 2>&1 & 

#sim demo: 136848 kinematic aug: 50
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_sugar_box --out-folder=sim/baked_data/pick_place_sg_wo_light_km_50 --task-name=pick_place --object-name=sugar_box --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=default  --img-data-aug=2 --kinematic-aug=50  > logs/play_sim1 2>&1 & 

#sim demo: 135076  kinematic aug: 50
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_sugar_box --out-folder=sim/baked_data/pick_place_sg_w_light_km_50 --task-name=pick_place --object-name=sugar_box --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=random  --img-data-aug=2 --kinematic-aug=50  > logs/play_sim2 2>&1 & 

#4 8 16 32 64 128
nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_sg_wo_light \
    --backbone-type=regnet_y_3_2gf \
    --sim-batch-size=32 \
    --lr=1e-5 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=3000 \
    --eval-start-epoch=300 \
    --eval-freq=100 > logs/train_sim_act 2>&1 &

nohup python main/train_act_adr.py \
    --sim-folder=sim/raw_data/pick_place_sugar_box \
    --sim-demo-folder=sim/baked_data/pick_place_sg_wo_light \
    --sim-aug-dataset-folder=sim/baked_data/pick_place_sg_aug_rank4 \
    --backbone-type=regnet_y_3_2gf \
    --sim-batch-size=128 \
    --lr=1e-5 \
    --kl_weight=200 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=600 \
    --randomness-rank=4 \
    --eval-freq=100 > logs/train_act_adr 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_sg_w_light \
    --backbone-type=regnet_y_3_2gf \
    --sim-batch-size=4 \
    --lr=1e-5 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-4 \
    --val-ratio=0.1 \
    --num-epochs=3000 \
    --eval-start-epoch=100 \
    --eval-freq=100 > logs/train_sim_act 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_sg_wo_light_km_50 \
    --backbone-type=regnet_y_3_2gf \
    --sim-batch-size=32 \
    --lr=1e-5 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=4000 \
    --eval-start-epoch=300 \
    --eval-freq=100 > logs/train_sim_act 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_sg_w_light_km_50 \
    --backbone-type=regnet_y_3_2gf \
    --sim-batch-size=32 \
    --lr=1e-5 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=4000 \
    --eval-start-epoch=300 \
    --eval-freq=100 > logs/train_sim_act 2>&1 &


The `eval-freq` argument specifies the frequency of evaluating and saving the model. which is 200 epochs in this case.

nohup python hand_teleop/player/player_augmentation.py \
      --task-name=pick_place \
      --object-name=mustard_bottle \
      --delta-ee-pose-bound=0.001 \
      --kinematic-aug=100 \
      --seed=20230914 \
      --frame-skip=1 > ./logs/play_aug0 2>&1 &

nohup python hand_teleop/player/player_augmentation.py \
      --task-name=pick_place \
      --object-name=sugar_box \
      --delta-ee-pose-bound=0.001 \
      --seed=20230914 \
      --frame-skip=1 > ./logs/play_aug2 2>&1 &

nohup python hand_teleop/player/player_augmentation.py \
      --task-name=pick_place \
      --object-name=tomato_soup_can \
      --delta-ee-pose-bound=0.001 \
      --seed=20230914 \
      --frame-skip=1 > ./logs/play_aug3 2>&1 &
      
```