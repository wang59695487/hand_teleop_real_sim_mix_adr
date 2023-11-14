[Experiment: Mix_Real_Sim Data](#user-content-experiment-simulated-data)

```bash

################################Mustard Bottle##########################################
#real demo: 58150   kinematic aug: 0
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --real-folder=real/raw_data/pick_place_mustard_bottle_large_scale --out-folder=real/baked_data/pick_place_mb_large_scale --task-name=pick_place --object-name=mustard_bottle --frame-skip=1 --real-delta-ee-pose-bound=0.001 --img-data-aug=1 --chunk-size=50 > logs/play_sim 2>&1 & 

nohup python main/train_act.py \
    --real-demo-folder=real/baked_data/pick_place_mb_large_scale \
    --sim-demo-folder=sim/baked_data/pick_place_mb_w_light_km_150 \
    --backbone-type=regnet_y_3_2gf \
    --real-batch-size=32 \
    --sim-batch-size=128 \
    --lr=1e-5 \
    --kl_weight=200 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=5000 \
    --eval-start-epoch=1000 \
    --eval-freq=100 > logs/train_real_act_0 2>&1 &

nohup python main/train_act.py \
    --real-demo-folder=real/baked_data/pick_place_mb_large_scale \
    --ckpt=trained_models/mb_w_light_km_150_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --real-batch-size=32 \
    --lr=1e-5 \
    --kl_weight=2000 \
    --num_queries=50 \
    --weight_decay=1e-4 \
    --val-ratio=0.1 \
    --num-epochs=10000 \
    --eval-start-epoch=5000 \
    --finetune \
    --eval-freq=100 > logs/finetune_real 2>&1 &


################################Tomato Soup Can##########################################
#real demo: 34365   kinematic aug: 0
python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=mvp --task-name=pick_place --object-name=tomato_soup_can --real-folder=real/raw_data/pick_place_tomato_soup_can --out-folder=real/baked_data/pick_place_tms --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001 --with-features=True 


nohup python main/train_act.py \
    --real-demo-folder=real/baked_data/pick_place_tms \
    --ckpt=trained_models/tms_w_light_km_50_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --real-batch-size=8 \
    --lr=1e-8 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=1000 \
    --eval-start-epoch=20 \
    --finetune \
    --eval-freq=100 > logs/finetune_real 2>&1 &

################################Sugar Box##########################################
#real demo: 33640   kinematic aug: 0
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --real-folder=real/raw_data/pick_place_sugar_box --out-folder=real/baked_data/pick_place_sg --task-name=pick_place --object-name=sugar_box --frame-skip=1 --real-delta-ee-pose-bound=0.001 --img-data-aug=1 --chunk-size=50 > logs/play_sim 2>&1 & 

nohup python main/train_act.py \
    --real-demo-folder=real/baked_data/pick_place_sg \
    --backbone-type=regnet_y_3_2gf \
    --real-batch-size=32 \
    --lr=1e-5 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=3500 \
    --eval-start-epoch=100 \
    --eval-freq=100 > logs/train_real_act_0 2>&1 &


nohup python main/train_act.py \
    --real-demo-folder=real/baked_data/pick_place_sg \
    --ckpt=trained_models/sg_wo_light_km_50_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --real-batch-size=32 \
    --lr=1e-7 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-4 \
    --val-ratio=0.1 \
    --num-epochs=4000 \
    --eval-start-epoch=100 \
    --finetune \
    --eval-freq=100 > logs/finetune_real 2>&1 &

################################pick_place##########################################
python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=pvr --real-folder=real/raw_data/pick_place_15 --out-folder=real/baked_data/pick_place_15 --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001 --with-features=True

python hand_teleop/player/play_multiple_demonstrations_act.py --real-folder=real/raw_data/pick_place_15 --out-folder=real/baked_data/pick_place_15 --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001

###########R3M, MVP, PVR #############
###R3M: 2048
###MVP: 768
###PVR: 2048


################################Pouring##########################################
python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --real-folder=real/raw_data/pouring --out-folder=real/baked_data/pouring --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001 

###########R3M, MVP, PVR #############
###R3M: 2048
###MVP: 768
###PVR: 2048


python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=r3m --real-folder=real/raw_data/pouring --out-folder=real/baked_data/pouring --frame-skip=1 --real-delta-ee-pose-bound=0.001 --img-data-aug=5 --chunk-size=50 --with-features=True

nohup python main/train_act.py \
    --real-demo-folder=real/baked_data/pouring \
    --backbone-type=regnet_y_3_2gf \
    --real-batch-size=128 \
    --lr=1e-5 \
    --kl_weight=200 \
    --num_queries=25 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=3500 \
    --eval-start-epoch=100 \
    --eval-freq=100 > logs/train_real_act_0 2>&1 &

################################dclaw##########################################
python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=r3m --task-name=dclaw --real-folder=real/raw_data/dclaw --out-folder=real/baked_data/dclaw --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001 --with-features=True 

python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=pvr --task-name=dclaw --real-folder=real/raw_data/dclaw --out-folder=real/baked_data/dclaw_pvr --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001 --with-features=True 

python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=mvp --task-name=dclaw --real-folder=real/raw_data/dclaw --out-folder=real/baked_data/dclaw_mvp --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001 --with-features=True 

python hand_teleop/player/play_multiple_demonstrations_act.py --task-name=dclaw --real-folder=real/raw_data/dclaw --out-folder=real/baked_data/dclaw --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001 

###########R3M, MVP, PVR #############
###R3M: 2048
###MVP: 768
###PVR: 2048
```
