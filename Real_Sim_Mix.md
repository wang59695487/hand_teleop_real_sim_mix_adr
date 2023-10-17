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
nohup python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --real-folder=real/raw_data/pick_place_tomato_soup_can --out-folder=real/baked_data/pick_place_tms --task-name=pick_place --object-name=tomato_soup_can --frame-skip=1 --real-delta-ee-pose-bound=0.001 --light-mode=default --img-data-aug=1  --kinematic-aug=0  > logs/play_sim 2>&1 & 


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
    --ckpt=trained_models/sg_w_light_km_50_sim/epoch_best.pt \
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
```