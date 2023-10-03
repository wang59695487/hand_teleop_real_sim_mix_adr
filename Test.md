
[Experiment: Simulated Data](#user-content-experiment-simulated-data)
nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_sg_wo_light_test_act\
    --ckpt=trained_models/sg_wo_light_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=1 \
    --eval-only > logs/eval_sim_sg 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_tms_wo_light\
    --ckpt=trained_models/tms_wo_light_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=3 \
    --eval-only > logs/eval_sim_tms 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_mb_wo_light\
    --ckpt=trained_models/mb_wo_light_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=3 \
    --eval-only > logs/eval_sim_mb 2>&1 &




nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_tms_wo_light\
    --ckpt=trained_models/tms_wo_light_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=4 \
    --eval-only > logs/eval_sim_mb1 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_tms_wo_light\
    --ckpt=trained_models/tms_w_light_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=4 \
    --eval-only > logs/eval_sim_mb2 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_tms_wo_light\
    --ckpt=trained_models/tms_wo_light_km_50_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=4 \
    --eval-only > logs/eval_sim_mb3 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_tms_wo_light\
    --ckpt=trained_models/tms_w_light_km_50_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=4 \
    --eval-only > logs/eval_sim_mb4 2>&1 &

##########################################################################################
nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_sg_wo_light_test_act \
    --ckpt=trained_models/sg_w_light_km_50_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=5 \
    --eval-only > logs/eval_sim_sg1 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_sg_wo_light_test_act \
    --ckpt=trained_models/sg_w_light_km_50_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=6 \
    --eval-only > logs/eval_sim_sg2 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_sg_wo_light_test_act \
    --ckpt=trained_models/sg_w_light_km_50_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=3 \
    --eval-only > logs/eval_sim_sg3 2>&1 &

nohup python main/train_act.py \
    --sim-demo-folder=sim/baked_data/pick_place_sg_wo_light_test_act \
    --ckpt=trained_models/sg_w_light_km_50_sim/epoch_best.pt \
    --backbone-type=regnet_y_3_2gf \
    --randomness-rank=4 \
    --eval-only > logs/eval_sim_sg4 2>&1 &
