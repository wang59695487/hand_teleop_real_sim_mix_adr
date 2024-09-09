# CyberDemo: Augmenting Simulated Human Demonstration for Real-World Dexterous Manipulation

<p align="center">
  <img src="docs/cyberdemo.avif" width="1000"/>
</p>

This repository contains a reference PyTorch implementation of the paper:

<b>CyberDemo: Augmenting Simulated Human Demonstration for Real-World Dexterous Manipulation</b> <br>
[Jun Wang*](https://wang59695487.github.io/),
[Yuzhe Qin*](https://yzqin.github.io/),
[Kaiming Kuang](https://kaimingkuang.github.io/),
[Yigit Korkmaz](https://ygtkorkmaz.github.io/),
[Akhilan Gurumoorthy](https://www.linkedin.com/in/akhilan-g/),
[Hao Su](https://cseweb.ucsd.edu/~haosu/index.html),
[Xiaolong Wang](https://xiaolonw.github.io/) <br>
[[Website](https://cyber-demo.github.io/)]
[[Paper](https://arxiv.org/pdf/2402.14795)]

## Installation

See [installation instructions](docs/install.md).

## Introduction

This paper provide a data augmentation and training method method for simulation demos to help Real-World Dexterous Manipulation.
1. Collect human demonstrations in simulation using any teleoperation method (usually 50 is sufficient) and gather 15 demonstrations in the real world for fine-tuning.
2. Use play_multiple_demonstrations_act.py to replay the simulation/Real demonstrations and process the data.
3. Use player_augmentation.py to verify the augmentation of the simulation demonstrations first.
4. Train and augment the simulation data using train_act_adr.py.
5. Fine-tune the model using the real-world demonstrations using train_real_act.py.

The following session only provides example script of our method. For baselines, checkout [baselines](docs/baseline.md).

## Step 1: Collect Human Demonstrations

how to use anyteleop server could see (https://yzqin.github.io/anyteleop/) for more details.


```
python main/teleop_hci_with_arm.py --task=pick_place --object=mustard_bottle --out-folder=YOUR_DATA_FOLDER
```

We have provided some simulation demos in [Google Drive](https://drive.google.com/drive/folders/1txUP17oU3quuq_xoxB3wltw0z_AtJaVu?usp=sharing).
Download the simulation data and extract it to the root directory of the project. The directory structure should be organized as follows:
  
```
sim
├── raw_data
│   └── task-name
└── baked_data
    └── task-name
```


We have also provided some real world demos in [Google Drive](https://drive.google.com/drive/folders/1ZLyf-KEDTPlUaSYFSfGW80BqsqjAqMYA?usp=drive_link).

## Step 2: Bake the Simulation Data

To bake the simulation data, run
```
python hand_teleop/player/play_multiple_demonstrations_act.py --sim-folder=sim/raw_data/pick_place_mustard_bottle --out-folder=sim/baked_data/pick_place_mustard_bottle_multi_view --task-name=pick_place_multi_view --frame-skip=1 --img-data-aug=1  --chunk-size=50

python hand_teleop/player/play_multiple_demonstrations_act.py --sim-folder=sim/raw_data/pick_place_sugar_box --out-folder=sim/baked_data/pick_place_sg_wo_light --task-name=pick_place --object-name=sugar_box --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=default --img-data-aug=1  --kinematic-aug=0 --chunk-size=50
```

To bake the real data, run
```
python hand_teleop/player/play_multiple_demonstrations_act.py --real-folder=real/raw_data/pick_place_15 --out-folder=real/baked_data/pick_place_15 --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001
```
### Configuration Parameters

- **with-feature:** store the feature of the data, default is False. Used with the backbone-type.

- **backbone-type:** Specify your vision model and Remember to specify the with-feature flag to True for using the feature.
  - For baseline options, you can choose from:
    - [R3M](https://sites.google.com/view/robot-r3m/)
    - [MVP](https://github.com/ir413/mvp)
    - [PVR](https://sites.google.com/view/pvr-control)
  - Moco, vit, clip \\
  - refer to the function `generate_feature_extraction_model` in `models/vision/feature_extractor.py` for more details.

- **sim-folder:** The folder containing the raw data.

- **out-folder:** The folder for the baked data.

- **task-name:** The task name of the data (e.g., `pick_place`, `pour`, `dclaw` from our provided dataset). The tag {task_name}_multi_view, such as pick_place_multi_view, is used for multi-view augmentation.

- **object-name:** The object name of the data (e.g., `mustard_bottle`, `sugar_box`, `tomato_soup_can`, `diverse_objects` from our provided dataset).

- **frame-skip:** The frame skip value for the data.

- **sim-delta-ee-pose-bound:** The bound for the delta end-effector pose. Data points with a delta end-effector pose smaller than this bound will be skipped.

- **light-mode:** The light mode of the data (options: default, random from our provided dataset).

- **img-data-aug:** The image data augmentation factor. Use `1` for simulation data and `5` (5 times the original demos) for real data.

- **image_augmenter**: T.AugMix() or T.Compose() for image augmentation, mainly using for real data.

- **kinematic-aug:** The kinematic augmentation factor, applicable only to simulation data (e.g., `50` means +50 kinematic demos).

- **sensitivity-check:** Specify whether to check and record the sensitivity of the data. Note that this process may be slow. The results will be stored in the specified `out-folder`.

### Example Command

To run the baseline, use the following command:

```bash
python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=mvp --task-name=dclaw --real-folder=real/raw_data/dclaw --out-folder=real/baked_data/dclaw_mvp --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001 --with-features=True
```

## Step 3: Test the Augmentation

To test the augmentation, run
```
python hand_teleop/player/player_augmentation.py \
      --sim-demo-folder=sim/raw_data/pick_place_tomato_soup_can \
      --task-name=pick_place \
      --object-name=tomato_soup_can \
      --delta-ee-pose-bound=0.001 \
      --seed=20230914 \
      --frame-skip=1 \
      --randomness-rank=1 \
      --save-video=True
```
### Configuration Parameters
- **sim-demo-folder**: The folder containing the raw simulation data.
- **task-name:** The task name of the data (e.g., `pick_place`, `pour`, `dclaw` from our provided dataset). The tag {task_name}_multi_view, such as pick_place_multi_view, is used for multi-view augmentation.
- **object-name:** The object name of the data (e.g., `mustard_bottle`, `sugar_box`, `tomato_soup_can`, `diverse_objects` from our provided dataset).
- **delta-ee-pose-bound:** The bound for the delta end-effector pose. Data points with a delta end-effector pose smaller than this bound will be skipped.
- **seed:** The random seed for the data augmentation.
- **frame-skip:** The frame skip value for the data.
- **randomness-rank:** The randomness rank for the data augmentation.
- **sensitivity-check:** Specify whether to use sensitivity-aware augmentation. Note that you need to have the sensitivity data stored in the `sim_dataset_folder` to use this option (see Step 2).
- **sim-dataset-folder:** The folder containing the baked data.
- **save-video:** Whether to save the video of the augmented data.



## Step 4: Train the Model with curriculum learning

To train the model in simulation, run
```
nohup python main/train_act_adr.py \
    --task-name=pick_place \
    --object-name=mustard_bottle \
    --sim-demo-folder=sim/raw_data/pick_place_mustard_bottle \
    --sim-dataset-folder=sim/baked_data/pick_place_mustard_bottle \
    --sim-aug-dataset-folder=sim/baked_data/pick_place_mustard_bottle_rank4 \
    --sim-batch-size=128 \
    --lr=1e-5 \
    --kl_weight=200 \
    --weight_decay=1e-2 \
    --val-ratio=0.1 \
    --num-epochs=500 \
    --randomness-rank=4 \
    --eval-freq=100 > logs/train_act_adr 2>&1 &
```
### Configuration Parameters
- **task-name:** The task name of the data (e.g., `pick_place`, `pour`, `dclaw` from our provided dataset). The tag {task_name}_multi_view, such as pick_place_multi_view, is used for multi-view augmentation.
- **object-name:** The object name of the data (e.g., `mustard_bottle`, `sugar_box`, `tomato_soup_can`, `diverse_objects` from our provided dataset).
- **sim-demo-folder:** The folder containing the raw simulation data.
- **sim-dataset-folder:** The folder containing the baked data.
- **sim-aug-dataset-folder:** The folder to store the augmented data generated during training.
- **sim-batch-size:** The batch size for the simulation data.
- **lr:** The learning rate for the training.
- **kl_weight:** The KL divergence weight for the training.
- **weight_decay:** The weight decay for the training.
- **val-ratio:** The ratio of the validation data.
- **num-epochs:** The number of epochs for the training.
- **randomness-rank:** The randomness rank for the data augmentation.
- **eval-freq:** The frequency of evaluating and saving the model.
- **sensitivity-check:** Specify whether to check and record the sensitivity of the data. Note that this process may be slow. The results will be stored in the baked simulation data.(see Step 2)

## Step 5: Fine-tune the Model with Real Demonstrations
To fine-tune the model with real demonstrations, run
```
python main/train_act.py \
    --real-demo-folder=real/baked_data/pick_place_sg \
    --ckpt=sim/baked_data/pick_place_rank4/epoch_best.pt \
    --real-batch-size=32 \
    --lr=1e-7 \
    --kl_weight=20 \
    --num_queries=50 \
    --weight_decay=1e-4 \
    --val-ratio=0.1 \
    --num-epochs=4000 \
    --eval-start-epoch=100 \
    --finetune \
    --eval-freq=100
```
### Configuration Parameters
- **real-demo-folder:** The folder containing the baked real-world data.(see Step 2)
- **ckpt:** The checkpoint of the model to be fine-tuned.
- **real-batch-size:** The batch size for the real data.
- **lr:** The learning rate for the fine-tuning.
- **kl_weight:** The KL divergence weight for the fine-tuning.
- **num_queries:** The number of queries for the fine-tuning.
- **weight_decay:** The weight decay for the fine-tuning.
- **val-ratio:** The ratio of the validation data.
- **num-epochs:** The number of epochs for the fine-tuning.
- **eval-start-epoch:** The epoch to start evaluating the model.
- **finetune:** Specify whether to fine-tune the model.
- **eval-freq:** The frequency of evaluating and saving the model.

## Citing

```
@inproceedings{wang2024cyberdemo,
        title={CyberDemo: Augmenting Simulated Human Demonstration for Real-World Dexterous Manipulation},
        author={Wang, Jun and Qin, Yuzhe and Kuang, Kaiming and Korkmaz, Yigit and Gurumoorthy, Akhilan and Su, Hao and Wang, Xiaolong},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={17952--17963},
        year={2024}
      }
```