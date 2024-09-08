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
2. Use player_augmentation.py to verify the augmentation of the simulation demonstrations first.
3. Use play_multiple_demonstrations_act.py to replay the simulation demonstrations and process the data.
4. Train and augment the data using train_act_adr.py.

The following session only provides example script of our method. For baselines, checkout [baselines](docs/baseline.md).

## Step 1: Collect Human Demonstrations

how to use anyteleop server could see (https://yzqin.github.io/anyteleop/) for more details.


```
python main/teleop_hci_with_arm.py --task=pick_place --object=mustard_bottle --out-folder=YOUR_DATA_FOLDER
```

We have provided some simulation demos in [Google Drive](https://drive.google.com/drive/folders/1txUP17oU3quuq_xoxB3wltw0z_AtJaVu?usp=sharing).
Download the simulation data and extract it to the root directory of the project. The directory structure should be organized as follows:
\begin{verbatim}
sim
├── raw_data
│   └── task-name
└── baked_data
\end{verbatim}

## Step 1: Bake the Simulation Data

To bake the simulation data, run
```
python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=regnet_y_3_2gf --sim-folder=sim/raw_data/pick_place_sugar_box --out-folder=sim/baked_data/pick_place_sg_wo_light --task-name=pick_place --object-name=sugar_box --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=default --img-data-aug=1  --kinematic-aug=0 --chunk-size=50
```

To bake the real data, run
```
python hand_teleop/player/play_multiple_demonstrations_act.py --real-folder=real/raw_data/pick_place_15 --out-folder=real/baked_data/pick_place_15 --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001
```
### Configuration Parameters

- **backbone-type:** Specify your vision model. In this case, we use `regnet_y_3_2gf`. For baseline options, you can choose from:
  - [R3M](https://sites.google.com/view/robot-r3m/)
  - [MVP](https://github.com/ir413/mvp)
  - [PVR](https://sites.google.com/view/pvr-control)

- **sim-folder:** The folder containing the raw data.

- **out-folder:** The folder for the baked data.

- **task-name:** The task name of the data (e.g., `pick_place`, `pour`, `dclaw` from our provided dataset).

- **object-name:** The object name of the data (e.g., `mustard_bottle`, `sugar_box`, `tomato_soup_can`, `diverse_objects` from our provided dataset).

- **frame-skip:** The frame skip value for the data.

- **sim-delta-ee-pose-bound:** The bound for the delta end-effector pose. Data points with a delta end-effector pose smaller than this bound will be skipped.

- **light-mode:** The light mode of the data (options: default, random from our provided dataset).

- **img-data-aug:** The image data augmentation factor. Use `1` for simulation data and `5` (5 times the original demos) for real data.

- **kinematic-aug:** The kinematic augmentation factor, applicable only to simulation data (e.g., `50` means +50 kinematic demos).

### Example Command

To run the baseline, use the following command:

```bash
python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=mvp --task-name=dclaw --real-folder=real/raw_data/dclaw --out-folder=real/baked_data/dclaw_mvp --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001 --with-features=True
- **with-feature:** store the feature of the data.
```

## Step 2: Student Policy Pretraining

In this section, we train a proprioceptive student policy by distilling from our trained oracle policy $f$.

Note we use the teacher rollout to train student policy, in contrast to DAgger in previous works.

```
scripts/train_student_sim.sh train.ppo.is_demon=True train.demon_path=ORACLE_CHECKPOINT_PATH 
```
We have provided a reference teacher checkpoint in [Google Drive](https://drive.google.com/file/d/1LCRFE6lvKSUDPpUfEATOmpDUPDbB7n8d/view?usp=sharing).

## Step 3: Open-Loop Replay in Real Hardware

To generate open-loop replay data for the student policy $\pi$, run
```
python real/robot_controller/teacher_replay.py --data-collect --exp=0 --replay_data_dir=REPLAY_DATA_DIR
```
where `REPLAY_DATA_DIR` is the directory to save the replay data.

Then process the replay data.

## Step 4: Real-world Fine-tuning

To fine-tune the student policy $\pi$ using real data, run
```
scripts/finetune_ppo.sh --real-dataset-folder=REAL_DATA_PATH --checkpoint-path=YOUR_CHECKPOINTPATH
```

## Real Data Download
Please download the real reference data from [Google Drive](https://drive.google.com/drive/folders/1TAMAvqLp3b5vEmdyrdcgW0kBW1GAxoyy?usp=sharing).
```
Real data:
  real_data.h5 is in the format of h5 file, which contains the following keys:
  -replay_demon_{idx}: the idx-th replay demonstration data
    - qpos: the current qpos of the robot
    - action: the delta action applied to the robot
    - current_target_qpos: the target qpos of the robot

  real_data_full.h5 is a full version of real_data.h5, which contains the following keys:
  -replay_demon_{idx}: the idx-th replay demonstration data
    - qpos: the current qpos of the robot
    - action: the delta action applied to the robot
    - current_target_qpos: the target qpos of the robot
    - rgb_ori: the original rgb image
    - rgb_c2d: the rgb image after camera2depth image processing
    - depth: the depth image
    - pc: the point cloud
    - obj_ends: the position of object ends 
```

## Acknowledgement

Note: This repository is built based on [Hora](https://github.com/HaozhiQi/hora) and [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs).

## Citing

If you find **PenSpin** or this codebase helpful in your research, please consider citing:

```
@article{wang2024penspin,
  author={Wang, Jun and Yuan, Ying and Che, Haichuan and Qi, Haozhi and Ma, Yi and Malik, Jitendra and Wang, Xiaolong},
  title={Lessons from Learning to Spin “Pens”},
  journal={arXiv:2405.07391},
  year={2024}
}
```