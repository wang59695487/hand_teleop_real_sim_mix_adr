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
3. Use play_multiple_demonstrations_act.py to replay the simulation/Real demonstrations and process the data.
4. Train and augment the data using train_act_adr.py.

The following session only provides example script of our method. For baselines, checkout [baselines](docs/baseline.md).

## Step 1: Collect Human Demonstrations

how to use anyteleop server could see (https://yzqin.github.io/anyteleop/) for more details.


```
python main/teleop_hci_with_arm.py --task=pick_place --object=mustard_bottle --out-folder=YOUR_DATA_FOLDER
```

We have provided some simulation demos in [Google Drive](https://drive.google.com/drive/folders/1txUP17oU3quuq_xoxB3wltw0z_AtJaVu?usp=sharing).
Download the simulation data and extract it to the root directory of the project. The directory structure should be organized as follows:
sim
├── raw_data
│   └── task-name
└── baked_data

We have also provided some real world demos in [Google Drive](https://drive.google.com/drive/folders/1ZLyf-KEDTPlUaSYFSfGW80BqsqjAqMYA?usp=drive_link).
## Step 2: Test the Augmentation

To test the augmentation, run
```
python hand_teleop
/player/player_augmentation.py --sim-folder=sim/raw_data/pick_place_sugar_box --

```

## Step 3: Bake the Simulation Data

To bake the simulation data, run
```
python hand_teleop/player/play_multiple_demonstrations_act.py --sim-folder=sim/raw_data/pick_place_sugar_box --out-folder=sim/baked_data/pick_place_sg_wo_light --task-name=pick_place --object-name=sugar_box --frame-skip=1 --sim-delta-ee-pose-bound=0.001 --light-mode=default --img-data-aug=1  --kinematic-aug=0 --chunk-size=50
```

To bake the real data, run
```
python hand_teleop/player/play_multiple_demonstrations_act.py --real-folder=real/raw_data/pick_place_15 --out-folder=real/baked_data/pick_place_15 --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001
```
### Configuration Parameters

- **backbone-type:** Specify your vision model. For baseline options, you can choose from:
  - [R3M](https://sites.google.com/view/robot-r3m/)
  - [MVP](https://github.com/ir413/mvp)
  - [PVR](https://sites.google.com/view/pvr-control)
  - Moco, vit, clip
  refer to the function `generate_feature_extraction_model` in `models/vision/feature_extractor.py` for more details.
  Remember to specify the with-feature flag to True if you are using the pre-trained vision model.

- **with-feature:** store the feature of the data, default is False.

- **sim-folder:** The folder containing the raw data.

- **out-folder:** The folder for the baked data.

- **task-name:** The task name of the data (e.g., `pick_place`, `pour`, `dclaw` from our provided dataset).

- **object-name:** The object name of the data (e.g., `mustard_bottle`, `sugar_box`, `tomato_soup_can`, `diverse_objects` from our provided dataset).

- **frame-skip:** The frame skip value for the data.

- **sim-delta-ee-pose-bound:** The bound for the delta end-effector pose. Data points with a delta end-effector pose smaller than this bound will be skipped.

- **light-mode:** The light mode of the data (options: default, random from our provided dataset).

- **img-data-aug:** The image data augmentation factor. Use `1` for simulation data and `5` (5 times the original demos) for real data.

- **image_augmenter**: T.AugMix() or T.Compose() for image augmentation, mainly using for real data.

- **kinematic-aug:** The kinematic augmentation factor, applicable only to simulation data (e.g., `50` means +50 kinematic demos).

### Example Command

To run the baseline, use the following command:

```bash
python hand_teleop/player/play_multiple_demonstrations_act.py --backbone-type=mvp --task-name=dclaw --real-folder=real/raw_data/dclaw --out-folder=real/baked_data/dclaw_mvp --frame-skip=1 --img-data-aug=5 --chunk-size=50 --real-delta-ee-pose-bound=0.001 --with-features=True
```

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