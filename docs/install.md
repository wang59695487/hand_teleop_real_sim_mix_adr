# Installation Instruction

In this document, we provide instructions of how to properly install this codebase. We highly recommend using a conda environment to simplify set up.

## Setup Conda Environment

You can skip this section if you are not using conda virtual environment.

```
conda create -y -n teleop python=3.8
conda activate teleop
conda install -c conda-forge urllib3 importlib_metadata
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## CyberDemo Installation

```
git clone https://github.com/wang59695487/hand_teleop_real_sim_mix_adr
cd ./hand_teleop_real_sim_mix_adr
pip install -r requirements.txt
```

## Download the assets
We have provided assets in [Google Drive](https://drive.google.com/drive/folders/1P1M9YcTWtFJaOMBrNcJMCbqfmhQvoqEe?usp=drive_link)
Download the assets and unzip it to the root directory of the project.