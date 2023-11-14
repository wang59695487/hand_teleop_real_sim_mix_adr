import cv2
import imageio
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2

real_image="/home/yigit/project/hand_teleop_real_sim_mix_adr/real/raw_data/dclaw_small_scale/0001/frame0000.png"
# rgb_pic = imageio.imread(real_image, pilmode="RGB")
# rgb_pic = rgb_pic.astype(np.uint8)
# img = np.moveaxis(rgb_pic,-1,0)[None, ...]
# # rgb_pic = cv2.copyMakeBorder(rgb_pic,80,80,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
# print(img.shape)
# imageio.imsave("./temp/demos/relocate-rgb.png", rgb_pic)

rgb_pic = torchvision.io.read_image(real_image, mode=torchvision.io.ImageReadMode.RGB)
rgb_pic = v2.Pad(padding=[0,80])(rgb_pic)
rgb_pic = v2.Resize(size=[224,224])(rgb_pic)
# rgb_pic = rgb_pic.permute(1,2,0)
# rgb_pic = rgb_pic.permute(2,0,1)[None, ...]
print(rgb_pic.shape)
torchvision.io.write_png(rgb_pic, "./temp/demos/relocate-rgb.png")

                    