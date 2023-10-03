import clip
import torch
from torch import optim, nn
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as T

from tqdm import tqdm
from skimage.transform import rotate, AffineTransform, warp
import numpy as np
import pickle
import os
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor(nn.Module):
  def __init__(self, backbone):
    super(FeatureExtractor, self).__init__()
    modules = list(backbone.children())[:-1]
    self.model = nn.Sequential(*modules)

  def forward(self, x):
    out = self.model(x)
    return out     


class ImageEncoder(nn.Module):
  def __init__(self):
    super(ImageEncoder, self).__init__()
    self.encoder = torch.nn.Sequential(
        torch.nn.Linear(2048, 1536),
        torch.nn.ReLU(),
        torch.nn.Linear(1536, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024),
    ) 
    self.decoder = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1536),
        torch.nn.ReLU(),
        torch.nn.Linear(1536, 2048)
    )

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def encode(self,x):
    encoded = self.encoder(x)
    return encoded


class StateDecoder(nn.Module):
  def __init__(self):
    super(StateDecoder, self).__init__()
    self.decoder = torch.nn.Sequential(
        torch.nn.Linear(58, 116),
        torch.nn.ReLU(),
        torch.nn.Linear(116, 232),
        torch.nn.ReLU(),
        torch.nn.Linear(232, 564)
    )
    self.encoder = torch.nn.Sequential(
        torch.nn.Linear(564, 232),
        torch.nn.ReLU(),
        torch.nn.Linear(232, 116),
        torch.nn.ReLU(),
        torch.nn.Linear(116, 58)
    )

  def forward(self, x):
    decoded = self.decoder(x)
    encoded = self.encoder(decoded)
    return encoded

  def decode(self,x):
    decoded = self.decoder(x)
    return decoded


def train_image_encoder(features):
  for model_name in os.listdir('./trained_models'):
    if model_name == 'trained_image_encoder.pt':
      image_encoder = ImageEncoder()
      best_checkpoint = torch.load('./trained_models/{}'.format(model_name))
      image_encoder.load_state_dict(best_checkpoint['model'])
      image_encoder = image_encoder.to(device)
      image_encoder.eval()
      return image_encoder

  obs = torch.from_numpy(np.array(features, dtype=np.float32))
  obs = obs.to(device)
  targets = torch.from_numpy(np.array(features, dtype=np.float32))
  targets = targets.to(device)
 
  epochs = 1000
  learning_rate = 0.01
  image_encoder = ImageEncoder()
  params = list(image_encoder.parameters())
  criterion = nn.MSELoss()
  optimizer = optim.AdamW(params=params, lr=learning_rate, weight_decay=0.001)
  image_encoder = image_encoder.to(device).float()
  criterion = criterion.to(device)
  norm_avg = np.linalg.norm(np.array(features), axis=1).mean()
  for t in range(epochs):
      losses = 0
      image_encoder.train()
      optimizer.zero_grad()
      outputs = image_encoder(obs)
      loss = criterion(outputs, targets)
      losses += loss.item()
      loss.backward()
      optimizer.step()
      print("Epoch: {}    Loss: {}".format(t, losses/(norm_avg*len(features))))
  print("Image Encoder Training Complete")
  model_dict = image_encoder.state_dict()
  state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
  model_path = './trained_models/trained_image_encoder.pt'
  torch.save(state_dict, model_path)

  return image_encoder


def train_state_decoder(robot_states):
  for model_name in os.listdir('./trained_models'):
    if model_name == 'trained_state_decoder.pt':
      state_decoder = StateDecoder()
      best_checkpoint = torch.load('./trained_models/{}'.format(model_name))
      state_decoder.load_state_dict(best_checkpoint['model'])
      state_decoder = state_decoder.to(device)
      state_decoder.eval()
      return state_decoder

  obs = torch.from_numpy(np.array(robot_states, dtype=np.float32))
  obs = obs.to(device)
  targets = torch.from_numpy(np.array(robot_states, dtype=np.float32))
  targets = targets.to(device)

  epochs = 750
  learning_rate = 0.001
  state_decoder = StateDecoder()
  params = list(state_decoder.parameters())
  criterion = nn.MSELoss()
  optimizer = optim.AdamW(params=params, lr=learning_rate, weight_decay=0.01)
  state_decoder = state_decoder.to(device).float()
  criterion = criterion.to(device)
  norm_avg = np.linalg.norm(np.array(robot_states), axis=1).mean()
  for t in range(epochs):
      losses = 0
      state_decoder.train()
      optimizer.zero_grad()
      outputs = state_decoder(obs)
      loss = criterion(outputs, targets)
      losses += loss.item()
      loss.backward()
      optimizer.step()
      print("Epoch: {}    Loss: {}".format(t, losses/(norm_avg*len(robot_states))))
  print("State Decoder Training Complete")

  model_dict = state_decoder.state_dict()
  state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
  model_path = './trained_models/trained_state_decoder.pt'
  torch.save(state_dict, model_path)

  return state_decoder


class ClassToken(nn.Module):

    def __init__(self, cls_token_idx=0):
        super().__init__()
        self.cls_token_idx = cls_token_idx
    
    def forward(self, tokens):
      if self.cls_token_idx == 0:
        img_feats = tokens["img_feats"][:, self.cls_token_idx][..., None, None]
        return img_feats
      else:
        return tokens["img_feats"]


class CLIPWrapper(nn.Module):

  def __init__(self, clip_model):
    pass


def generate_feature_extraction_model(backbone_type):
  #################Using data augmentation#################
  if "MoCo" in backbone_type:
    if backbone_type == "MoCo18":
      backbone = models.resnet18()
      checkpoint = torch.load('trained_models/mocov2_r18_imgnet_e100.pt')
    elif backbone_type == "MoCo50":
      backbone = models.resnet50()
      checkpoint = torch.load('trained_models/moco_v2_800ep_pretrain.pt')
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = backbone.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}  

    model = FeatureExtractor(backbone)
  elif "vit" in backbone_type:
    model = eval(f"models.{backbone_type}")(weights="IMAGENET1K_V1")
    model = nn.Sequential(
      create_feature_extractor(model, {"encoder": "img_feats"}),
      ClassToken(0)
    )
  elif "clip" in backbone_type:
    model, preprocess = clip.load(backbone_type.replace("clip_", ""), device=device)
    model = model.visual.float()
    preprocess = transforms.Compose([
      # preprocess.transforms[0],
      # preprocess.transforms[1],
      preprocess.transforms[4],
    ])
  else:
    model = eval(f"models.{backbone_type}")(weights="IMAGENET1K_V1")
    model = nn.Sequential(
      create_feature_extractor(model, {"avgpool": "img_feats"}),
      ClassToken("img_feats")
    )

  if "clip" not in backbone_type:

    preprocess = transforms.Compose([
      transforms.Resize((224, 224)),  # resize to 224*224
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

  model.eval()
  return model, preprocess

def augmentation_img(img,augmenter):

  img = img * 255.0  # case [0, 1]
  img = torch.clip(img, 0.0, 255.0)
  img = img.type(torch.uint8)
  img = augmenter(img)
  img = (img/255.0).type(torch.float32)

  return img

def concate_features(features, stack_robot_qpos, stack_frames = True):
  
  concatenated_obs = []
  concatenated_robot_qpos = []
  if stack_frames:
    for i in range(len(features)):
      if i == 0:
        concatenated_obs.append(np.concatenate((features[i],features[i],features[i],features[i])))
        concatenated_robot_qpos.append(np.concatenate((stack_robot_qpos[i],stack_robot_qpos[i],stack_robot_qpos[i],stack_robot_qpos[i])))
      elif i == 1:
        concatenated_obs.append(np.concatenate((features[i-1],features[i],features[i],features[i])))
        concatenated_robot_qpos.append(np.concatenate((stack_robot_qpos[i-1],stack_robot_qpos[i],stack_robot_qpos[i],stack_robot_qpos[i])))
      elif i == 2:
        concatenated_obs.append(np.concatenate((features[i-2],features[i-1],features[i],features[i])))
        concatenated_robot_qpos.append(np.concatenate((stack_robot_qpos[i-2],stack_robot_qpos[i-1],stack_robot_qpos[i],stack_robot_qpos[i])))
      else:
        concatenated_obs.append(np.concatenate((features[i-3],features[i-2],features[i-1],features[i])))
        concatenated_robot_qpos.append(np.concatenate((stack_robot_qpos[i-3],stack_robot_qpos[i-2],stack_robot_qpos[i-1],stack_robot_qpos[i])))
  # else:
  #   for feature, robot_state in zip(features, robot_states):
  #     concatenated_obs.append(np.concatenate((feature,robot_state)))

  return concatenated_obs, concatenated_robot_qpos

def generate_features(visual_baked, backbone_type="ResNet34", num_data_aug=5, augmenter = T.AugMix(), using_features=False):
  
  raw_imgs = []
  robot_states = []
  visual_dataset = dict(obs = [], action = [], robot_qpos = [])

  target_actions = []
  stack_robot_qpos = []
  
  if using_features:
    model, preprocess = generate_feature_extraction_model(backbone_type)
    model.eval()
    model.to(device)
  
  for i in range(len(visual_baked["action"])):
    act = visual_baked["action"][i]
    obs = visual_baked["obs"][i]
    qpos = visual_baked["robot_qpos"][i]
    raw_imgs.append(obs["relocate_view-rgb"])
    robot_states.append(obs["state"])
    target_actions.append(act)
    stack_robot_qpos.append(qpos)

  aug_concat_robot_qpos = []
  aug_concat_obs = []
  aug_concat_action = []
  for i in tqdm(range(num_data_aug)):
    features = []
    for img in tqdm(raw_imgs):
      #img = torch.from_numpy(np.moveaxis(img,-1,0)[None, ...])
      img = torch.moveaxis(img,-1,0)[None, ...]
      if i != 0:
        img = augmentation_img(img,augmenter)
      if using_features:
        img = preprocess(img)
        img = img.to(device)
        #print(img.size())
        with torch.no_grad():
            feature = model(img)
        features.append(feature.cpu().detach().numpy().reshape(-1))
      else:
        img = img.cpu().detach().numpy()
        features.append(img)

    concatenated_obs = features
    concatenated_robot_qpos = stack_robot_qpos
      
    aug_concat_robot_qpos.extend(concatenated_robot_qpos)
    aug_concat_obs.extend(concatenated_obs)
    aug_concat_action.extend(target_actions)

  visual_dataset["action"] = aug_concat_action
  visual_dataset["robot_qpos"] = aug_concat_robot_qpos
  visual_dataset["obs"] = aug_concat_obs

  return visual_dataset

if __name__ == "__main__":
  baked_demo_path = "./test_visual_baked.pickle"
  with open(baked_demo_path,'rb') as f:
    visual_baked = pickle.load(f)

  visual_dataset = generate_features(visual_baked, backbone_type="ResNet34", stack_frames=True, encode=True)
  with open('./test_baked_features.pickle', "wb") as f:
    pickle.dump(visual_dataset, f)