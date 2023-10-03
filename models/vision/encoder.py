import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms as T
from torchvision.models.feature_extraction import create_feature_extractor
from feature_extractor import FeatureExtractor, ClassToken

from termcolor import colored
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT_DIM = {2: 39, 4: 35, 6: 31, 8: 27, 10: 23, 11: 21, 12: 19}

def tie_weights(src, trg):
	assert type(src) == type(trg)
	trg.weight = src.weight
	trg.bias = src.bias

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class StateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StateEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, input_state):
        encoded_state = self.model(input_state)
        return encoded_state

    def copy_fc_weights_from(self, source, n=None):
        """Tie first n linear layers"""
        if n is None:
            n = len(self.model)
        for i in range(0,2*n,2):
            if type(self.model[i]) == nn.Linear:
                tie_weights(src=source.model[i], trg=self.model[i]) 


class ResNetEncoder(nn.Module):
    def __init__(self, backbone):
        super(ResNetEncoder, self).__init__()
        modules = list(backbone.children())[:-1]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        if self.model.training:
            out = self.model(x)
        else:
            out = self.model(x).cpu().numpy()
        return out     

    def copy_conv_weights_from(self, source, n=None):
        """Tie first n layers (sequential modules for ResNet)"""
        if n is None or n==-1:
            n = len(self.model)
        assert n<=len(self.model) and n>0 , \
            f'invalid number of shared layers, received {n} layers'
        for i in range(n):
            self.model[i] = source.model[i]


class EmbeddingNet(nn.Module):
    """
    Input shape must be (N, H, W, 3), where N is the number of frames.
    The class will then take care of transforming and normalizing frames.
    The output shape will be (N, E), where E is the embedding size.
    """
    def __init__(self, embedding_name, in_channels=3, num_filters=32, num_layers=5, train=False, disable_cuda=False):
        super(EmbeddingNet, self).__init__()

        self.embedding_name = embedding_name
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.embedding, self.transforms = \
            _get_embedding(embedding_name=embedding_name, in_channels=in_channels, num_filters=num_filters, num_layers=num_layers)
        dummy_in = torch.zeros(1, in_channels, 224, 224)
        dummy_in = self.transforms(dummy_in)
        self.in_shape = dummy_in.shape[1:]
        dummy_out = self._forward(dummy_in)
        self.out_size = np.prod(dummy_out.shape)
        print(colored(f"Embedding dim: {self.out_size}", 'green'))

        if torch.cuda.is_available() and not disable_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.embedding = self.embedding.to(device=self.device)
        self.training = self.embedding.training

    def _forward(self, observation):
        out = self.embedding(observation)
        return out

    def forward(self, observation):
        # observation.shape -> (N, H, W, 3)
        observation = observation.to(device)
        observation = observation.transpose(1, 2).transpose(1, 3).contiguous()
        observation = self.transforms(observation)

        if self.embedding.training:
            out = self._forward(observation)
            if self.embedding_name != "flexible":
                return out.view(-1, self.out_size).squeeze()
            else:
                return out.view(-1, out.size(0)).squeeze()
        else:
            with torch.no_grad():
                out = self._forward(observation)
                if self.embedding_name != "flexible":
                    return out.view(-1, self.out_size).squeeze().cpu().numpy()
                else:
                    return out.view(-1, out.size(0)).squeeze().cpu().numpy()
            
    def copy_conv_weights_from(self, source, n=None):
        """Tie first n convolutional layers"""
        if n is None or n==-1:
            n = len(self.embedding)
        assert n<=len(self.embedding) and n>0 , \
            f'invalid number of shared layers, received {n} layers'
        for i in range(0,2*n,2):
            if type(self.embedding[i]) == nn.Conv2d:
                tie_weights(src=source.embedding[i], trg=self.embedding[i])

def _get_embedding(embedding_name, in_channels, num_filters, num_layers):
    """
    See https://pytorch.org/vision/stable/models.html

    Args:
        embedding_name (str, 'random'): the name of the convolution model,
        in_channels (int, 3): number of channels of the input image,
        pretrained (bool, True): if True, the model's weights will be downloaded
            from torchvision (if possible),
        train (bool, False): if True the model will be trained during learning,
            if False its parameters will not change.

    """

    # Default transforms: https://pytorch.org/vision/stable/models.html
    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    # where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then
    # normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    transforms = nn.Sequential(
        T.Resize(256, interpolation=3) if 'mae' in embedding_name else T.Resize(256),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )

    assert in_channels == 3, 'Current models accept 3-channel inputs only.'

    # FIXED 5-LAYER CONV
    if embedding_name == 'random':
        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        ## Original Model
        model = nn.Sequential(
            init_(nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
        )
    elif embedding_name == 'fixed':

        # model = nn.Sequential(
        #     init_(nn.Conv2d(in_channels, 32, kernel_size=(5,5), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        # )
        model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1),
            # nn.ReLU(),
        )

        # # DrQ like model
        # model = nn.Sequential(
        #     init_(nn.Conv2d(in_channels, 32, kernel_size=(7,7), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(5,5), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)),
        #     nn.ReLU(),
        # )
        # remove imgnet normalization and resize to 84x84
        transforms = nn.Sequential(
            T.Resize(84),
        )
    
    elif embedding_name == "flexible":
        modules = [nn.Conv2d(in_channels, num_filters, kernel_size=(3,3), stride=2)]
        for i in range(num_layers - 1):
            modules.append(nn.ReLU())
            modules.append(nn.Conv2d(num_filters, num_filters, kernel_size=(3,3), stride=2))
        model = nn.Sequential(*modules)
        transforms = nn.Sequential(
            T.Resize(84),
        )

    # if train:
    #     model.train()
    #     for p in model.parameters():
    #         p.requires_grad = True
    # else:
    #     model.eval()
    #     for p in model.parameters():
    #         p.requires_grad = False

    return model, transforms


class PixelEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_shared_layers=4):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_shared_layers = num_shared_layers

        self.convs = nn.ModuleList(
			[nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		)
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_conv(self, obs, detach=False):
        obs = self.preprocess(obs)
        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            if i == self.num_shared_layers-1 and detach:
                conv = conv.detach()

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs, detach)
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        out = torch.tanh(h_norm)

        return out

    def copy_conv_weights_from(self, source, n=None):
        """Tie n first convolutional layers"""
        if n is None:
            n = self.num_layers
        for i in range(n):
            tie_weights(src=source.convs[i], trg=self.convs[i])


def make_encoder(
	use_visual_backbone, backbone_type, embedding_name, in_channels, num_layers, num_filters):
    if not use_visual_backbone:
        # assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
        # if num_shared_layers == -1 or num_shared_layers == None:
        #     num_shared_layers = num_layers
        # assert num_shared_layers <= num_layers and num_shared_layers > 0, \
        #     f'invalid number of shared layers, received {num_shared_layers} layers'
        assert embedding_name != None
        model = EmbeddingNet(
            embedding_name=embedding_name, in_channels=in_channels, num_filters=num_filters, num_layers=num_layers
        )
    else:
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

            model = ResNetEncoder(backbone)

        elif "vit" in backbone_type:
            model = eval(f"models.{backbone_type}")(weights="IMAGENET1K_V1")
            model = nn.Sequential(
            create_feature_extractor(model, {"encoder": "img_feats"}),
            ClassToken(0)
            )
        else:
            model = eval(f"models.{backbone_type}")(weights="IMAGENET1K_V1")
            model = nn.Sequential(
            create_feature_extractor(model, {"avgpool": "img_feats"}),
            ClassToken("img_feats")
            )

        model.eval()

        
    
    dummy_in = torch.zeros(1, 224, 224, in_channels)
    with torch.no_grad():
        model.eval()
        if not isinstance(model, EmbeddingNet):
            dummy_in = torch.zeros(1, in_channels, 224, 224)
        dummy_out = model(dummy_in)
        model_out_size = np.prod(dummy_out.shape)
    model.train()
    return model, model_out_size