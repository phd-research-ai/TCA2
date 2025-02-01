import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import clip
# from pixel2style2pixel.models.psp import pSp
from argparse import Namespace
from utils import normalize
from stylegan.model import Generator
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
import numpy as np
from models.stylegan2_generator import PixelNormLayer ,DenseBlock1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mapper(nn.Module):
    def __init__(self,  latent_dim=512):
        super(Mapper, self).__init__()
        
        layers = [PixelNormLayer()]
        for i in range(4):
            layers.append(
                DenseBlock1(latent_dim,latent_dim)
            )
        self.mapping = nn.Sequential(*layers)
    def forward(self, x):
        x = self.mapping(x)
        return x
    
class LevelsMapper(nn.Module):
    def __init__(self):
        super(LevelsMapper, self).__init__()
        self.course_mapping = Mapper()
        self.medium_mapping = Mapper()
        self.fine_mapping = Mapper()
    def forward(self, x):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]
        x_coarse = self.course_mapping(x_coarse)
        x_medium = self.medium_mapping(x_medium)
        x_fine = self.fine_mapping(x_fine)
        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)
        return out

def get_stylegan_generator(cfg):
    generator=Generator(1024, 512, 8)
    checkpoint = torch.load(cfg.paths.stylegan, map_location=device)
    generator.load_state_dict(checkpoint['g_ema'])
    generator.to(device)
    generator.eval()
    return generator


class LabelEncoder(nn.Module):
    def __init__(self, nf=512):
        super(LabelEncoder, self).__init__()
        self.nf = nf
        curr_dim = nf
        self.size = 2

        self.fc = nn.Sequential(
            # nn.Linear(512, 512), nn.ReLU(True), 
            nn.Linear(self.nf, curr_dim * self.size * self.size), nn.ReLU(True))

        transform = []
        for i in range(4):
            transform += [
                nn.ConvTranspose2d(curr_dim,
                                   curr_dim // 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                # nn.Upsample(scale_factor=(2, 2)),
                # nn.Conv2d(curr_dim, curr_dim//2, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=False),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2

        transform += [
            nn.Conv2d(curr_dim,
                      3,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False)
        ]
        self.transform = nn.Sequential(*transform)

    def forward(self,  label_feature):
        label_feature = self.fc(label_feature)
        label_feature = label_feature.view(label_feature.size(0), self.nf, self.size, self.size)
        label_feature = self.transform(label_feature)

        # mixed_feature = label_feature + image
        # mixed_feature = torch.cat((label_feature, image), dim=1)
        return label_feature
    
class TargetedGanAttack(nn.Module):
    def __init__(self, prompt):
        super().__init__()
        self.prompt=prompt
        self.mapper=LevelsMapper()
        encoder_lis = nn.Sequential(
            nn.Conv1d(19,
                      18,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.Tanh())
        self.encoder=nn.Sequential(*encoder_lis)
        


    def forward(self, detailcode,label):
        detailcode=detailcode.squeeze()
        batch_size=detailcode.shape[0]
        prompt=self.prompt[:,:205]
        prompt=prompt.repeat(batch_size,1,1)
        label=label.unsqueeze(1)
        label=nn.Softmax(label)
        label_prompt=torch.cat([label,prompt],dim=2)
        # print(label_prompt.shape)
        x_prompt=torch.cat([detailcode,label_prompt],dim=1)
        # print(x_prompt.shape)
        # print(self.encoder)
        mixed_feature=self.encoder(x_prompt)
        decoded_feature=mixed_feature*0.1 + detailcode
        detail=self.mapper(decoded_feature)
        return detail

    


    
class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, _ = clip.load("RN50", device="cuda")
        self.model.eval()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        # self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1,3,1,1)
        # self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1,3,1,1)

    def forward(self, image, text):
        # image=normalize(image)
        image = self.face_pool(image)
        text=text.repeat(image.shape[0],1)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity
    

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm1d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(dim_in,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm1d(dim_out,
                                                     affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_out,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm1d(dim_out,
                                                     affine=use_affine))

    def forward(self, x):
        return x + self.main(x)

    
@hydra.main(version_base=None, config_path="./config", config_name="config")
def test(cfg):
    prompt=torch.randn([1,1024]).to(device)
    model=TargetedGanAttack(prompt).to(device)
    label=torch.ones([2,1,307]).to(device)
    # label=torch.nn.functional.one_hot(data,num_classes=307).to(torch.float)
    detailcode=torch.ones([2,18,512]).to(device)
    result_images=model(detailcode,label)
    print(result_images.shape)
    # print(x.shape)
if __name__ == "__main__":
    test()
    