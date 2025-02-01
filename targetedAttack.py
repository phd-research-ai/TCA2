import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from omegaconf import DictConfig, OmegaConf
from data.dataset import get_dataset,get_adv_dataset
from utils import get_model,set_requires_grad,unnormalize
from stylegan.model import Generator
from model import CLIPLoss,TargetedGanAttack
import lpips
from prompt import get_prompt
import torch.nn.functional as F
import time

from torchmetrics.regression import CosineSimilarity
from models import build_generator
from torchvision import  transforms
import numpy as np

import hydra

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



def get_stylegan_generator(cfg):
    generator=build_generator(gan_type='stylegan2',resolution=1024)
    checkpoint = torch.load(cfg.paths.stylegan2, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.to(device)
    generator.eval()
    return generator



@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(0)
    train_dataloader,test_dataloader,train_dataset,test_dataset=get_adv_dataset(cfg)

    classifier=get_model(cfg)
    generator=get_stylegan_generator(cfg)
    prompt,text=get_prompt(cfg)
    
    
    model=TargetedGanAttack(prompt).to(device)
    model.apply(weights_init)

    

    num_epochs = cfg.optim.num_epochs
    cosine_similarity = CosineSimilarity(reduction = 'mean')
    clip_loss=CLIPLoss().to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    basecode_layer = int(np.log2(cfg.basecode_spatial_size) - 2) * 2
    basecode_layer = f'x{basecode_layer-1:02d}'
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=cfg.classifier.lr)
    start_time = time.time()

    for epoch in range(num_epochs):
        # model.train()

        running_loss = 0
        

        for i, (inputs, targets,detail_code,base_code) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            target_emb=classifier(targets)
            base_code=base_code.to(device)
            base_code=base_code.squeeze()
            detail_code=detail_code.to(device)
            detail_code=detail_code.squeeze()
            style=model(detail_code,target_emb)
            
            detail=style*0.1+detail_code
            generated_img = generator.synthesis(detail, randomize_noise=True,
                                                basecode_layer=basecode_layer, basecode=base_code
                                                )['image']
            img_gen=transforms(generated_img)
            optimizer.zero_grad()
            loss_clip=clip_loss(img_gen,text).sum()
            adv_emb = classifier(img_gen)
            
            
            lpips_loss = loss_fn_vgg(inputs, img_gen)
            loss_classifier=cosine_similarity(adv_emb, target_emb)
            loss=cfg.optim.beta*loss_clip+cfg.optim.delta*loss_classifier+lpips_loss
            loss.backward()
            optimizer.step()
            
            
            
            running_loss += loss.item() * inputs.size(0)
            

        epoch_loss = running_loss / len(train_dataset)
        print('[Train #{}] Loss: {:.4f}  Time: {:.4f}s'.format(epoch, epoch_loss,  time.time() - start_time))

    
    













if __name__ == "__main__":
    main()