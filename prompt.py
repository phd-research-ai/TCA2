import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from omegaconf import DictConfig, OmegaConf
from data.dataset import get_dataset,get_adv_dataset
from utils import get_model,set_requires_grad,unnormalize
import sys
import os
import clip
import torch.nn.functional as F
import time
import hydra
from scipy import io as spio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="./config", config_name="config")
def save_prompt(cfg):
    model, preprocess = clip.load("RN50", device=device)
    prompt=cfg.prompt
    text=clip.tokenize(prompt).to(device)
    with torch.no_grad():
            prompt = model.encode_text(text)
    prompt=prompt.cpu().numpy()
    spio.savemat("prompt.mat",{'prompt':prompt})
    
def get_prompt(cfg):
    model, preprocess = clip.load("RN50", device=device)
    prompt=cfg.prompt
    text=clip.tokenize(prompt).to(device)
    with torch.no_grad():
            prompt = model.encode_text(text)
    return prompt,text



if __name__ == "__main__":
    save_prompt()