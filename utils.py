import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models
import numpy as np
import PIL
from fr_model import IRSE_50, MobileFaceNet, IR_152, InceptionResnetV1


# def make_noise(batch, latent_dim, n_noise, device):
#     if n_noise == 1:
#         return torch.randn(batch, latent_dim, device=device)

#     noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

#     return noises



def get_model(config):

    
    if config.classifier.model == 'IRSE50':
        model = IRSE_50()
        model.load_state_dict(torch.load('pretrained_model/irse50.pth', map_location=torch.device('cuda')))
    elif config.classifier.model == 'MobileFace':
        model = MobileFaceNet(512)
        model.load_state_dict(torch.load('pretrained_model/mobile_face.pth', map_location=torch.device('cuda')))
    elif config.classifier.model == 'IR152':
        model = IR_152([112, 112])
        model.load_state_dict(torch.load('pretrained_model/ir152.pth', map_location=torch.device('cuda')))
    elif config.classifier.model == 'FaceNet':
        model = InceptionResnetV1(num_classes=8631)
        model.load_state_dict(torch.load('pretrained_model/facenet.pth', map_location=torch.device('cuda')))
    else:
        raise ValueError(f'Invalid model name: {config.classifier.model}')
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    
    return model

def unnormalize(image):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 3, 1, 1).float()
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 3, 1, 1).float()
    
    image = image.detach().cpu()
    image *= std
    image += mean
    image[image < 0] = 0
    image[image > 1] = 1

    return image

def normalize(image):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 3, 1, 1).float().cuda()
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 3, 1, 1).float().cuda()
    
    image = image.clone()
    image -= mean
    image /= std
    
    return image

def set_requires_grad( nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def preprocess(images, channel_order='RGB'):
    """Preprocesses the input images if needed.
    This function assumes the input numpy array is with shape [batch_size,
    height, width, channel]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. The returned images are with shape
    [batch_size, channel, height, width].
    NOTE: The channel order of input images is always assumed as `RGB`.
    Args:
      images: The raw inputs with dtype `numpy.uint8` and range [0, 255].
    Returns:
      The preprocessed images with dtype `numpy.float32` and range
        [-1, 1].
    """
    # input : numpy, np.uint8, 0~255, RGB, BHWC
    # output : numpy, np.float32, -1~1, RGB, BCHW

    image_channels = 3
    max_val = 1.0
    min_val = -1.0

    if image_channels == 3 and channel_order == 'BGR':
      images = images[:, :, :, ::-1]
    images = images / 255.0 * (max_val - min_val) + min_val
    images = images.astype(np.float32).transpose(0, 3, 1, 2)
    return images

def postprocess(images):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
    # input : tensor, -1~1, RGB, BCHW
    # output : np.uint8, 0~255, BGR, BHWC

    images = images.detach().cpu().numpy()
    images = (images + 1.) * 255. / 2.
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)[:,:,:,[2,1,0]]
    return images

def Lanczos_resizing(image_target, resizing_tuple=(256,256)):
    # input : -1~1, RGB, BCHW, Tensor
    # output : -1~1, RGB, BCHW, Tensor
    image_target_resized = image_target.clone().cpu().numpy()
    image_target_resized = (image_target_resized + 1.) * 255. / 2.
    image_target_resized = np.clip(image_target_resized + 0.5, 0, 255).astype(np.uint8)

    image_target_resized = image_target_resized.transpose(0, 2, 3, 1)
    tmps = []
    for i in range(image_target_resized.shape[0]):
        tmp = image_target_resized[i]
        tmp = Image.fromarray(tmp) # PIL, 0~255, uint8, RGB, HWC
        tmp = np.array(tmp.resize(resizing_tuple, PIL.Image.LANCZOS))
        tmp = torch.from_numpy(preprocess(tmp[np.newaxis,:])).cuda()
        tmps.append(tmp)
    return torch.cat(tmps, dim=0)


th_dict = {'IR152':(0.094632, 0.166788, 0.227922), 'IRSE50':(0.144840, 0.241045, 0.312703),
           'FaceNet':(0.256587, 0.409131, 0.591191), 'MobileFace':(0.183635, 0.301611, 0.380878)}

def asr_calculation(cos_sim_scores_dict):
    # Iterate each image pair's simi-score from "simi_scores_dict" and compute the attacking success rate
    for key, values in cos_sim_scores_dict.items():
        th01, th001, th0001 = th_dict[key]
        total = len(values)
        success01 = 0
        success001 = 0
        success0001 = 0
        for v in values:
            if v > th01:
                success01 += 1
            if v > th001:
                success001 += 1
            if v > th0001:
                success0001 += 1
        print(key, " attack success(far@0.1) rate: ", success01 / total)
        print(key, " attack success(far@0.01) rate: ", success001 / total)
        print(key, " attack success(far@0.001) rate: ", success0001 / total)