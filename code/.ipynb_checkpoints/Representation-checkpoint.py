import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
# ArcFace
import deepface as DeepFace
from deepface.basemodels import ArcFace

sys.path.append('/Users/trulskarlsson/Exjobb/BachelorThesis/code/insightface')
from recognition.arcface_torch import inference
from recognition.arcface_torch.backbones import get_model
import h5py
# AdaFace
from AdaFace.face_alignment import align
from AdaFace import inference
import torch
# MagFace
import os
sys.path.append('/Users/trulskarlsson/Exjobb/BachelorThesis/code/MagFace')
from inference.network_inf import builder_inf
import argparse
from termcolor import cprint
from utils import utils
import cv2
from termcolor import cprint
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision
import torch
import argparse
import numpy as np
import warnings
import time
import pprint
import os

# ArcFace NEW
parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
parser.add_argument('--network', type=str, default='r50', help='backbone network')
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--img', type=str, default=None)
args = parser.parse_args()
#arcface_model = 

# ArcFace
arcface_model = ArcFace.load_model()
arcface_model.load_weights("arcface_weights.h5")

# AdaFace
adaface_model = inference.load_pretrained_model('ir_50')

# MagFace
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--arch', default='iresnet100', help='Description of arch')
parser.add_argument('--inf_list', default='MagFace/inference/MagFaceImages/img.list', help='Description of inf_list')
parser.add_argument('--feat_list', default='MagFace/inference/MagFaceImages/feat.list', help='Description of feat_list')
parser.add_argument('--workers', type=int, default=4, help='Description of workers')
parser.add_argument('--batch_size', type=int, default=256, help='Description of batch_size')
parser.add_argument('--embedding_size', type=int, default=512, help='Description of embedding_size')
parser.add_argument('--resume', default='MagFace/inference/magface_epoch_00025.pth', help='Description of resume')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--cpu-mode', default="True", action='store_true', help='Use the CPU.')

# Parse the argument
args = parser.parse_args()
model = builder_inf(args)
model = torch.nn.DataParallel(model)
if not args.cpu_mode:
    model = model.cuda()

def mag(image):    
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.]),
    ])

    try:
        image_resize = cv2.resize(image, (112, 112))
    except:
        return None, None

    # Switch to evaluation mode
    model.eval()

    # Convert the image to tensor and add a batch dimension
    image_tensor = trans(image_resize.copy())
    image_tensor = torch.unsqueeze(image_tensor, 0)

    # Perform inference
    with torch.no_grad():
        # Pass the tensor through the model
        embedding_feat = model(image_tensor)

        # Normalize the features
        embedding_feat = torch.nn.functional.normalize(embedding_feat, p=2, dim=1)

        # Extract numpy array from tensor
        _feat = embedding_feat.squeeze().cpu().numpy()

    return _feat, image

def ada(image):
    bgr_input = inference.to_input(image)
    
    try:
        bgr_input = torch.nn.functional.interpolate(bgr_input, size=(112, 112), mode='bilinear', align_corners=False)
    except:
        return None, None
    
    feature, _ = adaface_model(bgr_input)
                        
    print("Representation done...")
    return feature.tolist()[0], image

def arc(image):
    try:
        img = cv2.resize(image, (112, 112))
    except:
        return None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model('r50', fp16=False)
    #"arcface_weights.h5"
    #net.load_state_dict(torch.load('backbone.pth'))
    net.load_state_dict(torch.load('backbone.pth', map_location=torch.device('cpu')))
    net.eval()
    #feat = net(img).numpy()
    feat = net(img).detach().numpy()
    return feat[0], image

def representation(image, extractor_model: str) -> list:
    if extractor_model == "ArcFace":
        return arc(image)
    elif extractor_model == "AdaFace":
        return ada(image)
    elif extractor_model == "MagFace":
        return mag(image)