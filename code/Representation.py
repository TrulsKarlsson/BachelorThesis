import matplotlib.pyplot as plt
import cv2
import numpy as np
# ArcFace
import deepface as DeepFace
from deepface.basemodels import ArcFace
# AdaFace
from AdaFace.face_alignment import align
from AdaFace import inference
import torch
# MagFace
import os
import sys
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
        image_resize = cv2.resize(image, (112, 112))
    except:
        return None, None

    img = np.expand_dims(image_resize, axis=0)
    img_representation = arcface_model.predict(img)[0]

    print("Representation done...")
    return img_representation, img

def representation(image, extractor_model: str) -> list:
    if extractor_model == "ArcFace":
        return arc(image)
    elif extractor_model == "AdaFace":
        return ada(image)
    elif extractor_model == "MagFace":
        return mag(image)