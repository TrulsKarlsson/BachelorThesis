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

arcface_model = ArcFace.load_model()
arcface_model.load_weights("arcface_weights.h5")

adaface_model = inference.load_pretrained_model('ir_50')

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

def representation(image, extractor_model: str):
    if extractor_model == "ArcFace":
        return arc(image)
    elif extractor_model == "AdaFace":
        return ada(image)