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

arcface_model = ArcFace.load_model()
arcface_model.load_weights("arcface_weights.h5")

adaface_model = inference.load_pretrained_model('ir_50')

def mag(image):
    def imshow(img):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    def show(idx_):
        imgname = imgnames[idx_]
        img = cv2.imread(imgname)
        imshow(img)
        print(img_2_mag[imgname], imgname)  

    try:
        image_resize = cv2.resize(image, (112, 112))
    except:
        return None, None
        
    # Skriv bilden på en path
    image_path = "/Users/trulskarlsson/Exjobb/BachelorThesis/code/MagFace/inference/MagFaceImages/CurrentFace.jpg"
    cv2.imwrite(image_path, image_resize[:, : , ::-1])

    # Skriv in img path i img.list
    list_file_path = "/Users/trulskarlsson/Exjobb/BachelorThesis/code/MagFace/inference/MagFaceImages/img.list"
    with open(list_file_path, "w") as file:
        file.write(image_path + "\n")
    
    # Mata MagFace med path (en bild i taget)
    os.chdir("/Users/trulskarlsson/Exjobb/BachelorThesis/code/MagFace/inference")
    os.system("time python gen_feat.py --inf_list MagFaceImages/img.list --feat_list MagFaceImages/feat.list --resume magface_epoch_00025.pth --cpu-mode")

    with open('MagFaceImages/feat.list', 'r') as file:
        lines = file.readlines()

    img_2_feats = {}
    img_2_mag = {}
    for line in lines:
        parts = line.strip().split(' ')
        imgname = parts[0]
        feats = [float(e) for e in parts[1:]]
        mag = np.linalg.norm(feats)
        img_2_feats[imgname] = feats/mag
        img_2_mag[imgname] = mag

    imgnames = list(img_2_mag.keys())
    mags = [img_2_mag[imgname] for imgname in imgnames]
    sort_idx = np.argsort(mags)

    feats = np.array([img_2_feats[imgnames[ele]] for ele in sort_idx])
    os.chdir("/Users/trulskarlsson/Exjobb/BachelorThesis/code")
    return feats[0], image

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
    elif extractor_model == "MagFace":
        return mag(image)