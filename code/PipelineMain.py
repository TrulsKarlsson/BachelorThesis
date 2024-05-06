import cv2
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import deepface as DeepFace
from deepface.basemodels import ArcFace
from deepface.modules import verification
import matplotlib.pyplot as plt
import numpy as np

##### Pipeline imports ####
from Detection import detection
from Alignment import alignment
from Representation import representation
from Verification import verification

def pipeline(image_path1: str, image_path2: str, extractor_model: str) -> (bool, bool, float):
    # Image 1
    print("\nImage 1 started...")
    faces = detection(image_path1)
    if faces is None or len(faces) == 0:
        print("Detection for image 1 failed...")
        return None, True, None
        
    img1_representation, img1 = representation(faces[0], extractor_model)
    if img1_representation is None or img1 is None:
        print("Representation for image 1 failed...")
        return None, True, None

    # Image 2
    print("\nImage 2 started...")
    faces = detection(image_path2)
    if faces is None or len(faces) == 0:
        print("Detection for image 2 failed...")
        return None, True, None
        
    img2_representation, img2 = representation(faces[0], extractor_model)
    if img2_representation is None or img2 is None:
        print("Representation for image 1 failed...")
        return None, True, None

    metric = "cosine"
    #metric = "euclidean"
    #metric = "euclidean_l2"
    
    same_person, distance = verification(img1_representation, img1, img2_representation, img2, metric, extractor_model)
    return same_person, False, distance