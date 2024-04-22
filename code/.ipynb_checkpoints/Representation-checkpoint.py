import matplotlib.pyplot as plt
import cv2
import numpy as np
import deepface as DeepFace
from deepface.basemodels import ArcFace
from deepface.modules import verification

model = ArcFace.load_model()
model.load_weights("arcface_weights.h5")

def representation(image):
    try:
        image_resize = cv2.resize(image, (112, 112))
    except:
        return None, None

    img = np.expand_dims(image_resize, axis=0)
    img_representation = model.predict(img)[0]

    print("Representation done...")
    return img_representation, img