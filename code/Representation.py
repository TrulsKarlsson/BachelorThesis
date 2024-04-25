import matplotlib.pyplot as plt
import cv2
import numpy as np
# ArcFace
import deepface as DeepFace
from deepface.basemodels import ArcFace
# AdaFace
from AdaFace.face_alignment import align
from AdaFace import inference

arcface_model = ArcFace.load_model()
arcface_model.load_weights("arcface_weights.h5")

adaface_model = inference.load_pretrained_model('ir_50')

def ada(image):
    #path = 'own-pictures/testBildTruls.jpeg'
    path = 'AdaFace/face_alignment/test_images/img1.jpeg'

    img = cv2.imread(path)
    plt.imshow(img[:, : , ::-1])
    plt.show()
    
    aligned_rgb_img = align.get_aligned_face(path)
    
    if aligned_rgb_img is None:
        print("aligned_rgb_img is NoneType")
    
    # Actual AdaFace (only rgd to bgr first)
    bgr_input = inference.to_input(aligned_rgb_img)
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

def representation(image, model: str):
    if model == "ArcFace":
        return arc(image)
    elif model == "AdaFace":
        return ada(image)