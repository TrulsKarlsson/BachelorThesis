from retinaface import RetinaFace

import matplotlib.pyplot as plt
import cv2

retinaface_model = RetinaFace.build_model()

def detection(image_path: str):    
    faces = RetinaFace.extract_faces(img_path = image_path, align = False, expand_face_area = 0, model = retinaface_model)
    return faces