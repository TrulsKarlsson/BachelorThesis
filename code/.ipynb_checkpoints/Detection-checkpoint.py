from retinaface import RetinaFace
from retinaface.commons import preprocess
import matplotlib.pyplot as plt
import cv2
import numpy as np

retinaface_model = RetinaFace.build_model()

def detection(image_path: str) -> list: 
    faces = RetinaFace.extract_faces(img_path = image_path, threshold = 0.9, align = True, model = retinaface_model)

    img = cv2.imread(image_path)
    plt.imshow(faces[0])
    plt.savefig('Face.png')
    plt.show()

    print("Detection done...")
    return faces