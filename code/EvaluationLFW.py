from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PipelineMain import pipeline
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageOps
import os

def evaluationLFW():
    folder_name = "lfwPairs"
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)
    
    #lfw_data = fetch_lfw_pairs(subset='test', color=True, resize=1)
    lfw_data = fetch_lfw_pairs(subset='test', color=True, resize=1, slice_=(slice(0, 250), slice(0, 250)), data_home=folder_path)
    pairs = lfw_data.pairs
    labels = lfw_data.target
    
    actuals = []
    predictions = []
    
    for i in range(0, pairs.shape[0]):
        pair = pairs[i]
        img1 = pair[0]
        img2 = pair[1]

        # Assuming img1 and img2 are paths to image files
        same_person, error, similarity_score = pipeline(img1, img2, extractor_model="AdaFace")
        
        if error:
            print("Error occurred while processing pair")
            continue
        
        if same_person:
            prediction = 1
        else:
            prediction = 0
            
        predictions.append(prediction)
        actual = True if labels[i] == 1 else False
        actuals.append(actual)
    
    accuracy = 100 * accuracy_score(actuals, predictions)
    precision = 100 * precision_score(actuals, predictions)
    recall = 100 * recall_score(actuals, predictions)
    f1 = 100 * f1_score(actuals, predictions)
    
    print("Accuracy: {:.2f}%".format(accuracy))
    print("Precision: {:.2f}%".format(precision))
    print("Recall: {:.2f}%".format(recall))
    print("F1 Score: {:.2f}%".format(f1))
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(actuals, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

if __name__ == "__main__":
    evaluationLFW()