import matplotlib.pyplot as plt
import numpy as np

def findCosineDistance(source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output	
    
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

# TODO: Change in future
def findThreshold(metric: str, extractor_model: str): 
        if extractor_model == "ArcFace":
            if metric == 'cosine':
                #return 0.3985 # sigma 0
                return 0.5569 # sigma 1
                #return 0.7153 # sigma 2
                #return 0.8737 # sigma 3
                #return 0.6879 # Binary tree classifier
            elif metric == 'euclidean':
                return 0
            elif metric == 'euclidean_l2':
                return 0
        elif extractor_model == "AdaFace":
            if metric == 'cosine':
                #return 0.412 # sigma 0
                #return 0.5385 # sigma 1
                #return 0.665 # sigma 2
                #return 0.8124 # sigma 3
                #return 0.7223 # Binary tree classifier
                #return 0.9
                #return 0.8
                #return 0.825
                #return 0.81
                return 0.815
            elif metric == 'euclidean':
                return 0
            elif metric == 'euclidean_l2':
                return 0
        elif extractor_model == "MagFace":
            if metric == 'cosine':
                #return 0.2821
                #return 0.3736
                #return 0.4651
                #return 0.5566
                return 0.6107
            elif metric == 'euclidean':
                return 0
            elif metric == 'euclidean_l2':
                return 0

def verification(img1_vec, img1, img2_vec, img2, metric, extractor_model) -> (bool, float):
    same_person = False
    distance = 0
    
    if metric == 'cosine':
        distance = findCosineDistance(img1_vec, img2_vec)
    elif metric == 'euclidean':
        distance = findEuclideanDistance(img1_vec, img2_vec)
    elif metric == 'euclidean_l2':
        distance = findEuclideanDistance(l2_normalize(img1_vec), l2_normalize(img2_vec))
    
    #------------------------------
    #verification
    
    threshold = findThreshold(metric, extractor_model)
    
    if distance <= threshold:
        #print("they are same person")
        same_person = True
    else:
        #print("they are different persons")
        same_person = False
    
    #print("Distance is ",round(distance, 2)," whereas as expected max threshold is ",round(threshold, 2))
    
    #------------------------------
    #display
    
    #fig = plt.figure()
#
    #ax1 = fig.add_subplot(1,2,1)
    #plt.axis('off')
    #plt.imshow(img1[0])#[:,:,::-1])
#
    #ax2 = fig.add_subplot(1,2,2)
    #plt.axis('off')
    #plt.imshow(img2[0])#[:,:,::-1])
#
    #plt.show()

    print("\nVerification done...")
    return same_person, distance