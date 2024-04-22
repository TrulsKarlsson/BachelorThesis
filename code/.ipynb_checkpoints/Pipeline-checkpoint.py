import cv2
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import dlib
import numpy as np
import deepface as DeepFace
from deepface.basemodels import ArcFace
from deepface.modules import verification

retinaface_model = RetinaFace.build_model()

model = ArcFace.load_model()
model.load_weights("arcface_weights.h5")

def detection(image_name: str):
    faces = RetinaFace.extract_faces(img_path = image_name, align = True, expand_face_area = 0, model = retinaface_model)
    
    #print("Detection done...")
    return faces

def alignment():
    # Done in detection section (for this time)
    # Might use other more complicated alignment methods later (InsightFace)
    #print("Alignment done...")
    return 

def representation(image):

    if image is None:
        return None, None

    try:
        image_resize = cv2.resize(image, (112, 112))
    except:
        return None, None
    
    img_batch = np.expand_dims(image_resize, axis=0)
    img_representation = model.predict(img_batch)[0]

    #print("Representation done...")
    return img_representation, img_batch

def verification(img1_representation, img1, img2_representation, img2):
    same_person = False
    distance = 0
    
    metric = "cosine"
    #metric = "euclidean"
    #metric = "euclidean_l2"
    
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
        #euclidean_distance = l2_normalize(euclidean_distance)
        return euclidean_distance

    def findThreshold(metric):
        if metric == 'cosine':
            #return 0.6871912959056619
            #return 0.10
            return 0.05
        elif metric == 'euclidean':
            return 4.1591468986978075
        elif metric == 'euclidean_l2':
            return 1.1315718048269017
    
    #img1_embedding = model.predict(img1)[0]
    #img2_embedding = model.predict(img2)[0]

    
    if metric == 'cosine':
        distance = findCosineDistance(img1_representation, img2_representation)
    elif metric == 'euclidean':
        distance = findEuclideanDistance(img1_representation, img2_representation)
    elif metric == 'euclidean_l2':
        distance = findEuclideanDistance(l2_normalize(img1_representation), l2_normalize(img2_representation))
    
    #------------------------------
    #verification
    
    #threshold = verification.find_threshold("retinaface", metric)
    threshold = findThreshold(metric)
    
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

    #ax1 = fig.add_subplot(1,2,1)
    #plt.axis('off')
    #plt.imshow(img1[0])#[:,:,::-1])
  
    #ax2 = fig.add_subplot(1,2,2)
    #plt.axis('off')
    #plt.imshow(img2[0])#[:,:,::-1])

    #plt.show()

    return same_person, distance

def pipeline(image_path1, image_path2):
    # Image 1
    #print("Image 1 started successfully...")
    faces = detection(image_path1)
    img1_representation, img1 = representation(faces[0])

    # Image 2
    #print("\nImage 2 started successfully...")
    faces = detection(image_path2)
    img2_representation, img2 = representation(faces[0])

    if img1 is None or img2 is None:
        return None, True, None
    else:
        same_person, distance = verification(img1_representation, img1, img2_representation, img2)
        return same_person, False, distance
    

def evaluation():
    main_folder_path = "labeled_faces_in_the_wild/lfw"

    first_person_counter = 0
    person_counter = 0

    total_distance = 0

    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0
    
    for root, dirs, files in os.walk(main_folder_path):
        for dir_name in dirs:
            if person_counter == 25: 
                person_counter = 0
            first_person_counter += 1
            print("First person counter:", first_person_counter)
            person_folder_path = os.path.join(root, dir_name)
            # Loop through the images in the person's folder
            for filename in os.listdir(person_folder_path):
                if person_counter == 25:
                    break
                
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path1 = os.path.join(person_folder_path, filename)

                    image1 = cv2.imread(image_path1)
                    if image1 is None:
                        continue
                    
                    # Compare this image with images of other persons
                    for other_root, other_dirs, other_files in os.walk(main_folder_path):
                        if person_counter == 25:
                            break
                        for other_dir_name in other_dirs:
                            if person_counter == 25:
                                break

                            person_counter += 1
                            print("Second person counter:", person_counter)
                            other_person_folder_path = os.path.join(other_root, other_dir_name)
                            if person_folder_path != other_person_folder_path:
                                for other_filename in os.listdir(other_person_folder_path):
                                    if other_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                        image_path2 = os.path.join(other_person_folder_path, other_filename)

                                        image2 = cv2.imread(image_path2)
                                        if image2 is None:
                                            continue
                                            
                                        same_person, error, distance = pipeline(image_path1, image_path2)

                                        if same_person is None:
                                            continue

                                        total_distance += distance

                                        if same_person and not error:
                                            if dir_name == other_dir_name:
                                                true_positive_count += 1
                                            else:
                                                false_positive_count += 1
                                        elif not same_person and not error:
                                            if dir_name != other_dir_name:
                                                true_negative_count += 1
                                            else:
                                                false_negative_count += 1

                                        if true_positive_count + true_negative_count + false_positive_count + false_negative_count == 5000:
                                            avg_distance = total_distance / 5000
                                            
                                            print("\n%%%%%%%%%%%%%%%%%  FINAL RESULTS  %%%%%%%%%%%%%%%%%\n")
                                            print("True Positives:", true_positive_count)
                                            print("False Positives:", false_positive_count)
                                            print("True Negatives:", true_negative_count)
                                            print("False Negatives:", false_negative_count)
                                            print("Average distance:", avg_distance)
                                            print("Threshold used: 0.10")
                                            print("Total comparisons: 5000")
                                            print("\n\n")
                                            exit()

                                        print("\n%%%%%%%%%%%%%%%%%  CURRENT RESULTS  %%%%%%%%%%%%%%%%%\n")
                                        print("True Positives:", true_positive_count)
                                        print("False Positives:", false_positive_count)
                                        print("True Negatives:", true_negative_count)
                                        print("False Negatives:", false_negative_count)
                                        print("Last distance:", distance)
                                        print("\n\n")

                                        

if __name__ == "__main__":
    #pipeline(image_name1, image_name2)
    evaluation()

# Evaluation
# Kör olika thresholds t.ex (0.1)
# Loopa ett ansikte med alla ndra, gå vidare till nästa snikte och gör med alla ndra ansikten igen, presentera resultatet för 0.1 thresholf
# Sen gå vidare 0.2 osv...
# Ta medelvärde på F1-scoret