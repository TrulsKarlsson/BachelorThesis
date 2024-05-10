import cv2
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from PipelineMain import pipeline  
from Verification import findThreshold

def evaluationOld(metric: str, extractor_model: str, run_cycles: int, threshold):
    main_folder_path = "labeled_faces_in_the_wild/lfw"

    # Run cycles: the total comparisons (50/50 split between intra- and intersimilarities)
    
    same_person_counter = 0
    different_person_counter = 0

    total_distance = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for root, dirs, files in os.walk(main_folder_path):
        for dir_name in dirs:
            
            person_folder_path = os.path.join(root, dir_name)
            
            for filename in os.listdir(person_folder_path):
                
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path1 = os.path.join(person_folder_path, filename)

                    image1 = cv2.imread(image_path1)
                    if image1 is None:
                        continue
                    
                    for other_root, other_dirs, other_files in os.walk(main_folder_path):
                    
                        for other_dir_name in other_dirs:

                            other_person_folder_path = os.path.join(other_root, other_dir_name)
    
                            for other_filename in os.listdir(other_person_folder_path):
                                if other_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    image_path2 = os.path.join(other_person_folder_path, other_filename)
                                    image2 = cv2.imread(image_path2)
                                    if image2 is None:
                                        continue

                                    if person_folder_path != other_person_folder_path and different_person_counter == (run_cycles/2) and same_person_counter != (run_cycles/2):
                                        break
                                    elif person_folder_path == other_person_folder_path and same_person_counter == (run_cycles/2) and different_person_counter != (run_cycles/2):
                                        break
                                    elif same_person_counter == (run_cycles/2) and different_person_counter == (run_cycles/2):
                                        avg_distance = total_distance / run_cycles
                                        accuracy  = (TP + TN)/(TP + TN + FP + FN)
                                        recall    = TP/(TP + FN)
                                        precision = TP/(TP + FP)
                                        F1_score  = (2 * precision * recall)/(precision + recall)
                                        os.system("clear")
                                        print("\n%%%%%%%%%%%%%%%%%  FINAL RESULTS   %%%%%%%%%%%%%%%%%\n")
                                        print("Statistics")
                                        print("True Positives:", TP)
                                        print("False Positives:", FP)
                                        print("True Negatives:", TN)
                                        print("False Negatives:", FN)
                                        print("Average distance:", avg_distance)
                                        
                                        print("\nMeasuring formulas")
                                        print("Accuracy:", accuracy)
                                        print("Recall:", recall)
                                        print("Precision:", precision)
                                        print("F1-score:", F1_score)
                                        
                                        print("\nSpecifications")
                                        print("Extractor model:", extractor_model)
                                        print("Distance formula:", metric)
                                        print("Threshold used:", threshold)
                                        print("Total comparisons:", run_cycles)
                                        print("Intrasimilar comparisons:", run_cycles/2)
                                        print("Intersimilar comparisons:", run_cycles/2)
                                        print("\n\n")
                                        exit()
                                
                            
                                    if person_folder_path != other_person_folder_path:
                                        different_person_counter += 1
                                    elif person_folder_path == other_person_folder_path:
                                        same_person_counter += 1
                                        
                                    same_person, error, distance = pipeline(image_path1, image_path2, metric, extractor_model)
                                    if same_person is None:
                                        continue
                                    total_distance += distance
                                    if same_person and not error:
                                        if dir_name == other_dir_name:
                                            TP += 1
                                        else:
                                            FP += 1
                                    elif not same_person and not error:
                                        if dir_name != other_dir_name:
                                            TN += 1
                                        else:
                                            FN += 1
                                        
                                        
                                    os.system("clear")
                                    print("\n%%%%%%%%%%%%%%%%%  CURRENT RESULTS  %%%%%%%%%%%%%%%%%\n")
                                    print("Distance formula:", metric)
                                    print("Model:", extractor_model)
                                    print("Threshold:", threshold)
                                    print("True Positives:", TP)
                                    print("False Positives:", FP)
                                    print("True Negatives:", TN)
                                    print("False Negatives:", FN)
                                    print("Last distance:", distance)
                                    print("Same person comparisons:", same_person_counter)
                                    print("Different person comparisons:", different_person_counter)
                                    print("\n\n")
                                        
                                        

if __name__ == "__main__":
    #extractor_model = "ArcFace"
    #extractor_model = "AdaFace"
    extractor_model = "MagFace"

    metric = "cosine"
    #metric = "euclidean"
    #metric = "euclidean_l2"

    threshold = findThreshold(metric, extractor_model)
    
    run_cycles = 1000
    evaluationOld(metric, extractor_model, run_cycles, threshold)

# Evaluation
# Kör olika thresholds t.ex (0.1)
# Loopa ett ansikte med alla ndra, gå vidare till nästa snikte och gör med alla ndra ansikten igen, presentera resultatet för 0.1 thresholf
# Sen gå vidare 0.2 osv...
# Ta medelvärde på F1-scoret