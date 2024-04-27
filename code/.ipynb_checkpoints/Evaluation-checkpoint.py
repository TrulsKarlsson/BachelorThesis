import cv2
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from PipelineMain import pipeline

def evaluation():
    # lfw import som sefik gjorde här

def evaluationOld():
    main_folder_path = "labeled_faces_in_the_wild/lfw"
    #main_folder_path = "evaluationOwnPictures"

    #run_cycles = 1060
    run_cycles = 5000
    #run_cycles = 10000

    bush_counter = 0
    
    first_person_counter = 0
    person_counter = 0

    total_distance = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for root, dirs, files in os.walk(main_folder_path):
        for dir_name in dirs:
            if person_counter == 25: 
                person_counter = 0
            first_person_counter += 1
            print("First person counter:", first_person_counter)
            #if bush_counter == 0:
                #dir_name = "__George_W_Bush"
                #bush_counter += 1
            
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

                            #print("Bush counter:", bush_counter)

                            #if bush_counter == 1:
                                #other_dir_name = "__George_W_Bush"
                                #bush_counter += 1
                                #print("Bush counter:", bush_counter)
                            other_person_folder_path = os.path.join(other_root, other_dir_name)
                            if person_folder_path != other_person_folder_path or person_folder_path == other_person_folder_path:
                                for other_filename in os.listdir(other_person_folder_path):
                                    if other_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                        image_path2 = os.path.join(other_person_folder_path, other_filename)

                                        image2 = cv2.imread(image_path2)
                                        if image2 is None:
                                            continue

                                        #extractor_model = "ArcFace"
                                        extractor_model = "AdaFace"
                                        same_person, error, distance = pipeline(image_path1, image_path2, extractor_model)

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

                                        if TP + TN + FP + FN == run_cycles:
                                            avg_distance = total_distance / run_cycles
                                            accuracy  = (TP + TN)/(TP + TN + FP + FN)
                                            recall    = TP/(TP + FN)
                                            precision = TP/(TP + FP)
                                            F1_score  = (2 * precision * recall)/(precision + recall)

                                            os.system("clear")
                                            print("\n%%%%%%%%%%%%%%%%%  FINAL RESULTS   %%%%%%%%%%%%%%%%%\n")

                                            print("Statistics (align == True)")
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
                                            print("Threshold used: 0.8124")
                                            print("Total comparisons:", run_cycles)
                                            #print("Intrasimilar comparisons:", run_cycles/2)
                                            #print("Intersimilar comparisons:", run_cycles/2)
                                            print("\n\n")
                                            exit()

                                        os.system("clear")
                                        print("\n%%%%%%%%%%%%%%%%%  CURRENT RESULTS  %%%%%%%%%%%%%%%%%\n")
                                        print("True Positives:", TP)
                                        print("False Positives:", FP)
                                        print("True Negatives:", TN)
                                        print("False Negatives:", FN)
                                        print("Last distance:", distance)
                                        print("\n\n")
                                        
                                        

if __name__ == "__main__":
    evaluationOld()

# Evaluation
# Kör olika thresholds t.ex (0.1)
# Loopa ett ansikte med alla ndra, gå vidare till nästa snikte och gör med alla ndra ansikten igen, presentera resultatet för 0.1 thresholf
# Sen gå vidare 0.2 osv...
# Ta medelvärde på F1-scoret