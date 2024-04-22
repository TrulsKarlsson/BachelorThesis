import cv2
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from PipelineMain import pipeline

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

                                        if true_positive_count + true_negative_count + false_positive_count + false_negative_count == 1:
                                            avg_distance = total_distance / 5000

                                            os.system("clear")
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

                                        os.system("clear")
                                        print("\n%%%%%%%%%%%%%%%%%  CURRENT RESULTS  %%%%%%%%%%%%%%%%%\n")
                                        print("True Positives:", true_positive_count)
                                        print("False Positives:", false_positive_count)
                                        print("True Negatives:", true_negative_count)
                                        print("False Negatives:", false_negative_count)
                                        print("Last distance:", distance)
                                        print("\n\n")
                                        
                                        

if __name__ == "__main__":
    evaluation()

# Evaluation
# Kör olika thresholds t.ex (0.1)
# Loopa ett ansikte med alla ndra, gå vidare till nästa snikte och gör med alla ndra ansikten igen, presentera resultatet för 0.1 thresholf
# Sen gå vidare 0.2 osv...
# Ta medelvärde på F1-scoret