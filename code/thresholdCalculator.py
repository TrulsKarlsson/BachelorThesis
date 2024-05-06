import pandas as pd
from PipelineMain import pipeline
import os
import matplotlib.pyplot as plt

def thresholdCalculator(extractor_model: str):
    df = pd.read_csv("master.csv")

    instances = df[['file_x', 'file_y']].values.tolist()

    results = []

    for pair in instances:
        image1_path, image2_path = pair
        
        # Call the pipeline function with the current pair of image paths
        image_path1 = "thresholdImages/" + image1_path
        image_path2 = "thresholdImages/" + image2_path
        
        result = pipeline(image_path1, image_path2, extractor_model)
        
        # Append the return values to the results list
        results.append(result)

    distances = []
    for i in range(0, len(instances)):
        distance = round(results[i][2], 4)
        distances.append(distance)

    df["distance"] = distances

    df[df.Decision == 'Yes'].distance.plot.kde(color='blue', label='Yes')
    
    # Plot KDE for 'No' class with red color
    df[df.Decision == 'No'].distance.plot.kde(color='red', label='No')
    
    # Add legend to the plot
    plt.legend()
    
    # Add labels and title
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('KDE Plot of Distances for Yes and No Classes')
    
    # Show the plot
    plt.show()

    # Statistical approach
    tp_mean = round(df[df.Decision == "Yes"]["distance"].mean(), 4)
    tp_std = round(df[df.Decision == "Yes"]["distance"].std(), 4)
    fp_mean = round(df[df.Decision == "No"]["distance"].mean(), 4)
    fp_std = round(df[df.Decision == "No"]["distance"].std(), 4)

    os.system("clear")
    threshold_sigma0 = round(tp_mean + 0 * tp_std, 4)
    threshold_sigma1 = round(tp_mean + 1 * tp_std, 4)
    threshold_sigma2 = round(tp_mean + 2 * tp_std, 4)
    threshold_sigma3 = round(tp_mean + 3 * tp_std, 4)  
    
    print("\n\n")
    print("\n%%%%%%%%%%%%%%%%%  FINAL RESULTS  %%%%%%%%%%%%%%%%%\n")
    print("Extractor model:", extractor_model)
    print("Threshold value (sigma = 0):", threshold_sigma0)
    print("Threshold value (sigma = 1):", threshold_sigma1)
    print("Threshold value (sigma = 2):", threshold_sigma2)
    print("Threshold value (sigma = 3):", threshold_sigma3)
    print("\n\n")

if __name__ == "__main__":
    extractor_model = "ArcFace"
    thresholdCalculator(extractor_model)