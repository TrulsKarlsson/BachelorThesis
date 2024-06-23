import pandas as pd
from PipelineMain import pipeline
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def thresholdCalculator(metric: str, extractor_model: str):
    df = pd.read_csv("master.csv")

    instances = df[['file_x', 'file_y']].values.tolist()

    results = []

    for pair in instances:
        image1_path, image2_path = pair
        

        image_path1 = "thresholdImages/" + image1_path
        image_path2 = "thresholdImages/" + image2_path
        
        result = pipeline(image_path1, image_path2, metric, extractor_model)
        
        results.append(result)

    distances = []
    for i in range(0, len(instances)):
        distance = round(results[i][2], 4)
        distances.append(distance)

    df["distance"] = distances

    df[df.Decision == 'Yes'].distance.plot.kde(color='blue', label='Intra')
    
    df[df.Decision == 'No'].distance.plot.kde(color='red', label='Inter')
    
    plt.legend()
    
    plt.xlabel('Similarity')
    plt.ylabel('Density')
    plt.title("KDE Plot of Distances for 'Intra' and 'Inter' Classes")
    
    plt.show()

    tp_mean = round(df[df.Decision == "Yes"]["distance"].mean(), 4)
    tp_std = round(df[df.Decision == "Yes"]["distance"].std(), 4)
    fp_mean = round(df[df.Decision == "No"]["distance"].mean(), 4)
    fp_std = round(df[df.Decision == "No"]["distance"].std(), 4)

    os.system("clear")
    threshold_sigma0 = round(tp_mean + 0 * tp_std, 4)
    threshold_sigma1 = round(tp_mean + 1 * tp_std, 4)
    threshold_sigma2 = round(tp_mean + 2 * tp_std, 4)
    threshold_sigma3 = round(tp_mean + 3 * tp_std, 4)
    
    X = df[['distance']]
    y = df['Decision']
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    threshold = clf.tree_.threshold[0]
    
    print("\n\n")
    print("\n%%%%%%%%%%%%%%%%%  FINAL RESULTS  %%%%%%%%%%%%%%%%%\n")
    print("    Distance formula:", metric)
    print("    Extractor model:", extractor_model)
    print("    Statistical threshold value (sigma = 0):", threshold_sigma0)
    print("    Statistical threshold value (sigma = 1):", threshold_sigma1)
    print("    Statistical threshold value (sigma = 2):", threshold_sigma2)
    print("    Statistical threshold value (sigma = 3):", threshold_sigma3)
    print("    Threshold value decided by the decision tree:", round(threshold, 4))
    print("\n\n")

if __name__ == "__main__":
    extractor_model = "MagFace"
    metric = "cosine"
    thresholdCalculator(metric, extractor_model)