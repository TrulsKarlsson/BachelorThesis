# Bachelor Thesis
GitHub repository for Truls Karlsson's bachelor thesis at Uppsala University 2024.

## Running the code
To run the code you need to first create a virtual environment on your local machine in a suitable place. After run:

* python3 -m venv env

* source ./env/bin/activate

Once in the (env), you need to load the requirements, please run: 

* pip install -r requirements.txt

You should now be able to run the python notebooks.

# The face recognition pipeline
The face recognition pipeline includes all 4 stages that are found in other state-of-the-art pipelines: detection, alignment, representation and verification [39]. All stages of the pipeline are divided up into their own Python file with the name of the stage, for example Detection.py, and then imported by the main Python file PipelineMain.py.

## PipelineMain.py
'''
def pipeline(image_path1: str, image_path2: str) -> (bool, bool, float)
'''
PipelineMain’s task is to initiate each stage of the pipeline and pass down return values to the next stage as well as taking care of all error handling (see A.1.1). The error handling is designed to be passed upwards in the code so
that PipelineMain handles the error and let the user know which stage that failed. pipeline is designed so that it takes two image paths and returns a tuple containing if it is the same person or not, if an error occurred and the distance between the two persons in the images.

## Detection.py
'''
def detection(image_path: str) -> list
'''
For the detection stage, the face detection model RetinaFace[8] was chosen. This model was chosen as it is one of the supported face detection methods of InsightFace [13]. The function takes a single image path and, if a face is detected, returns an image with only the face extracted. 

## Alignment.py
'''
TBD
'''
Currently, alignment is done in the Detection.py file by RetinaFace’s built-in face alignment tool because of limited time. This will be discussed further in the report under ’Future work’.

## Representation.py
'''
def representation(image, extractor_model: str) -> list
'''
The representation stage is arguably one of the most essential and important parts of this thesis. The goal of the representation function was to be simple and flexible, letting the user choose between the different state-of-the-art feature extractor models that are available. Therefore, the representation function is designed so that it takes an image as well as the user’s choice of feature extractor model that they want to use as a simple string. Depending on the feature extractor model chosen, representation will call an abstracted away function with the name of the model and perform the feature vector extraction and return a tuple with the vector representation as well as the original image.

## Verification.py
'''
def verification(img1_vec, img1, img2_vec, img2, metric, extractor_model)) -> (bool, float)
'''
The verification function takes two, of the same dimension, vector representations, the resized images of both persons for plotting, the choice of distance formula that should be performed on the vector representations and lastly the extractor model used to perform the vector representations in the stage before. The function then calculates the distance between the vector representations and compares it against the set threshold and returns whether it is the same person or not as well as the distance.
