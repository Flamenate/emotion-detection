# OpenCV Emotion Detection
This is my submission for a university project for the Computer Vision module.

## Setup
It is recommended to run this project in a virtualized environment.
```bash
python -m venv venv
```
Steps to run the project:
* Install Dependencies
```bash
pip install -r requirements.txt
```
* Download training dataset (only if you want to retrain the model)
* A few images for testing (optional)

## File Breakdown
 * [`deepface_demo.py`](./deepface_demo.py): Emotion Detection using DeepFace's analyze feature.
 * [`haar.py`](./haar.py): Cascade Classifier with Haar Cascades for face detection.
 * [`train.py`](./train.py): Model training.
 * [`emotion_model.h5`](./emotion_model.h5): Pretrained model.
 * [`main.py`](./main.py): Main file, using Haar Cascades + pretrained model to predict face emotion from webcam / from image (uncomment code as needed).

 ## Presentation
 You might need to request access.
 https://docs.google.com/presentation/d/1RrblJv2dbhqDKDst1RizXuvVrJQO3fzwB_lqzHM9a0s/edit?usp=sharing


## Sources
* [This Medium Article](https://medium.com/@amit25173/opencv-emotion-recognition-55592e299966)
* ChatGPT
* Kaggle (for datasets)
* OpenCV Documentation (Mainly [the cascade classifier part](https://docs.opencv.org/4.x/db/d28/* tutorial_cascade_classifier.html))
* Gamma