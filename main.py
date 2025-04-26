import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# emotion_model = load_model('emotion_model.h5')

# Define emotion labels (lezem in the same order as the FER2013 dataset)
emotion_labels = [
    'Angry', 
    'Disgust', 
    'Fear', 
    'Happy', 
    'Sad', 
    'Surprise', 
    'Neutral'
]

def detect_and_predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    
    for (x, y, w, h) in faces:
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_gray = cv2.resize(roi_gray, (48, 48))
        # roi_gray = roi_gray.astype('float') / 255.0
        # roi_gray = img_to_array(roi_gray)
        # roi_gray = np.expand_dims(roi_gray, axis=0)
        
        # # Predict the emotion
        # preds = emotion_model.predict(roi_gray)[0]
        # label = emotion_labels[preds.argmax()]
        
        # cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

def main():
    # cap = cv2.VideoCapture(0)
    # while True:
    #     _, frame = cap.read()
    #     detect_and_predict_emotion(frame)
        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    detect_and_predict_emotion(cv2.imread('faces.webp'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()