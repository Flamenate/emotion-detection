import cv2
from deepface import DeepFace

# Read an image
img = cv2.imread("face.png")

emotion_analysis = DeepFace.analyze(img, actions=['emotion'])

# Display the result
print("Detected emotion:", emotion_analysis[0]['dominant_emotion'])

# Show the image with detected emotion
cv2.putText(img, emotion_analysis[0]['dominant_emotion'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
cv2.imshow("Emotion Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()