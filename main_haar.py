import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model, Sequential
import tensorflow as tf

from haar import load_alt_cascade, load_default_cascade

# Check GPU availability
print("=" * 50)
print("GPU Configuration:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU available: {len(gpus)} GPU(s) detected")
    try:
        for gpu in gpus:
            print(f"  - {gpu.name}")
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠ GPU setup warning: {e}")
else:
    print("⚠ No GPU detected - running on CPU")
print("=" * 50)

emotion_model = load_model("best_emotion_model.keras", compile=False)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

face_cascade = load_alt_cascade()


def detect_and_predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Improved face detection parameters for better accuracy
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,      # Smaller step = more thorough (was 1.1)
        minNeighbors=5,        # Higher = fewer false positives (default was 3)
        minSize=(48, 48),      # Increased from 40 to match model input
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    texts = []
    rectangles = []
    
    if len(faces) == 0:
        return (texts, rectangles)
    
    # Batch preprocessing: prepare all faces at once
    face_rois = []
    face_coords = []
    
    for x, y, w, h in faces:
        # Extract and resize face ROI
        roi_gray = cv2.resize(gray[y : y + h, x : x + w], (48, 48))
        
        # Normalize to [0, 1]
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
        
        face_rois.append(roi_gray)
        face_coords.append((x, y, w, h))
    
    # Batch prediction: predict all faces at once (much faster than one-by-one)
    face_batch = np.array(face_rois)  # Shape: (num_faces, 48, 48, 1)
    predictions = emotion_model.predict(face_batch, verbose=0)  # Batch inference
    
    # Process results
    for i, (x, y, w, h) in enumerate(face_coords):
        emotion_probs = predictions[i]
        
        # Apply softmax temperature to reduce overconfidence
        # Lower temperature (0.5-0.8) = more confident, higher (1.2-2.0) = more spread out
        temperature = 0.7
        emotion_probs = np.exp(np.log(emotion_probs + 1e-8) / temperature)
        emotion_probs = emotion_probs / np.sum(emotion_probs)
        
        emotion_idx = emotion_probs.argmax()
        confidence = emotion_probs[emotion_idx]
        
        # Only show predictions with reasonable confidence (reduces false neutrals)
        if confidence < 0.35:
            label = f"Uncertain ({confidence*100:.1f}%)"
        else:
            label = f"{emotion_labels[emotion_idx]} ({confidence*100:.1f}%)"
        
        texts.append([
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        ])
        rectangles.append([(x, y), (x + w, y + h), (0, 255, 0), 2])
    
    return (texts, rectangles)


def main():
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    
    prev_tick = cv2.getTickCount()
    fps = 0
    texts, rectangles = [], []
    frame_skip = 3  # Process every 3rd frame (adjustable: lower = more responsive, higher = faster)
    iteration = 0
    
    print("Starting emotion detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_tick = cv2.getTickCount()

        iteration += 1
        # Only run detection every N frames to save CPU/GPU
        if iteration % frame_skip == 0:
            texts, rectangles = detect_and_predict_emotion(frame)
            iteration = 0

        # Draw cached results (fast - no model inference)
        for i in range(len(texts)):
            cv2.putText(frame, *texts[i])
            cv2.rectangle(frame, *rectangles[i])

        # Calculate FPS with exponential smoothing
        tick_diff = (current_tick - prev_tick) / cv2.getTickFrequency()
        prev_tick = current_tick
        current_fps = 1.0 / tick_diff if tick_diff > 0 else 0
        fps = fps * 0.9 + current_fps * 0.1  # Smooth the FPS value
        
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Faces: {len(texts)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 10 else (0, 0, 255),
            2,
        )

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
