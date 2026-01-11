import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
import tensorflow as tf
import mediapipe as mp

print("=" * 50)
print("GPU Configuration:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU available: {len(gpus)} GPU(s) detected")
    for gpu in gpus:
        print(f"  - {gpu.name}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠ GPU setup warning: {e}")
else:
    print("⚠ No GPU detected - running on CPU")
print("=" * 50)

emotion_model = load_model("best_emotion_model.keras", compile=False)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


def detect_and_predict_emotion(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    texts = []
    rectangles = []
    
    if not results.detections:
        return (texts, rectangles)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    h, w, _ = frame.shape
    face_rois = []
    face_coords = []
    
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)
        
        x = max(0, x)
        y = max(0, y)
        x2 = min(w, x + box_w)
        y2 = min(h, y + box_h)
        
        if x2 <= x or y2 <= y:
            continue
        
        roi_gray = gray[y:y2, x:x2]
        if roi_gray.size == 0:
            continue
            
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        face_rois.append(roi_gray)
        face_coords.append((x, y, box_w, box_h))
    
    if len(face_rois) == 0:
        return (texts, rectangles)
    
    # Batch prediction
    face_batch = np.array(face_rois)
    predictions = emotion_model.predict(face_batch, verbose=0)
    
    for i, (x, y, box_w, box_h) in enumerate(face_coords):
        emotion_probs = predictions[i]
        
        temperature = 0.7
        emotion_probs = np.exp(np.log(emotion_probs + 1e-8) / temperature)
        emotion_probs = emotion_probs / np.sum(emotion_probs)
        
        emotion_idx = emotion_probs.argmax()
        confidence = emotion_probs[emotion_idx]
        
        sorted_indices = np.argsort(emotion_probs)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]
        top1_conf = emotion_probs[top1_idx]
        top2_conf = emotion_probs[top2_idx]
        
        if top1_conf < 0.35:
            label = f"Uncertain ({top1_conf*100:.0f}%)"
        elif top1_conf - top2_conf < 0.15:  # Close call
            label = f"{emotion_labels[top1_idx]} ({top1_conf*100:.0f}%) / {emotion_labels[top2_idx]} ({top2_conf*100:.0f}%)"
        else:
            label = f"{emotion_labels[top1_idx]} ({top1_conf*100:.0f}%)"
        
        texts.append([
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (36, 255, 12),
            2,
        ])
        rectangles.append([(x, y), (x + box_w, y + box_h), (0, 255, 0), 2])
    
    return (texts, rectangles)

def main():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_tick = cv2.getTickCount()
    fps = 0
    texts, rectangles = [], []
    frame_skip = 5  # Process every 5th frame
    iteration = 0
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_tick = cv2.getTickCount()

        iteration += 1
        if iteration % frame_skip == 0:
            texts, rectangles = detect_and_predict_emotion(frame)
            iteration = 0

        for i in range(len(texts)):
            cv2.putText(frame, *texts[i])
            cv2.rectangle(frame, *rectangles[i])

        tick_diff = (current_tick - prev_tick) / cv2.getTickFrequency()
        prev_tick = current_tick
        current_fps = 1.0 / tick_diff if tick_diff > 0 else 0
        fps = fps * 0.9 + current_fps * 0.1
        
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
    face_detection.close()

if __name__ == "__main__":
    main()
