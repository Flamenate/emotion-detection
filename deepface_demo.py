import cv2
from deepface import DeepFace


def main():
    cap = cv2.VideoCapture(0)
    prev_frame_time = 0
    current_frame_time = 0
    while True:
        _, frame = cap.read()
        try:
            emotion_analysis = DeepFace.analyze(frame, actions=['emotion'])
        except ValueError:
            print("No face detected")
        else:
            cv2.putText(frame, emotion_analysis[0]['dominant_emotion'], (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        finally:
            current_frame_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_frame_time - prev_frame_time)
            prev_frame_time = current_frame_time
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if fps > 15 else (0, 0, 255), 2)

# Show the image with detected emotion
        
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()