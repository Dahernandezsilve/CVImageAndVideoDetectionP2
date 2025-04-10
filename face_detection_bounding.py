import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# Increase resolution to help track smaller/distant faces
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_face_detection.FaceDetection(
    model_selection=1,  # 1 works best for full-range detection
    min_detection_confidence=0.5
) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to RGB as MediaPipe expects RGB input.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        # If faces are detected, draw bounding boxes and keypoints.
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
        
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

cap.release()
cv2.destroyAllWindows()
# Note: This code requires the installation of OpenCV and MediaPipe.
# Make sure to have a webcam connected for real-time face detection.
