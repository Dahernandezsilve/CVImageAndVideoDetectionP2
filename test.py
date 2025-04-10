import cv2
import torch
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

print("GPU disponible:", torch.cuda.is_available())

# Cargar modelo YOLO
yolo = YOLO("yolov8n.pt")  # Puedes cambiar a yolov8s.pt o yolov8m.pt si deseas

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Captura de video
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Inicialización de tiempo para FPS
prev_time = 0

# Procesar video
with mp_holistic.Holistic(static_image_mode=False,
                          model_complexity=0,
                          smooth_landmarks=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        black_frame = np.zeros_like(frame)
        original_frame = frame.copy()

        # Detectar personas con YOLO
        results = yolo(frame, classes=[0], device=0)[0]

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # Convertir a RGB y procesar con MediaPipe
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            result = holistic.process(person_rgb)

            if result.pose_landmarks:
                # Dibujar en fondo negro
                annotated_black = np.zeros_like(person_crop)
                mp_drawing.draw_landmarks(
                    annotated_black,
                    result.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )
                black_frame[y1:y2, x1:x2] = annotated_black

                # Dibujar también sobre la imagen original
                annotated_original = person_crop.copy()
                mp_drawing.draw_landmarks(
                    annotated_original,
                    result.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )
                original_frame[y1:y2, x1:x2] = annotated_original

        # Calcular FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time else 0
        prev_time = curr_time

        # Mostrar FPS en las ventanas
        cv2.setWindowTitle("Fondo Negro", f"Fondo Negro - FPS: {fps}")
        cv2.setWindowTitle("Video Original", f"Video Original - FPS: {fps}")

        # Mostrar ambas ventanas
        cv2.imshow("Fondo Negro", black_frame)
        cv2.imshow("Video Original", original_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
