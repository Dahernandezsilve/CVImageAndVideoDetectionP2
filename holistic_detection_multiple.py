import cv2
import torch
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

print("GPU disponible:", torch.cuda.is_available())

# Cargar modelo YOLO
yolo = YOLO("yolov8n.pt")
# yolo = YOLO("yolov8m.pt")
# yolo = YOLO("yolov8s.pt")


# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Captura de video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Preguntar por fondo negro
use_black_background = input("¿Quieres fondo negro? (s/n): ").strip().lower() == 's'

# Inicialización de tiempo para FPS
prev_time = 0

# Procesar video
with mp_holistic.Holistic(static_image_mode=False,
                          model_complexity=0,  # Menos complejidad para mejorar velocidad
                          smooth_landmarks=False,  # No suavizar para mayor velocidad
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Opción de fondo negro
        if use_black_background:
            black_frame = np.zeros_like(frame)
        else:
            black_frame = frame.copy()

        # Detectar personas con YOLO
        results = yolo(frame, classes=[0], device=0)[0]  # Clase 0 = persona

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # Convertir a RGB y procesar con MediaPipe
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            result = holistic.process(person_rgb)

            if result.pose_landmarks:
                # Dibujar los landmarks sobre una copia de la persona
                annotated = np.zeros_like(person_crop) if use_black_background else person_crop.copy()
                mp_drawing.draw_landmarks(
                    annotated,
                    result.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )
                # Insertar la persona procesada en el frame completo
                black_frame[y1:y2, x1:x2] = annotated

        # Calcular FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))  # FPS calculado
        prev_time = curr_time

        # Mostrar FPS en el título de la ventana
        cv2.setWindowTitle("Multi-Person Holistic", f"FPS: {fps}")

        # Mostrar resultado
        cv2.imshow("Multi-Person Holistic", black_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()