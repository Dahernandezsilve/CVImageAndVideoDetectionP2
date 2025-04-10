import cv2
import mediapipe as mp
import time

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Configuración del detector Holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Configurar tamaño de la ventana
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Preguntar al usuario si quiere fondo normal o negro
use_black_background = input("¿Quieres fondo negro? (s/n): ").strip().lower() == 's'

# Variables para calcular FPS
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calcular FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Convertir BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Si el usuario eligió fondo negro, crear una imagen negra del mismo tamaño
    if use_black_background:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convertir de nuevo a BGR
        image[:] = (0, 0, 0)  # Poner todo en negro
    
    # Dibujar los resultados en la imagen
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Mostrar la imagen y actualizar el título con los FPS
    cv2.setWindowTitle('MediaPipe Holistic', f'MediaPipe Holistic - FPS: {int(fps)}')
    cv2.imshow('MediaPipe Holistic', image)
    
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
