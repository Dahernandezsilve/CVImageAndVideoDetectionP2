import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Drawing specs for mesh
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
connection_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255))

cap = cv2.VideoCapture(0)

# Increase resolution to help track distant faces
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    # Lower the detection confidence slightly to try to pick up smaller faces
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5
) as face_mesh:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mesh_drawing_spec,
                    connection_drawing_spec=connection_drawing_spec
                )

                # Compute and draw bounding box for each face
                h, w, _ = frame.shape
                xs = [landmark.x for landmark in face_landmarks.landmark]
                ys = [landmark.y for landmark in face_landmarks.landmark]
                x_min, x_max = int(min(xs)*w), int(max(xs)*w)
                y_min, y_max = int(min(ys)*h), int(max(ys)*h)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow('Multiple Face Mesh', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
