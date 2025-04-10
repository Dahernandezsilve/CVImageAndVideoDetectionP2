import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Customize the drawing specifications for the mesh.
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
connection_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255))

cap = cv2.VideoCapture(0)
# Increase resolution to capture more detail from distant faces.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_face_mesh.FaceMesh(
    max_num_faces=5,             # Set the number of faces to track concurrently.
    refine_landmarks=True,       # Additional refinement around eyes/lips.
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert frame from BGR to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Draw the face mesh if landmarks are detected.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mesh_drawing_spec,
                    connection_drawing_spec=connection_drawing_spec
                )

        cv2.imshow("Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

cap.release()
cv2.destroyAllWindows()
# This script captures video from the webcam, detects faces, and draws a mesh over the detected faces.
# The mesh is drawn with green landmarks and red connections.
# The script uses OpenCV for video capture and MediaPipe for face detection.
# The webcam resolution is set to 1280x720 for better detail.
# The script can be exited by pressing the ESC key.