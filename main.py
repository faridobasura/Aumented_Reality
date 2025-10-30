import cv2
import math
import argparse
import numpy as np
import mediapipe as mp
import logging
# STEP 1: Import the necessary modules.
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
HIP_LEFT = 23
HIP_RIGHT = 24

logger = logging.getLogger(__name__)

def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils  # Para dibujar los puntos y líneas

    cap = cv2.VideoCapture(0)  # Usa 0 para webcam
    cv2.namedWindow("Detección de Pose", cv2.WINDOW_NORMAL)
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o no se pudo leer.")
                break

            # Convertir a RGB para MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar pose
            results = pose.process(rgb)

            # Si hay detecciones, dibujar
            if results.pose_landmarks:
                logger.warning(f"PoseLandMarks{results.pose_landmarks}")
                #Dibuja todos los landmarks
                # mp_drawing.draw_landmarks(
                #    frame,
                #    results.pose_landmarks,
                #    mp_pose.POSE_CONNECTIONS,
                #    #mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                #    #mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                #)
                #ACCEDER A LOS LANDMARKS
                landmarks = results.pose_landmarks.landmark

                # Obtener coordenadas de hombros y caderas
                left_shoulder = landmarks[SHOULDER_LEFT]
                right_shoulder = landmarks[SHOULDER_RIGHT]
                left_hip = landmarks[HIP_LEFT]
                right_hip = landmarks[HIP_RIGHT]

                # Convertir coordenadas normalizadas a píxeles
                h, w, _ = frame.shape

                visibility_threshold = 0.7

                # Obtener coordenadas solo si son suficientemente visibles
                left_shoulder_visible = landmarks[SHOULDER_LEFT].visibility > visibility_threshold
                right_shoulder_visible = landmarks[SHOULDER_RIGHT].visibility > visibility_threshold
                left_hip_visible = landmarks[HIP_LEFT].visibility > visibility_threshold
                right_hip_visible = landmarks[HIP_RIGHT].visibility > visibility_threshold

                if left_shoulder_visible:
                    left_shoulder_x = int(left_shoulder.x * w)
                    left_shoulder_y = int(left_shoulder.y * h)
                    cv2.circle(frame, (left_shoulder_x, left_shoulder_y), 8, (0, 255, 255), -1)  # Amarillo

                if right_shoulder_visible:
                    right_shoulder_x = int(right_shoulder.x * w)
                    right_shoulder_y = int(right_shoulder.y * h)
                    cv2.circle(frame, (right_shoulder_x, right_shoulder_y), 8, (0, 255, 255), -1)  # Amarillo

                if left_hip_visible:
                    left_hip_x = int(left_hip.x * w)
                    left_hip_y = int(left_hip.y * h)
                    cv2.circle(frame, (left_hip_x, left_hip_y), 8, (255, 0, 255), -1)  # Magenta

                if right_hip_visible:
                    right_hip_x = int(right_hip.x * w)
                    right_hip_y = int(right_hip.y * h)   
                    cv2.circle(frame, (right_hip_x, right_hip_y), 8, (255, 0, 255), -1)  # Magenta             

                # 🔹 DIBUJAR LÍNEAS ENTRE HOMBROS Y CADERAS
                if left_shoulder_visible and right_shoulder_visible:
                    cv2.line(frame, (left_shoulder_x, left_shoulder_y), 
                            (right_shoulder_x, right_shoulder_y), (0, 255, 255), 3)
                    shoulder_slope = abs(right_shoulder_y - left_shoulder_y)

                if left_hip_visible and right_hip_visible:
                    cv2.line(frame, (left_hip_x, left_hip_y), 
                            (right_hip_x, right_hip_y), (255, 0, 255), 3)
                    hip_slope = abs(right_hip_y - left_hip_y)

                if left_shoulder_visible and left_hip_visible:
                    cv2.line(frame, (left_shoulder_x, left_shoulder_y), 
                        (left_hip_x, left_hip_y), (255, 255, 255), 2)

                if right_shoulder_visible and right_hip_visible:
                    cv2.line(frame, (right_shoulder_x, right_shoulder_y), 
                            (right_hip_x, right_hip_y), (255, 255, 255), 2)

                if left_shoulder_visible and right_shoulder_visible:
                    cv2.putText(frame, f"Hombros: {shoulder_slope:.2f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if left_hip_visible and right_hip_visible:
                    cv2.putText(frame, f"Caderas: {hip_slope:.2f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                if shoulder_slope > 20:
                    cv2.putText(frame, "INCLINACION DETECTADA", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Detección de Pose", frame)

                #Salir con ESC
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()