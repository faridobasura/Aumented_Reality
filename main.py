import os
import math
import logging
import argparse

#3rd party
import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
HIP_LEFT = 23
HIP_RIGHT = 24

path = os.path.expanduser('~/AR_python/Aumented_Reality/Black_T_Shirt_PNG_Clip_Art-3107.png')

logger = logging.getLogger(__name__)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Para dibujar los puntos y líneas
cap = cv2.VideoCapture(0)  # Usa 0 para webcam
cv2.namedWindow("Detección de Pose", cv2.WINDOW_NORMAL)

def parse_arguments():

    parser = argparse.ArgumentParser(description='Superposición de prenda con detección de pose')
    
    parser.add_argument(
        '--debug', 
        '-d',
        action='store_true',  
        help='Activar modo debug para mostrar información detallada'
    )
    
    return parser.parse_args()

def overlay_transparent(background, overlay, x, y):
    """Superpone una imagen RGBA sobre otra BGR."""
    h, w = overlay.shape[:2]
    if y >= background.shape[0] or x >= background.shape[1]:
        return background
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)
    overlay_crop = overlay[0:y2 - y1, 0:x2 - x1]

    if overlay_crop.shape[2] < 4:
        return background
    alpha = overlay_crop[:, :, 3] / 255.0
    for c in range(3):
        background[y1:y2, x1:x2, c] = (1 - alpha) * background[y1:y2, x1:x2, c] + alpha * overlay_crop[:, :, c]
    return background

def main():
    args = parse_arguments()
    
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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

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
                landmarks = results.pose_landmarks.landmark

                left_shoulder = landmarks[SHOULDER_LEFT]
                right_shoulder = landmarks[SHOULDER_RIGHT]
                left_hip = landmarks[HIP_LEFT]
                right_hip = landmarks[HIP_RIGHT]

                # Convertir coordenadas normalizadas a píxeles
                h, w, _ = frame.shape

                visibility_threshold = 0.7
                
                left_shoulder_visible = landmarks[SHOULDER_LEFT].visibility > visibility_threshold
                right_shoulder_visible = landmarks[SHOULDER_RIGHT].visibility > visibility_threshold
                left_hip_visible = landmarks[HIP_LEFT].visibility > visibility_threshold
                right_hip_visible = landmarks[HIP_RIGHT].visibility > visibility_threshold

                visible_shoulders = left_shoulder_visible and right_shoulder_visible
                visible_hips = left_hip_visible and right_hip_visible

                if left_shoulder_visible:
                    left_shoulder_x = int(left_shoulder.x * w)
                    left_shoulder_y = int(left_shoulder.y * h)

                if right_shoulder_visible:
                    right_shoulder_x = int(right_shoulder.x * w)
                    right_shoulder_y = int(right_shoulder.y * h)

                if left_hip_visible:
                    left_hip_x = int(left_hip.x * w)
                    left_hip_y = int(left_hip.y * h)
                    
                if right_hip_visible:
                    right_hip_x = int(right_hip.x * w)
                    right_hip_y = int(right_hip.y * h)   
                     
                if args.debug:             

                    if visible_shoulders:
                        cv2.circle(frame, (left_shoulder_x, left_shoulder_y), 8, (0, 255, 255), -1)  
                        cv2.circle(frame, (right_shoulder_x, right_shoulder_y), 8, (0, 255, 255), -1)  
                        cv2.line(frame, (left_shoulder_x, left_shoulder_y), 
                                (right_shoulder_x, right_shoulder_y), (0, 255, 255), 3)
                        shoulder_slope = abs(right_shoulder_y - left_shoulder_y)
                    
                    if visible_hips:
                        cv2.circle(frame, (left_hip_x, left_hip_y), 8, (255, 0, 255), -1)  
                        cv2.circle(frame, (right_hip_x, right_hip_y), 8, (255, 0, 255), -1) 
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

                    if visible_shoulders and visible_hips:

                        torso_width = int(math.sqrt(
                            (right_shoulder_x - left_shoulder_x) ** 2 +
                            (right_shoulder_y - left_shoulder_y) ** 2
                        ) * 1.2)

                        torso_height = int(math.sqrt(
                            (left_hip_x - left_shoulder_x) ** 2 +
                            (left_hip_y - left_shoulder_y) ** 2
                        ) * 1.4)

                        torso_center_x = int((left_shoulder_x + right_shoulder_x + left_hip_x) / 3)
                        torso_center_y = int((left_shoulder_y + right_shoulder_y + left_hip_y) / 3)

                        angle = math.degrees(math.atan2(right_shoulder_y - left_shoulder_y,
                                                        right_shoulder_x - left_shoulder_x)) + 180

                        if not hasattr(main, "cached_shirt") or main.cached_shirt["angle"] != angle or main.cached_shirt["size"] != (torso_width, torso_height):
                            resized = cv2.resize(shirt_png, (torso_width, torso_height))
                            M = cv2.getRotationMatrix2D((torso_width // 2, torso_height // 2), angle, 1.0)
                            rotated = cv2.warpAffine(resized, M, (torso_width, torso_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
                            main.cached_shirt = {"image": rotated, "angle": angle, "size": (torso_width, torso_height)}
                        else:
                            rotated = main.cached_shirt["image"]

                        x = torso_center_x - torso_width // 2
                        y = torso_center_y - torso_height // 2

                        frame = overlay_transparent(frame, rotated, x, y)

                    if shoulder_slope > 20:
                        cv2.putText(frame, "INCLINACION DETECTADA", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Detección de Pose", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try: 
        shirt_png = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        logger.warning(f"Error al cargar la imagen: {e}")

    main()