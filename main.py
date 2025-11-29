import os
import math
import logging
import argparse
import pygame
import numpy as np

# 3rd party
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Importar el cargador de objetos 3D
from utils.objectLoader import ObjectLoader
from utils.modelRenderer import Model3DRenderer

SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
HIP_LEFT = 23
HIP_RIGHT = 24

# Rutas de archivos
shirt_path = os.path.expanduser('~/AR_python/Aumented_Reality/Black_T_Shirt_PNG_Clip_Art-3107.png')
obj_path = os.path.expanduser('~/AR_python/Aumented_Reality/t_shirt_model/t_shirt.obj')

logger = logging.getLogger(__name__)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("Detección de Pose", cv2.WINDOW_NORMAL)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Superposición de prenda con detección de pose')
    parser.add_argument('--debug', '-d', action='store_true', help='Activar modo debug')
    parser.add_argument('--use-3d', action='store_true', help='Usar modelo 3D')
    parser.add_argument('--simplify-factor', type=int, default=3,
                       help='Factor de simplificación 3D (mayor = mejor rendimiento)')
    parser.add_argument('--render-mode', type=str, default='textured',  # ← 'textured' por defecto
                       choices=['textured', 'solid', 'wireframe', 'points'],  # ← Nuevas opciones
                       help='Modo de renderizado 3D')
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

def project_3d_to_2d(vertices, frame_shape, landmarks, obj_loader):
    """Proyecta vértices 3D a coordenadas 2D basándose en los puntos de anclaje"""
    h, w = frame_shape[:2]
    
    # Obtener posiciones de anclaje del modelo 3D
    anchor_3d = obj_loader.get_anchor_positions()
    
    # Obtener posiciones 2D de los landmarks
    left_shoulder_2d = (int(landmarks[SHOULDER_LEFT].x * w), int(landmarks[SHOULDER_LEFT].y * h))
    right_shoulder_2d = (int(landmarks[SHOULDER_RIGHT].x * w), int(landmarks[SHOULDER_RIGHT].y * h))
    left_hip_2d = (int(landmarks[HIP_LEFT].x * w), int(landmarks[HIP_LEFT].y * h))
    right_hip_2d = (int(landmarks[HIP_RIGHT].x * w), int(landmarks[HIP_RIGHT].y * h))
    
    # Calcular transformación
    model_points_3d = np.array([
        anchor_3d['left_shoulder'],
        anchor_3d['right_shoulder'], 
        anchor_3d['left_hip'],
        anchor_3d['right_hip']
    ], dtype=np.float32)
    
    image_points_2d = np.array([
        left_shoulder_2d,
        right_shoulder_2d,
        left_hip_2d,
        right_hip_2d
    ], dtype=np.float32)
    
    # Calcular matriz de transformación (simplificada)
    try:
        # Calcular centro y escala
        model_center_3d = np.mean(model_points_3d, axis=0)
        image_center_2d = np.mean(image_points_2d, axis=0)
        
        # Calcular escala basada en la distancia entre hombros
        model_shoulder_dist = np.linalg.norm(model_points_3d[0] - model_points_3d[1])
        image_shoulder_dist = np.linalg.norm(image_points_2d[0] - image_points_2d[1])
        scale = image_shoulder_dist / model_shoulder_dist if model_shoulder_dist > 0 else 1.0
        
        # Proyectar vértices a 2D
        projected_vertices = []
        for vertex in vertices:
            # Centrar y escalar
            centered = vertex - model_center_3d
            scaled = centered * scale
            # Proyectar a 2D (ignorar Z para simplificar)
            x_2d = int(image_center_2d[0] + scaled[0])
            y_2d = int(image_center_2d[1] + scaled[1])
            projected_vertices.append((x_2d, y_2d))
        
        return projected_vertices
    except Exception as e:
        print(f"Error en proyección 3D: {e}")
        return []

def draw_3d_model(frame, projected_vertices, faces, color=(0, 255, 0), thickness=1):
    """Dibuja el modelo 3D proyectado en el frame"""
    # Dibujar aristas
    for face in faces:
        if len(face) >= 3:  # Mínimo triángulo
            points = []
            for vertex_indices in face:
                vertex_idx = vertex_indices[0]  # Índice del vértice
                if vertex_idx < len(projected_vertices):
                    points.append(projected_vertices[vertex_idx])
            
            # Dibujar líneas para cada arista del polígono
            for i in range(len(points)):
                start_point = points[i]
                end_point = points[(i + 1) % len(points)]
                cv2.line(frame, start_point, end_point, color, thickness)
            
def mediaPipeRender():
    args = parse_arguments()

    # ─────────────────────────────
    # 1) Crear camera
    # ─────────────────────────────
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar imagen inicial")
        return

    FRAME_H, FRAME_W = frame.shape[:2]

    # ─────────────────────────────
    # 2) Cargar modelo SOLO UNA VEZ
    # ─────────────────────────────
    obj_loader = None
    model_renderer = None

    if args.use_3d:
        obj_loader = ObjectLoader()

        if obj_loader.load_obj(obj_path):
            model_renderer = Model3DRenderer(
                obj_loader,
                simplify_factor=args.simplify_factor,
                screen_width=FRAME_W,
                screen_height=FRAME_H
            )
        else:
            print("Error al cargar el modelo 3D, desactivando modo 3D.")
            args.use_3d = False

    # ─────────────────────────────
    # 3) Inicializar MediaPipe POSE
    # ─────────────────────────────
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        prev_time = cv2.getTickCount()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o no se pudo leer.")
                break

            # Calcular FPS
            current_time = cv2.getTickCount()
            time_diff = (current_time - prev_time) / cv2.getTickFrequency()
            if time_diff > 0:
                fps = 1 / time_diff
            prev_time = current_time

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                left_shoulder_visible = landmarks[SHOULDER_LEFT].visibility > 0.8
                right_shoulder_visible = landmarks[SHOULDER_RIGHT].visibility > 0.8
                left_hip_visible = landmarks[HIP_LEFT].visibility > 0.8
                right_hip_visible = landmarks[HIP_RIGHT].visibility > 0.8

                visible_shoulders = left_shoulder_visible and right_shoulder_visible
                visible_hips = left_hip_visible and right_hip_visible
                
                 # Render del modelo 3D si existe pose                    
                if args.use_3d and model_renderer and visible_shoulders and visible_hips:
                    # Usar modelo 3D optimizado
                    frame = model_renderer.render_on_frame(frame, results.pose_landmarks)

                    
                    #if projected_vertices:
                    #    model_renderer.render_model(frame, projected_vertices, render_mode=args.render_mode)
                        
                elif visible_shoulders and visible_hips:
                    # Usar imagen 2D (código original optimizado)
                    h, w = frame.shape[:2]
                    
                    left_shoulder_x = int(landmarks[SHOULDER_LEFT].x * w)
                    left_shoulder_y = int(landmarks[SHOULDER_LEFT].y * h)
                    right_shoulder_x = int(landmarks[SHOULDER_RIGHT].x * w)
                    right_shoulder_y = int(landmarks[SHOULDER_RIGHT].y * h)
                    left_hip_x = int(landmarks[HIP_LEFT].x * w)
                    left_hip_y = int(landmarks[HIP_LEFT].y * h)

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

                    angle = math.degrees(math.atan2(
                        right_shoulder_y - left_shoulder_y,
                        right_shoulder_x - left_shoulder_x
                    )) + 180

                    # Cache de la camiseta
                    if not hasattr(mediaPipeRender, "cached_shirt") or \
                       mediaPipeRender.cached_shirt["angle"] != angle or \
                       mediaPipeRender.cached_shirt["size"] != (torso_width, torso_height):
                        resized = cv2.resize(shirt_png, (torso_width, torso_height))
                        M = cv2.getRotationMatrix2D((torso_width // 2, torso_height // 2), angle, 1.0)
                        rotated = cv2.warpAffine(resized, M, (torso_width, torso_height), 
                                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
                        mediaPipeRender.cached_shirt = {
                            "image": rotated, 
                            "angle": angle, 
                            "size": (torso_width, torso_height)
                        }
                    else:
                        rotated = mediaPipeRender.cached_shirt["image"]

                    x = torso_center_x - torso_width // 2
                    y = torso_center_y - torso_height // 2

                    frame = overlay_transparent(frame, rotated, x, y)

            if args.debug and results.pose_landmarks:
                mp_drawing.draw_landmarks(
                   frame,
                   results.pose_landmarks,
                   mp_pose.POSE_CONNECTIONS
                )

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ACTUALIZA ESTA LÍNEA:
            mode_text = f"Modo: {'3D-' + args.render_mode if args.use_3d else '2D'}"
            cv2.putText(frame, mode_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Opcional: mostrar controles disponibles
            controls_text = "Controles: T=Textura S=Solido W=Wireframe P=Puntos"
            cv2.putText(frame, controls_text, (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Detección de Pose", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('3'):  # Cambiar a modo 3D
                args.use_3d = True
                if not obj_loader:
                    obj_loader = ObjectLoader()
                    if obj_loader.load_obj(obj_path):
                        model_renderer = Model3DRenderer(obj_loader, args.simplify_factor)
            elif key == ord('2'):  # Cambiar a modo 2D
                args.use_3d = False

            # ✅ AGREGA ESTOS NUEVOS CONTROLES:
            elif key == ord('t') and model_renderer:  # Modo texturas
                args.render_mode = 'textured'
                print("🎨 Modo: Texturas")
            elif key == ord('s') and model_renderer:  # Modo sólido
                args.render_mode = 'solid'
                print("🎨 Modo: Color sólido")
            elif key == ord('w') and model_renderer:  # Modo wireframe
                args.render_mode = 'wireframe' 
                print("🎨 Modo: Wireframe")
            elif key == ord('p') and model_renderer:  # Modo puntos
                args.render_mode = 'points'
                print("🎨 Modo: Puntos")
                
            elif key == ord('r'):  # recargar modelo y renderer
                if obj_loader and obj_loader.load_obj(obj_path):
                    model_renderer = Model3DRenderer(obj_loader, simplify_factor=args.simplify_factor,
                                                     screen_width=FRAME_W, screen_height=FRAME_H)
                    print("Modelo recargado.")


    cap.release()
    cv2.destroyAllWindows()

def main():
    global shirt_png
    
    # Cargar imagen 2D
    try: 
        shirt_png = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
        if shirt_png is None:
            print(f"Error: No se pudo cargar la imagen en {shirt_path}")
            return
    except Exception as e:
        logger.warning(f"Error al cargar la imagen: {e}")
        return

    # Inicializar Pygame para el cargador de objetos 3D
    pygame.init()
    
    mediaPipeRender()

if __name__ == "__main__":
    main()