import os
import math
import logging
import argparse
import pygame
import numpy as np
import glfw 

# 3rd party
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Importar el cargador de objetos 3D
from utils.objectLoader import ObjModel
from utils import twoD_Render
from utils.modelRenderer import ModelRenderer
from utils.modelUtils import compute_torso_frame


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
    parser.add_argument('--render-mode',
            type=str,
            default='wireframe',
            choices=['textured', 'wireframe'],
            help='Modo de renderizado 3D'
        )

    return parser.parse_args()

def render_shirt_adaptive(model_renderer, torso_width, torso_height, angle, render_mode, 
                         landmarks_3d=None, use_auto_alignment=True):
    """
    Renderiza la camisa con alineación automática a landmarks
    """
    if model_renderer is None:
        return None

    model_renderer.set_viewport(torso_width, torso_height)
    model_renderer.set_render_mode(render_mode)

    # Si tenemos landmarks, alinear automáticamente
    if landmarks_3d is not None and use_auto_alignment:
        print(f"🔍 Alineando modelo con {len(landmarks_3d)} landmarks...")
        
        # Usar un factor de escala fijo para pruebas
        scale_factor = 2.5  # <-- Ajusta este valor según necesites
        
        rotation, scale, translation = model_renderer.align_model_with_landmarks(
            landmarks_3d, 
            scale_multiplier=scale_factor  # Usar escala fija
        )
        
        if rotation is not None:
            # Aplicar transformación
            model_renderer.set_model_transform(translation, rotation, scale)
            
            # Verificar alineación
            avg_error = model_renderer.verify_alignment_quality(landmarks_3d)
            logger.warning(f"GRADO DE ERROR: {avg_error}")

    raw_frame = model_renderer.render_to_image()
    if raw_frame is None:
        return None
    
    rgba = twoD_Render.opengl_to_transparent_rgba(raw_frame)
    return rgba

def correct_coordinate_system(translation, rotation_mat):
    """
    Corrección mejorada con inversión de giro
    """
    # 1. Invertir Y (arriba/abajo)
    translation_corrected = translation.copy()
    translation_corrected[1] = -translation[1]
    
    # 2. Invertir eje X para corregir giro contrario
    # Cuando el usuario gira a la derecha, la rotación debe ser positiva en Y
    # Si está invertido, invertimos el eje X
    flip_x = np.array([
        [-1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]
    ])
    
    rotation_corrected = rotation_mat @ flip_x
    
    # 3. DEBUG: Mostrar ángulos de Euler para verificar
    # Convertir matriz de rotación a ángulos de Euler
    sy = np.sqrt(rotation_corrected[0,0] * rotation_corrected[0,0] + 
                 rotation_corrected[1,0] * rotation_corrected[1,0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rotation_corrected[2,1], rotation_corrected[2,2])
        y = np.arctan2(-rotation_corrected[2,0], sy)
        z = np.arctan2(rotation_corrected[1,0], rotation_corrected[0,0])
    else:
        x = np.arctan2(-rotation_corrected[1,2], rotation_corrected[1,1])
        y = np.arctan2(-rotation_corrected[2,0], sy)
        z = 0
    
    print(f"🎯 Ángulos de Euler después de corrección: "
          f"X={np.degrees(x):.1f}°, Y={np.degrees(y):.1f}°, Z={np.degrees(z):.1f}°")
    
    return translation_corrected, rotation_corrected

            
def mediaPipeRender():
    args = parse_arguments()

    # ─────────────────────────────
    # 1) Crear camera
    # ─────────────────────────────
    DESIRED_WIDTH = 1280
    DESIRED_HEIGHT = 720
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar imagen inicial")
        return

    FRAME_H, FRAME_W = frame.shape[:2]
    
    # Crear ventana OpenCV grande
    cv2.namedWindow("Detección de Pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detección de Pose", FRAME_W, FRAME_H)

    # ─────────────────────────────
    # 2) Cargar modelo SOLO UNA VEZ (usar args ya obtenido)
    # ─────────────────────────────
    obj_loaded = None
    model_renderer = None

    try:
        if args.use_3d:
            try:
                obj_loaded = ObjModel.load_obj(obj_path)
                if obj_loaded is not None:
                    # Calcular tamaño de render basado en el tamaño del frame de la cámara
                    render_width = max(512, min(FRAME_W, 1024))  # Entre 512 y 1024
                    render_height = max(512, min(FRAME_H, 1024))

                    model_renderer = ModelRenderer(width=render_width, height=render_height, obj=obj_loaded)
                    model_renderer.set_render_mode(args.render_mode)

                    print(f"✅ Modelo 3D cargado. Tamaño de render: {render_width}x{render_height}")
                else:
                    print("❌ objectLoader devolvió None para el OBJ.")
                    args.use_3d = False
            except Exception as e:
                logger.error(f"Error cargando OBJ: {e}")
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

                # Calcula tamaño del frame inmediatamente (para usar en coordenadas)
                h, w = frame.shape[:2]

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

                    # ---- Calcular todas las coordenadas y métricas UNA vez ----
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

                    x = torso_center_x - torso_width // 2
                    y = torso_center_y - torso_height // 2

                    angle = math.degrees(math.atan2(
                        right_shoulder_y - left_shoulder_y,
                        right_shoulder_x - left_shoulder_x
                    )) + 180

                    # ---- Si los dos hombros y ambas caderas son visibles procedemos ----
                    if visible_shoulders and visible_hips:

                        # después de calcular visible_shoulders and visible_hips
                        # En la sección donde procesas los landmarks 3D:
                        if args.use_3d and model_renderer:
                            # get 3D points
                            if getattr(results, "pose_world_landmarks", None):
                                pl = results.pose_world_landmarks.landmark

                                # Crear diccionario de landmarks 3D
                                landmarks_3d = {
                                    "left_shoulder": [pl[SHOULDER_LEFT].x, pl[SHOULDER_LEFT].y, pl[SHOULDER_LEFT].z],
                                    "right_shoulder": [pl[SHOULDER_RIGHT].x, pl[SHOULDER_RIGHT].y, pl[SHOULDER_RIGHT].z],
                                    "left_hip": [pl[HIP_LEFT].x, pl[HIP_LEFT].y, pl[HIP_LEFT].z],
                                    "right_hip": [pl[HIP_RIGHT].x, pl[HIP_RIGHT].y, pl[HIP_RIGHT].z]
                                }

                                # Opción A: Usar alineación automática con Procrustes
                                rotation, scale, translation = model_renderer.align_model_with_landmarks(landmarks_3d)

                                # Opción B: Si la alineación automática no funciona, usar tu método actual
                                # pero con la escala calculada para que coincidan los hombros
                                if scale <= 0:  # Si la alineación falló
                                    pL = np.array(landmarks_3d["left_shoulder"])
                                    pR = np.array(landmarks_3d["right_shoulder"])
                                    pH_left = np.array(landmarks_3d["left_hip"])
                                    pH_right = np.array(landmarks_3d["right_hip"])
                                    pH = (pH_left + pH_right) / 2.0

                                    translation, rotation_mat, detected_width = compute_torso_frame(pL, pR, pH)
                                    translation, rotation_mat = correct_coordinate_system(translation, rotation_mat)

                                    # Calcular escala basada en distancia de hombros
                                    if model_renderer.model_shoulder_dist > 0:
                                        scale = detected_width / model_renderer.model_shoulder_dist
                                        # Ya no multiplicamos por 2.5, usamos la escala exacta
                                        # scale *= 1.0  # Sin ajuste adicional
                                    else:
                                        scale = 1.0

                                # Renderizar con la transformación calculada
                                img_3d = render_shirt_adaptive(
                                    model_renderer,
                                    torso_width,
                                    torso_height,
                                    angle,
                                    args.render_mode,
                                    landmarks_3d=landmarks_3d,  # Pasar los landmarks
                                    use_auto_alignment=True     # Activar alineación automática
                                )

                            if img_3d is not None:
                                frame = twoD_Render.overlay_transparent(frame, img_3d, x, y)

                        else:
                            # ---- Modo 2D (cache) ----
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

                            frame = twoD_Render.overlay_transparent(frame, rotated, x, y)

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
                
                elif key == ord('3'):  
                    args.use_3d = True
                    print("🔄 Cambiado a modo 3D")

                elif key == ord('2'):  
                    args.use_3d = False
                    print("🔄 Cambiado a modo 2D")

                elif key == ord('w'):
                    args.render_mode = "wireframe"
                    print("🔄 Cambiado a modo WIRE FRAME")

                elif key == ord('t'):
                    args.render_mode = "textured"
                    print("🖼 Cambiado a modo TEXTURED")
    except KeyboardInterrupt:
        print("\n🛑 Interrupción por teclado")
    finally:
        # Limpiar recursos
        if args.use_3d and model_renderer is not None:
            model_renderer.cleanup()

        # Liberar cámara y cerrar ventanas
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
    
    mediaPipeRender()

if __name__ == "__main__":
    main()