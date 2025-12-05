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

    raw_frame = model_renderer.render_to_image()
    if raw_frame is None:
        return None
    
    rgba = twoD_Render.opengl_to_transparent_rgba(raw_frame)
    return rgba

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
                    if obj_loaded:
                        print(f"📐 Rango de coordenadas del modelo:")
                        vertices_array = np.array(obj_loaded.vertices)
                        print(f"  X: [{vertices_array[:, 0].min():.3f}, {vertices_array[:, 0].max():.3f}]")
                        print(f"  Y: [{vertices_array[:, 1].min():.3f}, {vertices_array[:, 1].max():.3f}]")
                        print(f"  Z: [{vertices_array[:, 2].min():.3f}, {vertices_array[:, 2].max():.3f}]")
                        print(f"  Centro: {vertices_array.mean(axis=0)}")
                    # Calcular tamaño de render basado en el tamaño del frame de la cámara
                    render_width = max(512, min(FRAME_W, 1024))  # Entre 512 y 1024
                    render_height = max(512, min(FRAME_H, 1024))

                    model_renderer = ModelRenderer(width=render_width, height=render_height, obj=obj_loaded)
                    model_renderer.set_render_mode(args.render_mode)

                    if model_renderer is not None:
                        # Test de orientación simple
                        print("\n🧪 TEST DE ORIENTACIÓN BÁSICA")
                        
                        # Aplicar transformación de prueba
                        test_translation = np.array([0.0, 0.0, 0.0])
                        test_rotation = np.eye(3)  # Rotación identidad
                        test_scale = 1.0
                        
                        model_renderer.set_model_transform(test_translation, test_rotation, test_scale)
                        
                        # Renderizar una imagen
                        test_img = model_renderer.render_to_image()
                        
                        if test_img is not None:
                            # Mostrar estadísticas de la imagen
                            print(f"  Imagen de test: {test_img.shape}")
                            print(f"  Valor medio de píxeles: {np.mean(test_img):.2f}")
                            
                            # Verificar si hay algo visible (no todo negro)
                            if np.mean(test_img) > 10:
                                print("  ✅ Modelo visible en renderizado")
                            else:
                                print("  ⚠️  Modelo NO visible (posiblemente fuera de vista)")

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
                    ) * 1.7)

                    torso_height = int(math.sqrt(
                        (left_hip_x - left_shoulder_x) ** 2 +
                        (left_hip_y - left_shoulder_y) ** 2
                    ) * 1.5)

                    torso_center_x = int((left_shoulder_x + right_shoulder_x + left_hip_x) / 3)
                    torso_center_y = int((left_shoulder_y + right_shoulder_y + left_hip_y) / 3)

                    x = torso_center_x - torso_width // 2
                    y = torso_center_y - torso_height // 2

                    angle = math.degrees(math.atan2(
                        right_shoulder_y - left_shoulder_y,
                        right_shoulder_x - left_shoulder_x
                    )) + 180

                    if visible_shoulders and visible_hips:
                        if args.use_3d and model_renderer:
                            if getattr(results, "pose_world_landmarks", None):
                                pl = results.pose_world_landmarks.landmark

                                landmarks_3d = {
                                    "left_shoulder": [pl[SHOULDER_LEFT].x, 
                                                     pl[SHOULDER_LEFT].y,  
                                                     pl[SHOULDER_LEFT].z],
                                    "right_shoulder": [pl[SHOULDER_RIGHT].x, 
                                                      pl[SHOULDER_RIGHT].y,  
                                                      pl[SHOULDER_RIGHT].z],
                                    "left_hip": [pl[HIP_LEFT].x, 
                                                pl[HIP_LEFT].y,           
                                                pl[HIP_LEFT].z],
                                    "right_hip": [pl[HIP_RIGHT].x, 
                                                 pl[HIP_RIGHT].y,   
                                                 pl[HIP_RIGHT].z]
                                }

                                # ✅ CORRECCIÓN 2: Calcular escala dinámica basada en píxeles
                                torso_size_pixels = max(torso_width, torso_height)
                                # Ajusta este factor según tu modelo específico (prueba valores entre 0.5 y 2.0)
                                scale_multiplier = torso_size_pixels / 400.0  # ✅ Cambiado de 500 a 400

                                rotation, scale, translation = model_renderer.align_model_with_landmarks(
                                    landmarks_3d, 
                                    scale_multiplier=scale_multiplier
                                )

                                # Renderizar con la transformación calculada
                                img_3d = render_shirt_adaptive(
                                    model_renderer,
                                    torso_width,
                                    torso_height,
                                    angle,
                                    args.render_mode,
                                    landmarks_3d=landmarks_3d,
                                    use_auto_alignment=True
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