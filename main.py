import os
import math
import logging
import numpy as np

# 3rd party
import cv2
import mediapipe as mp

# Importar el cargador de objetos 3D
from utils.objectLoader import ObjModel
from utils import twoD_Render
from utils.modelRenderer import ModelRenderer


from utils.app_args import args

SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
HIP_LEFT = 23
HIP_RIGHT = 24

# Rutas de archivos
shirt_path = os.path.expanduser('~/AR_python/Aumented_Reality/models/Black_T_Shirt_PNG_Clip_Art-3107.png')
obj_path = os.path.expanduser('~/AR_python/Aumented_Reality/models/obj/t_shirt.obj')

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("Detección de Pose", cv2.WINDOW_NORMAL)


def load_textures(model_renderer, textures_dir):
    """
    Carga texturas SOLO desde archivos (no crea texturas por defecto)
    
    Returns:
        list: Nombres de texturas cargadas exitosamente
    """
    textures_loaded = []
    
    print(f"\n🎨 Cargando texturas desde archivos...")
    print(f"   Directorio: {textures_dir}")
    
    # SOLO las texturas de archivo que quieres
    texture_files = [
        ("shirt_black", "black_solid.png"),
        ("shirt_white", "white_solid.png"),
        ("shirt_red", "red_solid.png"),
    ]
    
    files_found = 0
    
    for tex_name, filename in texture_files:
        filepath = os.path.join(textures_dir, filename)
        
        if os.path.exists(filepath):
            if model_renderer.texture_renderer.load_texture(tex_name, filepath):
                textures_loaded.append(tex_name)
                files_found += 1
            else:
                print(f"   ❌ Error cargando: {filename}")
        else:
            print(f"   ⚠️  Archivo no encontrado: {filename}")
    
    print(f"\n📊 Total de texturas cargadas: {files_found}")
    
    return textures_loaded

def list_available_textures(textures_dir):
    """
    Lista las texturas disponibles en el directorio especificado.
    """
    print("\n📋 TEXTURAS DISPONIBLES (desde archivos):")
    print("=" * 60)
    
    texture_files = [
        ("shirt_black", "black_solid.png", "Playera negra sólida"),
        ("shirt_white", "white_solid.png", "Playera blanca sólida"),
       # ("shirt_blue", "blue_solid.png", "Playera azul sólida"),
        ("shirt_red", "red_solid.png", "Playera roja sólida"),
    ]
    
    available_files = []
    unavailable_files = []
    
    for tex_name, filename, description in texture_files:
        filepath = os.path.join(textures_dir, filename)
        
        if os.path.exists(filepath):
            available_files.append((tex_name, filename, description))
        else:
            unavailable_files.append((tex_name, filename, description))
    
    if available_files:
        print("\n  ✅ DISPONIBLES:")
        for tex_name, filename, description in available_files:
            print(f"    • {tex_name:15} - {description}")
            print(f"        Archivo: {filename}")
    
    if unavailable_files:
        print(f"\n  ⚠️  NO DISPONIBLES (archivos faltantes):")
        for tex_name, filename, description in unavailable_files:
            print(f"    • {tex_name:15} - {description}")
            print(f"        Archivo faltante: {filename}")
    
    print("\n🎮 USO:")
    print("  Para usar una textura: python main.py --texture NOMBRE")
    print("  Ejemplo: python main.py --texture shirt_black")
    print("=" * 60)
    
    return available_files

def print_info(frame=None, fps=0, model_renderer=None):
    info_y = 30
    line_height = 30
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += line_height
    # Modo de renderizado
    mode_text = f"Modo: {'3D-' + args.render_mode if args.use_3d else '2D'}"
    cv2.putText(frame, mode_text, (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += line_height
    # Textura activa
    if model_renderer and args.use_3d:
        tex_name = model_renderer.texture_renderer.active_texture_name or "unknown"
        cv2.putText(frame, f"Textura: {tex_name}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += line_height
        # NUEVO: Mostrar brillo
        brightness_text = f"Brillo: {model_renderer.brightness:.2f}"
        cv2.putText(frame, brightness_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Amarillo
        info_y += line_height
    # Ayuda
    cv2.putText(frame, "Presiona 'H' para ayuda", (10, FRAME_H - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
def read_keyboard(key, model_renderer=None, has_uvs=False):
    
    # En el bucle principal, actualiza los controles:
    # Cambiar texturas con teclas 1-4
    if key == ord('1') and model_renderer and "shirt_black" in model_renderer.texture_renderer.list_textures():
        model_renderer.texture_renderer.set_active_texture("shirt_black")
        print(f"🎨 Textura: shirt_black")
    elif key == ord('2') and model_renderer and "shirt_white" in model_renderer.texture_renderer.list_textures():
        model_renderer.texture_renderer.set_active_texture("shirt_white")
        print(f"🎨 Textura: shirt_white")
    #elif key == ord('3') and model_renderer and "shirt_blue" in model_renderer.texture_renderer.list_textures():
    #    model_renderer.texture_renderer.set_active_texture("shirt_blue")
    #    print(f"🎨 Textura: shirt_blue")
    elif key == ord('4') and model_renderer and "shirt_red" in model_renderer.texture_renderer.list_textures():
        model_renderer.texture_renderer.set_active_texture("shirt_red")
        print(f"🎨 Textura: shirt_red")
    elif key == ord('t') and model_renderer and has_uvs:
        args.render_mode = "textured"
        model_renderer.set_render_mode("textured")
        print(f"🎨 Modo: textured")
    elif key == ord('w') and model_renderer:
        args.render_mode = "wireframe"
        model_renderer.set_render_mode("wireframe")
        print(f"🎨 Modo: wireframe")
    elif key == ord('v'):
        if model_renderer:
            args.use_3d = True
            print(f"🔄 Modo 3D activado")
    elif key == ord('u'):
        args.use_3d = False
        print(f"🔄 Modo 2D activado")
    elif key == ord('l'):  # Listar texturas disponibles
        if model_renderer:
            textures = model_renderer.texture_renderer.list_textures()
            print(f"\n📋 Texturas disponibles ({len(textures)}):")
            for i, tex in enumerate(textures, 1):
                active = "*" if tex == model_renderer.texture_renderer.active_texture_name else " "
                print(f"   {i:2}. [{active}] {tex}")
    elif key == ord('+') or key == ord('='):  # Tecla + o =
        if model_renderer:
            model_renderer.increase_brightness(0.05)
            print(f"💡 Brillo: {model_renderer.brightness:.2f}")
    
    elif key == ord('-') or key == ord('_'):  # Tecla -
        if model_renderer:
            model_renderer.decrease_brightness(0.05)
            print(f"💡 Brillo: {model_renderer.brightness:.2f}")
    
    elif key == ord('0'):  # Resetear brillo
        if model_renderer:
            model_renderer.set_brightness(0.2)
            print(f"💡 Brillo reseteado: {model_renderer.brightness:.2f}")
    
    elif key == ord('h'):  # Mostrar ayuda
        print("\n" + "="*60)
        print("🎮 CONTROLES DEL TECLADO")
        print("="*60)
        print("TEXTURAS:")
        print("  1 - Textura negra")
        print("  2 - Textura blanca")
        print("  4 - Textura roja")
        print("\nMODOS DE RENDERIZADO:")
        print("  T - Modo textured (con texturas)")
        print("  W - Modo wireframe (alambre)")
        print("  V - Activar modo 3D")
        print("  U - Activar modo 2D")
        print("\nBRILLO:")
        print("  + / = - Aumentar brillo")
        print("  -     - Disminuir brillo")
        print("  0     - Resetear brillo (0.2)")
        print("\nOTROS:")
        print("  L     - Listar texturas disponibles")
        print("  H     - Mostrar esta ayuda")
        print("  ESC   - Salir")
        print("="*60 + "\n")

def render_shirt_adaptive(model_renderer, torso_width, torso_height, render_mode):
    """
    Renderiza la camisa con la transformación actual
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
    
    # Si se solicita listar texturas, hacerlo y salir
    if args.list_textures:
        list_available_textures(args.textures_dir)
        return

    # Configurar cámara
    DESIRED_WIDTH = 1280
    DESIRED_HEIGHT = 720
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar imagen inicial")
        return

    FRAME_H, FRAME_W = frame.shape[:2]
    
    cv2.namedWindow("Detección de Pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detección de Pose", FRAME_W, FRAME_H)

    # Cargar modelo 3D
    obj_loaded = None
    model_renderer = None
    
    # Cargar imagen 2D para modo de respaldo
    shirt_png = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
    if shirt_png is None:
        print(f"⚠️  No se pudo cargar la imagen 2D")
        shirt_png = None

    try:
        print(f"\n📦 Cargando modelo 3D desde: {obj_path}")
        obj_loaded = ObjModel.load_obj(obj_path)
        
        if obj_loaded is not None:
            # Verificar si el modelo tiene coordenadas UV
            has_uvs = obj_loaded.has_texture_coordinates()
            print(f"   ✅ Modelo cargado: {len(obj_loaded.vertices)} vértices")
            print(f"   🎨 Coordenadas UV: {'SÍ' if has_uvs else 'NO'}")
            
            if not has_uvs and args.render_mode == "textured":
                print(f"   ⚠️  Modo 'textured' no disponible sin coordenadas UV")
                args.render_mode = "wireframe"
            
            # Calcular tamaño de render
            render_width = max(512, min(FRAME_W, 1024))
            render_height = max(512, min(FRAME_H, 1024))
            
            # Crear renderizador
            model_renderer = ModelRenderer(width=render_width, height=render_height, obj=obj_loaded)
            model_renderer.set_render_mode(args.render_mode)
            
            # Cargar texturas SOLO desde archivos
            textures_loaded = load_textures(model_renderer, args.textures_dir)
            
            # Verificar si se cargaron texturas
            if textures_loaded:
                # Si se especificó una textura, intentar usarla
                if args.texture and args.texture in textures_loaded:
                    model_renderer.texture_renderer.set_active_texture(args.texture)
                    print(f"   🎯 Textura inicial: {args.texture}")
                else:
                    # Si no se especificó o no está disponible, usar la primera
                    if args.texture:
                        print(f"   ⚠️  Textura '{args.texture}' no encontrada")
                    first_texture = textures_loaded[0]
                    model_renderer.texture_renderer.set_active_texture(first_texture)
                    print(f"   🎯 Textura inicial (primera disponible): {first_texture}")
                
                # Mostrar texturas disponibles
                print(f"\n📋 Texturas cargadas ({len(textures_loaded)}):")
                for tex in textures_loaded:
                    active = "*" if tex == model_renderer.texture_renderer.active_texture_name else " "
                    print(f"   [{active}] {tex}")
            else:
                print(f"   ⚠️  No se cargaron texturas desde archivos")
                print(f"   ⚠️  Asegúrate de que los archivos existan en: {args.textures_dir}")
                
                if args.render_mode == "textured":
                    print(f"   ⚠️  Cambiando a modo 'wireframe' (sin texturas)")
                    args.render_mode = "wireframe"
                    model_renderer.set_render_mode("wireframe")
        else:
            print("❌ No se pudo cargar el modelo 3D")
            args.use_3d = False

    except Exception as e:
        logger.error(f"Error inicializando 3D: {e}")
        print(f"❌ Error: {e}")
        args.use_3d = False
        model_renderer = None

    # Cache para modo 2D
    cached_shirt = {
        "image": None,
        "angle": None,
        "size": None
    }

    # Inicializar MediaPipe POSE
    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            prev_time = cv2.getTickCount()
            fps = 0
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Fin del video o no se pudo leer.")
                    break

                frame_count += 1
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

                    if visible_shoulders and visible_hips:
                        # Calcular coordenadas del torso
                        left_shoulder_x = int(landmarks[SHOULDER_LEFT].x * w)
                        left_shoulder_y = int(landmarks[SHOULDER_LEFT].y * h)
                        right_shoulder_x = int(landmarks[SHOULDER_RIGHT].x * w)
                        right_shoulder_y = int(landmarks[SHOULDER_RIGHT].y * h)
                        left_hip_x = int(landmarks[HIP_LEFT].x * w)
                        left_hip_y = int(landmarks[HIP_LEFT].y * h)

                        # Calcular dimensiones
                        torso_width = int(math.sqrt(
                            (right_shoulder_x - left_shoulder_x) ** 2 +
                            (right_shoulder_y - left_shoulder_y) ** 2
                        ) * 1.7)

                        torso_height = int(math.sqrt(
                            (left_hip_x - left_shoulder_x) ** 2 +
                            (left_hip_y - left_shoulder_y) ** 2
                        ) * 1.5)

                        torso_width = max(torso_width, 100)
                        torso_height = max(torso_height, 150)

                        # Calcular centro
                        torso_center_x = int((left_shoulder_x + right_shoulder_x + left_hip_x) / 3)
                        torso_center_y = int((left_shoulder_y + right_shoulder_y + left_hip_y) / 3)

                        x = torso_center_x - torso_width // 2
                        y = torso_center_y - torso_height // 2

                        angle = math.degrees(math.atan2(
                            right_shoulder_y - left_shoulder_y,
                            right_shoulder_x - left_shoulder_x
                        )) + 180

                        if args.use_3d and model_renderer:
                            # MODO 3D
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

                                # Calcular escala
                                torso_size_pixels = max(torso_width, torso_height)
                                scale_multiplier = torso_size_pixels / 400.0

                                # Alinear modelo
                                rotation, scale, translation = model_renderer.align_model_with_landmarks(
                                    landmarks_3d, 
                                    scale_multiplier=scale_multiplier
                                )

                                # Aplicar transformación y renderizar
                                if rotation is not None and translation is not None:
                                    model_renderer.set_model_transform(translation, rotation, scale)
                                    
                                    img_3d = render_shirt_adaptive(
                                        model_renderer,
                                        torso_width,
                                        torso_height,
                                        args.render_mode
                                    )

                                    if img_3d is not None:
                                        frame = twoD_Render.overlay_transparent(frame, img_3d, x, y)
                        else:
                            # MODO 2D
                            if shirt_png is not None:
                                # Usar cache para mejor rendimiento
                                if (cached_shirt["angle"] != angle or 
                                    cached_shirt["size"] != (torso_width, torso_height)):
                                    
                                    resized = cv2.resize(shirt_png, (torso_width, torso_height))
                                    M = cv2.getRotationMatrix2D((torso_width // 2, torso_height // 2), angle, 1.0)
                                    rotated = cv2.warpAffine(resized, M, (torso_width, torso_height),
                                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
                                    cached_shirt["image"] = rotated
                                    cached_shirt["angle"] = angle
                                    cached_shirt["size"] = (torso_width, torso_height)
                                else:
                                    rotated = cached_shirt["image"]

                                frame = twoD_Render.overlay_transparent(frame, rotated, x, y)

                    # Dibujar landmarks (modo debug)
                    if args.debug and results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                           frame,
                           results.pose_landmarks,
                           mp_pose.POSE_CONNECTIONS,
                           mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                           mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )

                # Mostrar información en pantalla
                print_info(frame=frame, fps=fps, model_renderer=model_renderer)

                cv2.imshow("Detección de Pose", frame)

                # Manejar teclado
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break

                read_keyboard(key, model_renderer=model_renderer, has_uvs=obj_loaded.has_texture_coordinates() if obj_loaded else False)
                    
    except KeyboardInterrupt:
        print("\n🛑 Interrupción por teclado")
    except Exception as e:
        logger.error(f"Error en el bucle principal: {e}")
        print(f"❌ Error: {e}")
    finally:
        # Limpiar recursos
        print("\n🧹 Limpiando recursos...")
        if model_renderer is not None:
            model_renderer.cleanup()
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Programa finalizado correctamente")


    
def main():
    print("""
    ========================================
    🎮 SISTEMA DE REALIDAD AUMENTADA 3D
    ========================================
    """)
    
    # Mostrar ayuda de uso
    print("Uso: python main.py [OPCIONES]")
    print("\nEjemplos:")
    print("  python main.py --use-3d --render-mode textured --texture shirt_black")
    print("  python main.py --list-textures  # Ver texturas disponibles")
    print("\nPara más opciones: python main.py --help")
    print("=" * 60)
    
    mediaPipeRender()

if __name__ == "__main__":
    main()