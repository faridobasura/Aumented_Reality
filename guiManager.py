
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from utils.app_args import args
from utils.modelRenderer import ModelRenderer
from utils import twoD_Render
from utils.poseSmoother import PoseSmoother
from guiManagerDesigner import ARAppDesigner
# Constantes
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
HIP_LEFT = 23
HIP_RIGHT = 24

class ARApp(ARAppDesigner):
    
    def __init__(self, cap, mp_pose, mp_drawing, pose_connections, obj_loaded, textures_dir, shirt_path):

        super().__init__()

        # Variables de entrada
        self.cap = cap
        self.mp_pose = mp_pose
        self.obj_loaded = obj_loaded
        self.textures_dir = textures_dir
        self.shirt_path = shirt_path
        
        # Variables de aplicación
        self.model_renderer = None
        self.textures_loaded = []
        self.shirt_png = None
        self.running = True
        
        # Variables de control
        self.current_brightness = 0.2
        self.current_y_offset = -0.3
        self.current_x_offset = -0.1
        self.current_shoulder_ref = 0.25
        
        # Variables de video
        self.fps = 0
        self.frame_count = 0
        self.prev_time = cv2.getTickCount()

        self.mp_drawing = mp_drawing
        self.pose_connections = pose_connections

        self.pose_smoother = PoseSmoother(
            window_size=5,      # Promediar últimos 5 frames
            alpha=0.4           # 0.4 = suavizado moderado (ajusta entre 0.2-0.6)
        )
        self.last_valid_landmarks = None
        self.frames_without_pose = 0
        self.max_frames_without_pose = 10  # Máximo de frames sin pose antes de resetear
        
        # Cargar modelo
        self.load_model()
        
        # Cargar imagen 2D
        self.shirt_png = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
        
        # Iniciar bucle de video
        self.update_video()
    
    def set_render_mode(self, mode):
        if self.model_renderer:
            self.model_renderer.set_render_mode(mode)
            args.render_mode = mode
            self.info_label.SetText(f"Modo: {mode.upper()}", foreground="green")
    
    def apply_texture(self, texture=None):
        if texture and self.model_renderer:
            self.model_renderer.texture_renderer.set_active_texture(texture)
            self.info_label.SetText(f" Textura: {texture}", foreground="green")
    
    def update_brightness(self, value):
        self.current_brightness = float(value)
        self.brightness_label.setText(f"{self.current_brightness:.2f}")
        if self.model_renderer:
            self.model_renderer.brightness = self.current_brightness
    
    def update_y_offset(self, value):
        self.current_y_offset = float(value)
        self.y_label.setText(f"{self.current_y_offset:.2f}")    

    def update_x_offset(self, value):
        self.current_x_offset = float(value)
        self.x_label.setText(f"{self.current_x_offset:.2f}")    

    
    def update_shoulder_ref(self, value):
        self.current_shoulder_ref = float(value)
        self.shoulder_label.setText(f"{self.current_shoulder_ref:.2f}")    
    
    def reset_controls(self):
        self.update_brightness(0.2)
        self.update_y_offset(-0.3)
        self.update_x_offset(-0.1)
        self.update_shoulder_ref(0.25)

        self.info_label.setText(" Configuración reseteada")
        self.info_label.setStyleSheet("color: green; font-weight: bold;")

    def load_model(self):
        from utils.objectLoader import ObjModel

        print(f"\n Cargando modelo 3D")

        if self.obj_loaded is not None:
            has_uvs = self.obj_loaded.has_texture_coordinates()
            print(f"    Modelo cargado: {len(self.obj_loaded.vertices)} vértices")
            print(f"    Coordenadas UV: {'Sí' if has_uvs else 'NO'}")

            FRAME_W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            FRAME_H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            render_width = max(512, min(FRAME_W, 1024))
            render_height = max(512, min(FRAME_H, 1024))

            self.model_renderer = ModelRenderer(width=render_width, height=render_height, obj=self.obj_loaded)
            self.model_renderer.set_render_mode(args.render_mode)

            # cargar texturas
            self.textures_loaded = self._load_textures()

            if self.textures_loaded:
                # textura seleccionada por args o default
                if args.texture and args.texture in self.textures_loaded:
                    selected = args.texture
                else:
                    selected = self.textures_loaded[0]

                # Activar textura en el renderer
                self.model_renderer.texture_renderer.set_active_texture(selected)


    
    def _load_textures(self):
        textures_loaded = []
        
        print(f"\n Cargando texturas desde archivos...")

        texture_pairs = [
            ("new_shirt", "pclShirt_without_CLOTHTEXT.png", "new_shirt_bake_normals.png"),
            ("shirt_spidey", "shirt_spidey_diffuse.png", "new_shirt_bake_normals.png"),
            ("shirt_new_order", "shirt_diffuse_plc.png", "new_shirt_bake_normals.png"),
        ]
        
        for tex_name, diffuse_file, normal_file in texture_pairs:
            import os
            diffuse_path = os.path.join(self.textures_dir, diffuse_file)
            normal_path = os.path.join(self.textures_dir, normal_file)
            
            if os.path.exists(diffuse_path):
                if self.model_renderer.texture_renderer.load_texture(tex_name, diffuse_path, "diffuse"):
                    print(f" Textura difusa '{tex_name}' cargada")
                    
                    if os.path.exists(normal_path):
                        if self.model_renderer.texture_renderer.load_texture(tex_name, normal_path, "normal"):
                            print(f"    Mapa de normales cargado")
                    
                    textures_loaded.append(tex_name)
        
        return textures_loaded
    def update_video(self):
        """Actualiza el video en tiempo real"""
        ret, frame = self.cap.read()
        if not ret:
            self.info_label.setText(" No se pudo capturar video")
            self.info_label.setStyleSheet("color: red; font-weight: bold;")
            return
        
        CAMERA_ID = 2
        if CAMERA_ID == 0:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        frame_mp = frame.copy()
        self.frame_count += 1
        h, w = frame_mp.shape[:2]
        
        # Calcular FPS
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.prev_time) / cv2.getTickFrequency()
        if time_diff > 0:
            self.fps = 1 / time_diff
        self.prev_time = current_time
        
        # Procesar con MediaPipe
        rgb = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            if args.debug and results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                   frame,
                   results.pose_landmarks,
                   self.pose_connections,
                   self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                   self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            left_shoulder_visible = landmarks[SHOULDER_LEFT].visibility > 0.8
            right_shoulder_visible = landmarks[SHOULDER_RIGHT].visibility > 0.8
            left_hip_visible = landmarks[HIP_LEFT].visibility > 0.8
            right_hip_visible = landmarks[HIP_RIGHT].visibility > 0.8
            
            visible_shoulders = left_shoulder_visible and right_shoulder_visible
            visible_hips = left_hip_visible and right_hip_visible
            
            if visible_shoulders and visible_hips and self.model_renderer:
                self.frames_without_pose = 0

                # Calcular coordenadas del torso
                left_shoulder_x = int(landmarks[SHOULDER_LEFT].x * w)
                left_shoulder_y = int(landmarks[SHOULDER_LEFT].y * h)
                right_shoulder_x = int(landmarks[SHOULDER_RIGHT].x * w)
                right_shoulder_y = int(landmarks[SHOULDER_RIGHT].y * h)
                left_hip_x = int(landmarks[HIP_LEFT].x * w)
                left_hip_y = int(landmarks[HIP_LEFT].y * h)
                
                # Calcular dimensiones
                import math
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
                
                # MODO 3D
                if getattr(results, "pose_world_landmarks", None):
                    pl = results.pose_world_landmarks.landmark
                    
                    landmarks_3d = {
                        "left_shoulder": [pl[SHOULDER_LEFT].x, pl[SHOULDER_LEFT].y, pl[SHOULDER_LEFT].z],
                        "right_shoulder": [pl[SHOULDER_RIGHT].x, pl[SHOULDER_RIGHT].y, pl[SHOULDER_RIGHT].z],
                        "left_hip": [pl[HIP_LEFT].x, pl[HIP_LEFT].y, pl[HIP_LEFT].z],
                        "right_hip": [pl[HIP_RIGHT].x, pl[HIP_RIGHT].y, pl[HIP_RIGHT].z]
                    }
                    
                    # Calcular escala
                    right_shoulder_3d = np.array([pl[SHOULDER_RIGHT].x, pl[SHOULDER_RIGHT].y, pl[SHOULDER_RIGHT].z])
                    left_shoulder_3d = np.array([pl[SHOULDER_LEFT].x, pl[SHOULDER_LEFT].y, pl[SHOULDER_LEFT].z])
                    shoulder_distance_3d = np.linalg.norm(right_shoulder_3d - left_shoulder_3d)
                    scale_multiplier = shoulder_distance_3d / self.current_shoulder_ref
                    
                    # Alinear modelo
                    rotation, scale, translation = self.model_renderer.align_model_with_landmarks(
                        landmarks_3d, 
                        scale_multiplier=scale_multiplier
                    )
                    
                    # SUAVIZAR TRANSFORMACIÓN
                    if rotation is not None and translation is not None:
                        translation_arr = np.array(translation, dtype=np.float32)
                        rotation_arr = np.array(rotation, dtype=np.float32)
                        
                        # Suavizar
                        translation_arr, rotation_arr, scale = self.pose_smoother.smooth_transform(
                            translation_arr, rotation_arr, scale
                        )
                        
                        # Aplicar offsets
                        translation_arr[0] += self.current_x_offset
                        translation_arr[1] += self.current_y_offset
                        
                        self.model_renderer.set_model_transform(translation_arr, rotation_arr, scale)
                        
                        img_3d = self._render_shirt(torso_width, torso_height)
                        
                        try:
                            if img_3d is not None:
                                frame = twoD_Render.overlay_transparent(frame, img_3d, x, y)
                        except Exception as e:
                            print(f" Error al superponer playera: {e}")
            else:
                # Usar última pose válida si no se detecta
                self.frames_without_pose += 1
                if self.frames_without_pose > self.max_frames_without_pose:
                    self.pose_smoother.reset()
                    self.frames_without_pose = 0
        
        # Agregar FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convertir BGR a RGB para Qt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Mostrar en label
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))
    
    def _render_shirt(self, torso_width, torso_height):
        """Renderiza la playera 3D"""
        if self.model_renderer is None:
            return None

        self.model_renderer.set_viewport(torso_width, torso_height)
        self.model_renderer.set_render_mode(args.render_mode)

        raw_frame = self.model_renderer.render_to_image()
        if raw_frame is None:
            return None
        
        rgba = twoD_Render.opengl_to_transparent_rgba(raw_frame)
        return rgba
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        print("\n🛑 Cerrando aplicación...")
        self.running = False
        if self.model_renderer:
            self.model_renderer.cleanup()
        if self.mp_pose:
            self.mp_pose.close()
        self.cap.release()
        self.root.destroy()