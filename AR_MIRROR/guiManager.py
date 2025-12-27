
import cv2
import logging
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap

from utils import twoD_Render
from utils.app_args import args
from utils.poseSmoother import PoseSmoother
from guiManagerDesigner import ARAppDesigner
from properties.properties import properties
from utils.modelRenderer import ModelRenderer
#from utils.twoD_Render import draw_silhouette

# Constantes
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
HIP_LEFT = 23
HIP_RIGHT = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
RIGHT_ANKLE = 28    
LEFT_HEEL = 29        
RIGHT_HEEL = 30      
LEFT_TOES = 31  
RIGHT_TOES = 32 


logger = logging.getLogger(__name__)
class ARApp(ARAppDesigner):
    
    def __init__(self, cap, mp_pose, mp_drawing, pose_connections, obj_loaded):

        super().__init__()

        # Variables de entrada
        self.cap = cap
        self.mp_pose = mp_pose
        self.obj_loaded = obj_loaded
        self.silhouette = properties.resources.silhouette
        self.textures_dir = properties.resources.textures_dir
        self.twoD_shirt_path = properties.resources.twoD_shirt_path
        
        # Variables de aplicación
        self.model_renderer = None
        self.twoD_renderer = None
        self.textures_loaded = []
        self.shirt_png = None
        self.running = True
        
        # Variables de control
        self.current_brightness = 0.0
        self.current_y_offset = -0.3
        self.current_x_offset = -0.1
        self.current_shoulder_ref = 0.25

        self.reference_y_bottom = 1
        self.reference_y_top = 0.7
        self.enable_reference_axis = False

        if args.debug:
            self.enable_reference_axis = True

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
        self.shirt_png = cv2.imread(self.twoD_shirt_path, cv2.IMREAD_UNCHANGED)
        self.silhouette_png = cv2.imread(self.silhouette, cv2.IMREAD_UNCHANGED)
        
        # Iniciar bucle de video
        self.update_video()
    
    def keyPressEvent(self, event):

        if event.isAutoRepeat():
            event.ignore()
            return

        key = event.key()
    
        if key == Qt.Key_R:
            self.reset_controls()
            logger.info("Controles reseteados")
            event.accept()
        
        elif key == Qt.Key_F:
            properties.settings.Fullscreen = not properties.settings.Fullscreen
            if properties.settings.Fullscreen:
                self.showFullScreen()
                QApplication.processEvents()
            else:
                self.showNormal()
                #self.resize(properties.WINDOW_WIDTH, properties.WINDOW_HEIGHT)
                QApplication.processEvents()

            event.accept()
        # Si se presiona la tecla 'Escape', cerrar la aplicación
        elif key == Qt.Key_Escape:
            self.on_closing()
            event.accept()
        # Teclas para ajustar brillo
        elif key == Qt.Key_Left:
            self.current_brightness = max(0.0, self.current_brightness - 0.05)
            self.update_brightness(self.current_brightness)
            logger.info(f"Brillo: {self.current_brightness:.2f}")
            event.accept()
        elif key == Qt.Key_Right:
            self.current_brightness = min(1.0, self.current_brightness + 0.05)
            self.update_brightness(self.current_brightness)
            logger.info(f"Brillo: {self.current_brightness:.2f}")
            event.accept()
        
        # También podemos manejar combinaciones de teclas, por ejemplo Ctrl+R
        #elif key == Qt.Key_R and (event.modifiers() & Qt.ControlModifier):
        #    print("Ctrl+R presionado")
        
        # Llamar al método de la clase base para manejar otros eventos estándar
        super().keyPressEvent(event)

    def set_render_mode(self, mode):
        if self.model_renderer:
            self.model_renderer.set_render_mode(mode)
            args.render_mode = mode
            self.info_label.SetText(f"Modo: {mode.upper()}", foreground="green")
    
    def update_brightness(self, value):
        self.current_brightness = float(value)
        #self.brightness_label.setText(f"{self.current_brightness:.2f}")
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

        logger.info(f" Cargando modelo 3D")

        if self.obj_loaded is not None:
            has_uvs = self.obj_loaded.has_texture_coordinates()
            if args.debug:
                logger.debug(f"    Modelo cargado: {len(self.obj_loaded.vertices)} vértices")
                logger.debug(f"    Coordenadas UV: {'Sí' if has_uvs else 'NO'}")

            FRAME_W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            FRAME_H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            render_width = max(720, min(FRAME_W, 1024))
            render_height = max(720, min(FRAME_H, 1024))

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
        
        logger.info(f" Cargando texturas desde archivos...")

        texture_pairs = [
            ("new_shirt", "new_shirt_bake_diffuse.png", "new_shirt_bake_normals.png"),
            ("shirt_spidey", "shirt_spidey_diffuse.png", "new_shirt_bake_normals.png"),
            ("shirt_new_order", "shirt_diffuse_plc.png", "new_shirt_bake_normals.png"),
        ]
        
        for tex_name, diffuse_file, normal_file in texture_pairs:
            import os
            diffuse_path = os.path.join(self.textures_dir, diffuse_file)
            normal_path = os.path.join(self.textures_dir, normal_file)
            
            if os.path.exists(diffuse_path):
                if self.model_renderer.texture_renderer.load_texture(tex_name, diffuse_path, "diffuse"):
                    #print(f" Textura difusa '{tex_name}' cargada")
                    #
                    #if os.path.exists(normal_path):
                    #    if self.model_renderer.texture_renderer.load_texture(tex_name, normal_path, "normal"):
                    #        print(f"    Mapa de normales cargado")
                    
                    textures_loaded.append(tex_name)
        
        return textures_loaded
    def update_video(self):

        ret, frame = self.cap.read()
        if not ret:
            self.info_label.setText(" No se pudo capturar video")
            self.info_label.setStyleSheet("color: red; font-weight: bold;")
            return
        
        if properties.CAMERA_ID == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        frame_mp = frame.copy()
        self.frame_count += 1
        h, w = frame_mp.shape[:2]
        
        # Calcular FPS
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.prev_time) / cv2.getTickFrequency()
        if time_diff > 0:
            self.fps = 1 / time_diff
        self.prev_time = current_time

        # Dibujar línea de referencia si está habilitada
        if self.enable_reference_axis:

            line_x_top = int(self.reference_y_top * h)
            cv2.line(frame, (0, line_x_top), (w, line_x_top), (125, 200, 130), 2)  
            cv2.putText(frame, f"Ref Line Top: {self.reference_y_top:.2f}", 
                       (w - 200, line_x_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 255), 1)
        
            line_x_bottom = int(self.reference_y_bottom * h)
            cv2.line(frame, (0, line_x_bottom), (w, line_x_bottom), (0, 255, 255), 2)  
            cv2.putText(frame, f"Ref Line Bottom: {self.reference_y_bottom:.2f}", 
                       (w - 200, line_x_bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 255), 1)
            
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

            left_ankle_visible = landmarks[LEFT_ANKLE].visibility > 0.8
            right_ankle_visible = landmarks[RIGHT_ANKLE].visibility > 0.8

            left_heel_visible = landmarks[LEFT_HEEL].visibility > 0.8
            right_heel_visible = landmarks[RIGHT_HEEL].visibility > 0.8

            left_toes_visible = landmarks[LEFT_TOES].visibility > 0.8
            right_toes_visible = landmarks[RIGHT_TOES].visibility > 0.8

            
            visible_shoulders = left_shoulder_visible and right_shoulder_visible
            visible_hips = left_hip_visible and right_hip_visible
            
            visible_ankles = left_ankle_visible and right_ankle_visible
            visible_heels = left_heel_visible and right_heel_visible
            visible_toes = left_toes_visible and right_toes_visible

            visible_feets = visible_ankles and visible_heels and visible_toes

            visible_body = visible_shoulders and visible_hips and visible_feets

            left_ankle_y = int(landmarks[LEFT_ANKLE].y * h)
            right_ankle_y = int(landmarks[RIGHT_ANKLE].y * h)

            left_heel_y = int(landmarks[LEFT_HEEL].y * h)
            right_heel_y = int(landmarks[RIGHT_HEEL].y * h)

            left_toes_y = int(landmarks[LEFT_TOES].y * h)
            right_toes_y = int(landmarks[RIGHT_TOES].y * h)

            ankle_avg_y = (left_ankle_y + right_ankle_y) / 2
            heels_avg_y = (left_heel_y + right_heel_y) / 2
            toes_avg_y = (left_toes_y + right_toes_y) / 2

            line_x_top_px = int(self.reference_y_top * h)
            line_x_bottom_px = int(self.reference_y_bottom * h)
            
            #logger.warning(f"\n line_x_top_px: {line_x_top_px} \n line_x_bottom_px: {line_x_bottom_px}")

            #logger.warning(f"\n ankle_avg_y: {ankle_avg_y} \n heels_avg_y: {heels_avg_y} \n toes_avg_y: {toes_avg_y}")

            # Verificar si los pies están dentro del rango

            #####       EJE - Y : AUMENTA HACIA ABAJO
            ankles_in_range = True if ((ankle_avg_y > line_x_top_px ) and(ankle_avg_y < line_x_bottom_px)) else False
            heels_in_range = True if ((heels_avg_y > line_x_top_px ) and(heels_avg_y < line_x_bottom_px)) else False
            toes_in_range = True if ((toes_avg_y > line_x_top_px ) and(toes_avg_y < line_x_bottom_px)) else False

            feets_in_range = ankles_in_range and heels_in_range and toes_in_range
            
            properties.should_project = feets_in_range and visible_body

            #logger.warning(f"Should project; {properties.should_project} \n feets_in_range {feets_in_range} \n visible_body {visible_body}")

            should_project = properties.should_project
            #logger.warning(f"\nankles_in_range: {ankles_in_range} \nheels_in_range: {heels_in_range} \n toes_in_range: {toes_in_range}\nfeets_in_range: {feets_in_range} \n should_project: {should_project}")
            # Mostrar información de debug
            status = "EN RANGO" if should_project else "FUERA DE RANGO"
            color = (0, 255, 0) if should_project else (0, 0, 255)
            cv2.putText(frame, f"Estado: {status}", (10, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # rellenar la zona entre líneas 
            if self.enable_reference_axis:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, line_x_top_px), (w, line_x_bottom_px), 
                             (0, 100, 0), -1)  # Relleno verde semitransparente
                alpha = 0.1  # Transparencia
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            if not properties.should_project:
                #self._hide_gif_overlay()
   
                #self.show_silhouette_overlay()
                
                # Mostrar información de debug
                status = "EN RANGO" if should_project else "        FUERA DE RANGO \n POR FAVOR POSICIONATE EN LA SILUETA"
                color = (0, 255, 0) if should_project else (0, 0, 255)
                self.info_label.setText(f" {status} ")
                self.info_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
            else:      
                #if visible_shoulders and visible_hips and self.model_renderer:
                self.show_silhouette_overlay()
                self._hide_gif_overlay()

                if visible_body and should_project and self.model_renderer:
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
                                logger.warning(f" Error al superponer playera: {e}")
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

        # Obtener el tamaño del QLabel
        label_width = self.video_label.width()
        label_height = self.video_label.height()

        # ESCALAR PARA OCUPAR TODO EL ESPACIO (sin mantener aspecto)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        self.video_label.setPixmap(scaled_pixmap)

    def _render_shirt(self, torso_width, torso_height):
        if self.model_renderer is None:
            return None

        self.model_renderer.set_viewport(torso_width, torso_height)
        self.model_renderer.set_render_mode(args.render_mode)

        raw_frame = self.model_renderer.render_to_image()
        if raw_frame is None:
            return None
        
        rgba = twoD_Render.opengl_to_transparent_rgba(raw_frame)
        return rgba
    
    def show_silhouette_overlay(self):
        pixmap = QPixmap(properties.resources.silhouette)
        self.silhouette_label.setPixmap(pixmap)
        self.silhouette_label.raise_()   
        self.silhouette_label.show()

    def on_closing(self):

        self.closeWindowSignal.emit()
        logger.info("Cerrando aplicación...")
        self.running = False

        # Detener el timer de actualización de video
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        # Cerrar ventana de debug si existe
        if hasattr(self, 'debug_window') and self.debug_window is not None:
            self.debug_window.close()
            self.debug_window = None

        # Limpiar recursos de MediaPipe si existe y no está ya cerrado
        if hasattr(self, 'mp_pose') and self.mp_pose is not None:
            try:
                self.mp_pose.close()
            except Exception as e:
                logger.warning(f"Error al cerrar MediaPipe: {e}")
            finally:
                self.mp_pose = None

        # Liberar la cámara si existe
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

        # Limpiar el renderizador 3D si existe
        if hasattr(self, 'model_renderer') and self.model_renderer is not None:
            try:
                self.model_renderer.cleanup()
            except Exception as e:
                logger.warning(f"Error al limpiar el renderizador: {e}")
            finally:
                self.model_renderer = None

        logger.info("Aplicación cerrada correctamente")
        # Cerrar la ventana principal
        self.close()
