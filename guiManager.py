"""
guiManager.py - Interfaz Tkinter para AR Shirt
Contiene la clase ARShirtApp que maneja toda la interfaz gráfica
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

from utils.app_args import args
from utils.modelRenderer import ModelRenderer
from utils import twoD_Render

# Constantes
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
HIP_LEFT = 23
HIP_RIGHT = 24


class ARShirtApp:
    """Clase principal de la interfaz Tkinter para AR Shirt"""
    
    def __init__(self, root, cap, mp_pose, obj_loaded, textures_dir, shirt_path):
        """
        Inicializa la aplicación
        
        Args:
            root: Ventana Tkinter principal
            cap: Capturador de video (cv2.VideoCapture)
            mp_pose: Objeto MediaPipe Pose
            obj_loaded: Modelo 3D cargado
            textures_dir: Directorio de texturas
            shirt_path: Ruta de la imagen 2D de la playera
        """
        self.root = root
        self.root.title("🎽 AR Shirt - Control Panel")
        self.root.geometry("1400x900")
        
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
        
        # Crear interfaz
        self.create_ui()
        
        # Cargar modelo
        self.load_model()
        
        # Cargar imagen 2D
        self.shirt_png = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
        
        # Iniciar bucle de video
        self.update_video()
        
        # Manejar cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    # ============================================================
    # CREAR INTERFAZ
    # ============================================================
    
    def create_ui(self):
        """Crea la interfaz Tkinter"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para video (izquierda)
        video_frame = ttk.LabelFrame(main_frame, text="📹 Vista Previa", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg='black', width=640, height=480)
        self.video_label.pack()
        
        # Frame para controles (derecha)
        control_frame = ttk.LabelFrame(main_frame, text="⚙️ Controles", padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Agregar scrollbar para controles
        canvas = tk.Canvas(control_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # ========== MODOS DE RENDERIZADO ==========
        ttk.Label(scrollable_frame, text="📐 Modo Renderizado:", font=('Arial', 10, 'bold')).pack()
        
        mode_frame = ttk.Frame(scrollable_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(mode_frame, text="🎨 Textured", 
                  command=lambda: self.set_render_mode('textured')).pack(side=tk.LEFT, padx=2)
        ttk.Button(mode_frame, text="📈 Wireframe", 
                  command=lambda: self.set_render_mode('wireframe')).pack(side=tk.LEFT, padx=2)
        ttk.Button(mode_frame, text="⚫ Solid", 
                  command=lambda: self.set_render_mode('solid')).pack(side=tk.LEFT, padx=2)
        
        # ========== TEXTURAS ==========
        ttk.Label(scrollable_frame, text="🎭 Cambiar Textura:", font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        
        texture_frame = ttk.Frame(scrollable_frame)
        texture_frame.pack(fill=tk.X, pady=5)
        
        self.texture_var = tk.StringVar()
        self.texture_combo = ttk.Combobox(texture_frame, textvariable=self.texture_var, state='readonly')
        self.texture_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(texture_frame, text="Aplicar", 
                  command=self.apply_texture).pack(side=tk.LEFT)
        
        # ========== BRILLO ==========
        ttk.Label(scrollable_frame, text="💡 Brillo:", font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        
        brightness_frame = ttk.Frame(scrollable_frame)
        brightness_frame.pack(fill=tk.X, pady=5)
        
        self.brightness_slider = ttk.Scale(brightness_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                          command=self.update_brightness)
        self.brightness_slider.set(0.2)
        self.brightness_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.brightness_label = ttk.Label(brightness_frame, text="0.20", width=5)
        self.brightness_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # ========== POSICIÓN Y ==========
        ttk.Label(scrollable_frame, text="📍 Posición Y:", font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        
        y_frame = ttk.Frame(scrollable_frame)
        y_frame.pack(fill=tk.X, pady=5)
        
        self.y_slider = ttk.Scale(y_frame, from_=-1, to=1, orient=tk.HORIZONTAL,
                                 command=self.update_y_offset)
        self.y_slider.set(-0.3)
        self.y_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.y_label = ttk.Label(y_frame, text="-0.30", width=5)
        self.y_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # ========== POSICIÓN X ==========
        ttk.Label(scrollable_frame, text="📍 Posición X:", font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        
        x_frame = ttk.Frame(scrollable_frame)
        x_frame.pack(fill=tk.X, pady=5)
        
        self.x_slider = ttk.Scale(x_frame, from_=-1, to=1, orient=tk.HORIZONTAL,
                                 command=self.update_x_offset)
        self.x_slider.set(-0.1)
        self.x_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.x_label = ttk.Label(x_frame, text="-0.10", width=5)
        self.x_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # ========== ESCALA ==========
        ttk.Label(scrollable_frame, text="📏 Referencia Hombros (m):", font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        
        scale_frame = ttk.Frame(scrollable_frame)
        scale_frame.pack(fill=tk.X, pady=5)
        
        self.shoulder_slider = ttk.Scale(scale_frame, from_=0.1, to=0.5, orient=tk.HORIZONTAL,
                                        command=self.update_shoulder_ref)
        self.shoulder_slider.set(0.25)
        self.shoulder_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.shoulder_label = ttk.Label(scale_frame, text="0.25", width=5)
        self.shoulder_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # ========== INFORMACIÓN ==========
        self.info_label = ttk.Label(scrollable_frame, text="Cargando...", foreground="blue")
        self.info_label.pack(pady=(15, 5))
        
        # ========== BOTONES FINALES ==========
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(button_frame, text="ℹ️  Ayuda", 
                  command=self.show_help).pack(pady=5)
        ttk.Button(button_frame, text="🔄 Resetear", 
                  command=self.reset_controls).pack(pady=5)
        ttk.Button(button_frame, text="❌ Salir", 
                  command=self.on_closing).pack(pady=5)
    
    # ============================================================
    # EVENTOS DE CONTROLES
    # ============================================================
    
    def set_render_mode(self, mode):
        """Cambia el modo de renderizado"""
        if self.model_renderer:
            self.model_renderer.set_render_mode(mode)
            args.render_mode = mode
            self.info_label.config(text=f"✅ Modo: {mode.upper()}", foreground="green")
    
    def apply_texture(self):
        """Aplica la textura seleccionada"""
        texture = self.texture_var.get()
        if texture and self.model_renderer:
            self.model_renderer.texture_renderer.set_active_texture(texture)
            self.info_label.config(text=f"✅ Textura: {texture}", foreground="green")
    
    def update_brightness(self, value):
        """Actualiza el brillo"""
        self.current_brightness = float(value)
        self.brightness_label.config(text=f"{self.current_brightness:.2f}")
        if self.model_renderer:
            self.model_renderer.brightness = self.current_brightness
    
    def update_y_offset(self, value):
        """Actualiza el offset Y"""
        self.current_y_offset = float(value)
        self.y_label.config(text=f"{self.current_y_offset:.2f}")
    
    def update_x_offset(self, value):
        """Actualiza el offset X"""
        self.current_x_offset = float(value)
        self.x_label.config(text=f"{self.current_x_offset:.2f}")
    
    def update_shoulder_ref(self, value):
        """Actualiza la referencia de hombros"""
        self.current_shoulder_ref = float(value)
        self.shoulder_label.config(text=f"{self.current_shoulder_ref:.2f}")
    
    def reset_controls(self):
        """Resetea todos los controles a valores por defecto"""
        self.brightness_slider.set(0.2)
        self.y_slider.set(-0.3)
        self.x_slider.set(-0.1)
        self.shoulder_slider.set(0.25)
        self.info_label.config(text="✅ Configuración reseteada", foreground="green")
    
    def show_help(self):
        """Muestra la ventana de ayuda"""
        messagebox.showinfo("Ayuda", """
            🎽 AR SHIRT - CONTROLES

            📐 Modo Renderizado:
              - Textured: Con texturas
              - Wireframe: Estructura
              - Solid: Color sólido

            🎭 Texturas:
              - Selecciona en el combo
              - Presiona Aplicar

            💡 Brillo: Ajusta intensidad

            📍 Posición: Mueve X e Y

            📏 Escala: Referencia de hombros
              (valores menores = playera más grande)
                    """)
    
    def load_model(self):
        """Carga el modelo 3D y las texturas"""
        from utils.objectLoader import ObjModel
        
        print(f"\n🎪 Cargando modelo 3D")
        
        if self.obj_loaded is not None:
            has_uvs = self.obj_loaded.has_texture_coordinates()
            print(f"   ✅ Modelo cargado: {len(self.obj_loaded.vertices)} vértices")
            print(f"   🎨 Coordenadas UV: {'Sí' if has_uvs else 'NO'}")
            
            FRAME_W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            FRAME_H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            render_width = max(512, min(FRAME_W, 1024))
            render_height = max(512, min(FRAME_H, 1024))
            
            self.model_renderer = ModelRenderer(width=render_width, height=render_height, obj=self.obj_loaded)
            self.model_renderer.set_render_mode(args.render_mode)
            
            self.textures_loaded = self._load_textures()
            
            if self.textures_loaded:
                if args.texture and args.texture in self.textures_loaded:
                    self.model_renderer.texture_renderer.set_active_texture(args.texture)
                    selected = args.texture
                else:
                    selected = self.textures_loaded[0]
                    self.model_renderer.texture_renderer.set_active_texture(selected)
                
                self.texture_combo['values'] = self.textures_loaded
                self.texture_var.set(selected)
                self.info_label.config(text=f"✅ Modelo cargado", foreground="green")
    
    def _load_textures(self):
        """Carga las texturas disponibles"""
        textures_loaded = []
        
        print(f"\n🎨 Cargando texturas desde archivos...")

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
                    print(f"✅ Textura difusa '{tex_name}' cargada")
                    
                    if os.path.exists(normal_path):
                        if self.model_renderer.texture_renderer.load_texture(tex_name, normal_path, "normal"):
                            print(f"   🌟 Mapa de normales cargado")
                    
                    textures_loaded.append(tex_name)
        
        return textures_loaded
    
    def update_video(self):
        """Actualiza el video en tiempo real"""
        ret, frame = self.cap.read()
        if not ret:
            self.info_label.config(text="❌ No se pudo capturar video", foreground="red")
            if self.running:
                self.root.after(100, self.update_video)
            return
        
        CAMERA_ID = 0
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
            
            left_shoulder_visible = landmarks[SHOULDER_LEFT].visibility > 0.8
            right_shoulder_visible = landmarks[SHOULDER_RIGHT].visibility > 0.8
            left_hip_visible = landmarks[HIP_LEFT].visibility > 0.8
            right_hip_visible = landmarks[HIP_RIGHT].visibility > 0.8
            
            visible_shoulders = left_shoulder_visible and right_shoulder_visible
            visible_hips = left_hip_visible and right_hip_visible
            
            if visible_shoulders and visible_hips and self.model_renderer:
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
                    
                    # Aplicar transformación
                    if rotation is not None and translation is not None:
                        translation[0] += self.current_x_offset
                        translation[1] += self.current_y_offset
                        
                        self.model_renderer.set_model_transform(translation, rotation, scale)
                        
                        img_3d = self._render_shirt(torso_width, torso_height)
                        
                        if img_3d is not None:
                            frame = twoD_Render.overlay_transparent(frame, img_3d, x, y)
        
        # Agregar FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convertir para Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(frame_pil.resize((640, 480)))
        
        self.video_label.config(image=frame_tk)
        self.video_label.image = frame_tk
        
        # Continuar actualizando
        if self.running:
            self.root.after(30, self.update_video)
    
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