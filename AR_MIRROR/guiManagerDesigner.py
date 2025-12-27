from PyQt5.QtGui import QMovie, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLabel, QPushButton, QSlider, QGroupBox, 
                              QScrollArea, QMessageBox, QFrame, QGridLayout)
import cv2
import os

import logging
from utils.app_args import args
from properties.properties import properties
from properties.config import OUT_OF_RANGE_GIF
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
HIP_LEFT = 23
HIP_RIGHT = 24

logger = logging.getLogger(__name__)

class DebugWindow(QMainWindow):

    def __init__(self, controller):
        super().__init__()

        self.controller = controller 

        self.setWindowTitle("Controles Debug")
        self.setGeometry(int(properties.WINDOW_WIDTH*0.35), int(properties.WINDOW_HEIGHT*0.15), properties.WINDOW_WIDTH, int(properties.WINDOW_HEIGHT*0.35))

        self.controller.closeWindowSignal.connect(self.close)
        central = QWidget()
        layout = QVBoxLayout(central)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: green; font-weight: bold;")

        self.controller.create_control_section(layout)

        self.setCentralWidget(central)

    def create_control_section(self, layout):
        btn = QPushButton("Reset Tracking")
        btn.clicked.connect(self.controller.reset_tracking)
        layout.addWidget(btn)
class ARAppDesigner(QMainWindow):

    closeWindowSignal = pyqtSignal()
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Espejo AR")

        self.gif_overlay = None
        self.gif_frames = []
        self.current_gif_frame_index = 0
        self.gif_timer = QTimer()
        self.gif_timer.timeout.connect(self._update_gif_frame)
        self.gif_path = None  # Ruta al GIF que usarás

        
        self.load_out_of_range_gif(OUT_OF_RANGE_GIF)  

        if not properties.settings.Fullscreen:
            self.resize(properties.WINDOW_WIDTH, properties.WINDOW_HEIGHT)
            self.move(100, 100)  # Posición fija inicial
        
        # Crear interfaz
        self.init_ui()

        # Esperar un momento antes de iniciar timer
        QTimer.singleShot(100, self.start_video_timer)

        if args.debug:
            QTimer.singleShot(200, self._create_debug_window)

    def start_video_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(50) 

    def _create_debug_window(self):
        
        self.debug_window = DebugWindow(controller=self)
        self.debug_window.show()
        logger.info("Ventana debug abierta")
        QTimer.singleShot(10, lambda: [
            self.raise_(),
            self.activateWindow()
        ])
    
    def init_ui(self):
        self.setWindowTitle("Espejo AR")

        self.setGeometry(int(properties.WINDOW_WIDTH*0.45), int(properties.WINDOW_HEIGHT*0.9), properties.WINDOW_WIDTH, properties.WINDOW_HEIGHT)
        
        # Layout principal horizontal
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        self.info_label = QLabel(" ESPEJO AR ")
        self.info_label.setStyleSheet("color: blue; font-weight: bold;")

        self.gif_overlay_label = QLabel(self)
        self.gif_overlay_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: none;
            }
        """)
        self.gif_overlay_label.setScaledContents(True)
        #self.gif_overlay_label.setAttribute(Qt.WA_TransparentForMouseEvents)  # Permitir clicks a través del label
        self.gif_overlay_label.setGeometry(0, 0, self.width(), self.height())
        self.gif_overlay_label.lower()  # Enviar atrás inicialmente
        self.gif_overlay_label.hide()
        
        self.silhouette_label = QLabel(self)
        self.silhouette_label.setStyleSheet("background: transparent;")
        self.silhouette_label.setGeometry(0, 0, int(properties.WINDOW_WIDTH*0.15),int(properties.WINDOW_HEIGHT*0.86))
        self.silhouette_label.setScaledContents(True)
        self.silhouette_label.move(int(properties.WINDOW_WIDTH*0.42),int(properties.WINDOW_HEIGHT*0.1))
        self.silhouette_label.hide()

        # === PANEL central: VIDEO ===
        central_panel_widget = QWidget()
        central_panel_layout = QVBoxLayout(central_panel_widget)

        self.video_container = QGroupBox("Video en Vivo")
        video_inner_layout = QVBoxLayout(self.video_container)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(int(properties.WINDOW_WIDTH*0.625), int(properties.WINDOW_HEIGHT*0.9))
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setScaledContents(True)
        video_inner_layout.addWidget(self.video_label)
        central_panel_layout.addWidget(self.video_container)              
        central_panel_layout.addStretch()

        ## Panel izquierdo
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)

        # Scroll area para controles
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(int(properties.WINDOW_WIDTH*0.25))
        left_scroll.setMinimumHeight(int(properties.WINDOW_HEIGHT*0.9))
        
        scroll_widget_left_area = QWidget()
        scroll_layout_left_area = QVBoxLayout(scroll_widget_left_area)

        self.create_catalog_section(scroll_layout_left_area)

        scroll_layout_left_area.addStretch()
        left_scroll.setWidget(scroll_widget_left_area)
        left_panel_layout.addWidget(left_scroll)

        ## Panel derecho

        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)

        # Scroll area para controles
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(int(properties.WINDOW_WIDTH*0.25))
        right_scroll.setMinimumHeight(int(properties.WINDOW_HEIGHT*0.9))
        
        scroll_widget_right_area = QWidget()
        scroll_layout_right_area = QVBoxLayout(scroll_widget_right_area)

        self.create_catalog_section(scroll_layout_right_area)

        scroll_layout_right_area.addStretch()
        right_scroll.setWidget(scroll_widget_right_area)
        right_panel_layout.addWidget(right_scroll)

        main_layout.addWidget(left_panel_widget)
        main_layout.addWidget(central_panel_widget)
        main_layout.addWidget(right_panel_widget)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        if args.debug:
            self.setStyleSheet("border: 2px solid black;")
    
    def create_catalog_section(self, layout):

        title = QLabel("Cambiar Modelo:")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)

        catalog_container = QWidget()
        grid = QGridLayout(catalog_container)
        grid.setSpacing(10)

        self.catalog = [
            {
                "name": "Playera Blanca",
                "texture": "new_shirt",
                "image": properties.resources.shirt_white
            },
            {
                "name": "Playera Spiderman",
                "texture": "shirt_spidey",
                "image": properties.resources.shirt_spidey
            },
            {
                "name": "Playera New Order",
                "texture": "shirt_new_order",
                "image": properties.resources.shirt_pcl
            }
        ]

        for index, item in enumerate(self.catalog):

            btn = QPushButton()
            btn.setToolTip(item["name"])

            image_path = str(item["image"])
            pixmap = QPixmap(image_path)
            
            if pixmap.isNull():
                logger.warning(f" No se pudo cargar: {item['image']}")

            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(QSize(int(properties.WINDOW_WIDTH*0.1), int(properties.WINDOW_HEIGHT*0.2)))

            btn.clicked.connect(lambda checked=False, tex=item["texture"]: self.apply_texture(tex))
            
            grid.setColumnStretch(0, 1)
            grid.addWidget(btn, index, 0)

        layout.addWidget(catalog_container)

    def create_control_section(self, layout):


        
        self.create_brightness_section(layout)
        
        self.create_y_position_section(layout)
        
        self.create_x_position_section(layout)
        
        self.create_scale_section(layout)
        
        self.create_action_buttons(layout)
    
    def create_brightness_section(self, layout):

        title = QLabel(" Brillo:")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        slider_layout = QHBoxLayout()
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(0)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(20)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        
        self.brightness_label = QLabel("0.20")
        self.brightness_label.setMinimumWidth(40)
        
        slider_layout.addWidget(self.brightness_slider)
        slider_layout.addWidget(self.brightness_label)
        layout.addLayout(slider_layout)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
    
    def create_y_position_section(self, layout):

        title = QLabel(" Posición Y:")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        slider_layout = QHBoxLayout()
        
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setMinimum(-100)
        self.y_slider.setMaximum(100)
        self.y_slider.setValue(-30)
        self.y_slider.valueChanged.connect(self.update_y_offset)
        
        self.y_label = QLabel("-0.30")
        self.y_label.setMinimumWidth(40)
        
        slider_layout.addWidget(self.y_slider)
        slider_layout.addWidget(self.y_label)
        layout.addLayout(slider_layout)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
    
    def create_x_position_section(self, layout):

        title = QLabel(" Posición X:")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        slider_layout = QHBoxLayout()
        
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(-100)
        self.x_slider.setMaximum(100)
        self.x_slider.setValue(-10)
        self.x_slider.valueChanged.connect(self.update_x_offset)
        
        self.x_label = QLabel("-0.10")
        self.x_label.setMinimumWidth(40)
        
        slider_layout.addWidget(self.x_slider)
        slider_layout.addWidget(self.x_label)
        layout.addLayout(slider_layout)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
    
    def create_scale_section(self, layout):
        title = QLabel(" Referencia Hombros (m):")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        slider_layout = QHBoxLayout()
        
        self.shoulder_slider = QSlider(Qt.Horizontal)
        self.shoulder_slider.setMinimum(10)
        self.shoulder_slider.setMaximum(50)
        self.shoulder_slider.setValue(25)
        self.shoulder_slider.valueChanged.connect(self.update_shoulder_ref)
        
        self.shoulder_label = QLabel("0.25")
        self.shoulder_label.setMinimumWidth(40)
        
        slider_layout.addWidget(self.shoulder_slider)
        slider_layout.addWidget(self.shoulder_label)
        layout.addLayout(slider_layout)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
    
    def create_action_buttons(self, layout):

        help_btn = QPushButton("ℹ  Ayuda")
        help_btn.clicked.connect(self.show_help)
        layout.addWidget(help_btn)
        
        reset_btn = QPushButton(" Resetear")
        reset_btn.clicked.connect(self.reset_controls)
        layout.addWidget(reset_btn)
        
        exit_btn = QPushButton(" Salir")
        exit_btn.clicked.connect(self.on_closing)
        layout.addWidget(exit_btn)
    
    def apply_texture(self, texture):
        if not self.model_renderer:
            return

        if texture not in self.textures_loaded:
            logger.warning(f"Textura no cargada: {texture}")
            return
        
        logger.info(f"Textura aplicada: {texture}")

        self.model_renderer.texture_renderer.set_active_texture(texture)

        #self.info_label.setText(f"Textura activa: {texture}")
        #self.info_label.setStyleSheet("color: green; font-weight: bold;")
    
    def update_brightness(self, value):
        pass

    def update_y_offset(self, value):
        pass
    
    def update_x_offset(self, value):
        pass
    
    def update_shoulder_ref(self, value):
        pass
    
    def reset_controls(self):
        pass
    
    def show_help(self):
        """Muestra la ventana de ayuda"""
        QMessageBox.information(self, "Ayuda", """
                      Espejo AR

                     Modo Renderizado:
                      - Textured: Con texturas
                      - Wireframe: Estructura
                      - Solid: Color sólido

                     Texturas:
                      - Selecciona en los botones
                      - Se aplica automáticamente

                     Brillo: Ajusta intensidad

                     Posición: Mueve X e Y

                     Escala: Referencia de hombros
                      (valores menores = playera más grande)
                            """)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Forzar actualización del video label si existe
        if hasattr(self, 'video_label') and self.video_label:
            # Actualizar el pixmap con el nuevo tamaño
            self.video_label.updateGeometry()

            # Si hay un pixmap actual, escalarlo
            current_pixmap = self.video_label.pixmap()
            if current_pixmap and not current_pixmap.isNull():
                # Escalar al nuevo tamaño del label
                scaled = current_pixmap.scaled(
                    self.video_label.size(),
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled)

        # Si está en modo debug, también redimensionar la ventana debug
        if hasattr(self, 'debug_window') and self.debug_window:
            if not self.isFullScreen():
                # Posicionar ventana debug relativa a la principal
                self.debug_window.move(int(properties.WINDOW_WIDTH*0.9), int(properties.WINDOW_HEIGHT*0.35))

        if hasattr(self, 'gif_overlay_label'):
            window_width = self.width()
            window_height = self.height()
            self.gif_overlay_label.setGeometry(0, 0, window_width, window_height)

        if hasattr(self, 'silhouette_label'):
            #self.gif_overlay_label.setGeometry(0, 0, window_width, window_height)
            window_width = self.width()
            window_height = self.height()
            #self.silhouette_label.setGeometry(0, 0, int(window_width*0.35),int(window_height*0.98))


    def load_out_of_range_gif(self, gif_path):
        try:
            # Expandir ~ a la ruta home del usuario
            gif_path = os.path.expanduser(gif_path)

            if not os.path.exists(gif_path):
                logger.warning(f"Archivo GIF no encontrado: {gif_path}")
                return

            cap = cv2.VideoCapture(gif_path)

            self.gif_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Se redimensionará dinámicamente según el tamaño de la ventana
                self.gif_frames.append(frame)

            cap.release()

            if self.gif_frames:
                logger.info(f"GIF cargado: {len(self.gif_frames)} frames")
                self.current_gif_frame_index = 0
                # Iniciar timer para animar (50ms = 20 fps)
                if not args.debug and not properties.should_project:
                    self.gif_timer.start(50)
            else:
                logger.warning(f"El GIF no contiene frames: {gif_path}")
        except Exception as e:
            logger.warning(f"Error cargando GIF: {e}")


    def _update_gif_frame(self):
        if self.gif_frames:
            self.current_gif_frame_index = (self.current_gif_frame_index + 1) % len(self.gif_frames)

            if not properties.should_project:
                self._show_gif_overlay()


    def _show_gif_overlay(self):
        if not self.gif_frames or self.current_gif_frame_index >= len(self.gif_frames):
            return

        try:
            # Obtener el tamaño actual de la ventana
            window_width = self.width()
            window_height = self.height()

            # Redimensionar el frame del GIF al tamaño de la ventana
            gif_frame = self.gif_frames[self.current_gif_frame_index].copy()
            gif_resized = cv2.resize(gif_frame, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

            # Convertir BGR a RGB para Qt
            gif_rgb = cv2.cvtColor(gif_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = gif_rgb.shape
            bytes_per_line = ch * w

            # Convertir a QImage
            from PyQt5.QtGui import QImage
            qt_image = QImage(gif_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Convertir a QPixmap
            pixmap = QPixmap.fromImage(qt_image)

            # Mostrar en el label overlay
            self.gif_overlay_label.setPixmap(pixmap)
            self.gif_overlay_label.setGeometry(0, 0, window_width, window_height)
            self.gif_overlay_label.raise_()  # Traer al frente
            self.gif_overlay_label.show()

        except Exception as e:
            logger.warning(f"Error mostrando GIF overlay: {e}")


    def _hide_gif_overlay(self):

        if hasattr(self, 'gif_overlay_label'):
            self.gif_overlay_label.hide()
        
    def on_closing(self):
        pass