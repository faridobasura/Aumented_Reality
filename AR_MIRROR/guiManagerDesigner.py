from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLabel, QPushButton, QSlider, QGroupBox, 
                              QScrollArea, QMessageBox, QFrame, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
import cv2
import numpy as np

from utils.app_args import args
from properties.properties import properties
import logging
# Constantes
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
HIP_LEFT = 23
HIP_RIGHT = 24

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 1024

logger = logging.getLogger(__name__)

class DebugWindow(QMainWindow):

    def __init__(self, controller):
        super().__init__()

        self.controller = controller  # 👈 referencia a la clase hija

        self.setWindowTitle("Controles Debug")
        self.setGeometry(int(WINDOW_WIDTH*0.35), int(WINDOW_HEIGHT*0.15), WINDOW_WIDTH, int(WINDOW_HEIGHT*0.35))

        central = QWidget()
        layout = QVBoxLayout(central)

        self.info_label = QLabel(" Debug Mode Active")
        self.info_label.setStyleSheet("color: red; font-weight: bold;")

        self.controller.create_control_section(layout)

        self.setCentralWidget(central)

    def create_control_section(self, layout):
        btn = QPushButton("Reset Tracking")
        btn.clicked.connect(self.controller.reset_tracking)
        layout.addWidget(btn)
class ARAppDesigner(QMainWindow):
    
    def __init__(self):
        super().__init__()

        # Crear interfaz
        self.init_ui()

        # Timer para actualizar video
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)  # 30ms ~ 33 FPS

        if args.debug:
            self._create_debug_window()

    def _create_debug_window(self):
        
        self.debug_window = DebugWindow(controller=self)
        self.debug_window.show()
        logger.info("Ventana debug abierta")
    
    def init_ui(self):
        self.setWindowTitle("Espejo AR")

        logger.warning(" Iniciando Interfaz Gráfica...")
        self.setGeometry(int(WINDOW_WIDTH*0.45), int(WINDOW_HEIGHT*0.9), WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Layout principal horizontal
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # === PANEL central: VIDEO ===
        central_panel_widget = QWidget()
        central_panel_layout = QVBoxLayout(central_panel_widget)

        self.video_container = QGroupBox("Video en Vivo")
        video_inner_layout = QVBoxLayout(self.video_container)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(int(WINDOW_WIDTH*0.625), int(WINDOW_HEIGHT*0.9))
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
        left_scroll.setMaximumWidth(int(WINDOW_WIDTH*0.25))
        left_scroll.setMinimumHeight(int(WINDOW_HEIGHT*0.9))
        
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
        right_scroll.setMaximumWidth(int(WINDOW_WIDTH*0.25))
        right_scroll.setMinimumHeight(int(WINDOW_HEIGHT*0.9))
        
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

        BUTTON_SIZE = QSize(int(WINDOW_WIDTH*0.1), int(WINDOW_HEIGHT*0.2))

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
            btn.setMinimumSize(int(WINDOW_WIDTH*0.1), int(WINDOW_HEIGHT*0.2))

            image_path = str(item["image"])
            pixmap = QPixmap(image_path)
            
            if pixmap.isNull():
                logger.warning(f" No se pudo cargar: {item['image']}")

            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(QSize(int(WINDOW_WIDTH*0.1), int(WINDOW_HEIGHT*0.2)))

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
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)
    
    def apply_texture(self, texture):
        if not self.model_renderer:
            return

        if texture not in self.textures_loaded:
            print(f"Textura no cargada: {texture}")
            return

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
    
    
    def closeEvent(self):
        pass