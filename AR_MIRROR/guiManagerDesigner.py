from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLabel, QPushButton, QSlider, QGroupBox, 
                              QScrollArea, QMessageBox, QFrame)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
import cv2
import numpy as np

from utils.app_args import args

# Constantes
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
HIP_LEFT = 23
HIP_RIGHT = 24

WINDOW_WIDTH = 1204
WINDOW_HEIGHT = 1024

class ARAppDesigner(QMainWindow):
    
    def __init__(self):
        super().__init__()

        # Crear interfaz
        self.init_ui()
        
        # Timer para actualizar video
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)  # 30ms ~ 33 FPS
    
    def init_ui(self):
        self.setWindowTitle("Espejo AR")
        self.setGeometry(int(WINDOW_WIDTH*0.22), int(WINDOW_WIDTH*0.77), WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal horizontal
        main_layout = QHBoxLayout(central_widget)
        
        # === PANEL central: VIDEO ===
        video_group = QGroupBox(" Vista Previa")
        video_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(int(WINDOW_WIDTH*0.625), int(WINDOW_HEIGHT*0.9))
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setScaledContents(True)
        
        video_layout.addWidget(self.video_label)
        video_group.setLayout(video_layout)
        

        ## Panel izquierdo
        control_group_left = QGroupBox(" Controles")
        control_left_main_layout = QVBoxLayout()

        # Scroll area para controles
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(int(WINDOW_WIDTH*0.25))
        
        scroll_widget_left_area = QWidget()
        scroll_layout_left_area = QVBoxLayout(scroll_widget_left_area)

        self.create_control_sections(scroll_layout_left_area)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        scroll_layout_left_area.addWidget(self.info_label)

        scroll_layout_left_area.addStretch()
        scroll.setWidget(scroll_widget_left_area)
        control_left_main_layout.addWidget(scroll)
        control_group_left.setLayout(control_left_main_layout)

        ## Panel derecho

        control_group_right = QGroupBox(" Controles")
        control_rigth_main_layout = QVBoxLayout()

        # Scroll area para controles
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(int(WINDOW_WIDTH*0.25))
        
        scroll_widget_rigth_area = QWidget()
        scroll_layout_rigth_area = QVBoxLayout(scroll_widget_rigth_area)

        self.create_control_sections(scroll_layout_rigth_area)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        scroll_layout_rigth_area.addWidget(self.info_label)

        scroll_layout_rigth_area.addStretch()
        scroll.setWidget(scroll_widget_rigth_area)
        control_rigth_main_layout.addWidget(scroll)
        control_group_right.setLayout(control_rigth_main_layout)

        #control_group.setFixedSize(400, 600)  # Ancho fijo: 400px, Alto fijo: 600px
        #video_group.setFixedSize(800, 600)

        # IMPORTANTE: Quita los weights del main_layout
        main_layout.addWidget(control_group_left)
        main_layout.addWidget(video_group)
        main_layout.addWidget(control_group_right)



        if args.debug:
            self.setStyleSheet("border: 2px solid black;")
    

    def create_control_sections(self, layout):

        self.create_texture_section(layout)
        
        self.create_brightness_section(layout)
        
        self.create_y_position_section(layout)
        
        self.create_x_position_section(layout)
        
        self.create_scale_section(layout)
        
        self.create_action_buttons(layout)

    def create_texture_section(self, layout):

        title = QLabel(" Cambiar Textura:")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        self.texture_buttons_frame = QWidget()
        self.texture_buttons_layout = QVBoxLayout(self.texture_buttons_frame)
        self.texture_buttons_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.texture_buttons_frame)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
    
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
        pass
    
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