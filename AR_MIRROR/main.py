import os
import sys
import cv2
import mediapipe as mp
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow

from guiManager import ARApp
from utils.app_args import args
from utils.objectLoader import ObjModel
from properties.properties import properties
from utils.logger import logger, set_log_level

if args.debug:
    set_log_level('debug')
else:
    set_log_level('info')
logger.info("----------------------------------------")
logger.info("Starting AR MIRROR GUI")

class ARController:
    def __init__(self):
        self.cap = None
        self.mirror_window = None
        
    def start(self):
        # Inicializar cámara
        self.cap = cv2.VideoCapture(properties.CAMERA_ID)

        if not self.cap.isOpened():
            logger.warning(" No se pudo abrir la cámara")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, properties.WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, properties.WINDOW_HEIGHT)

        FRAME_W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        FRAME_H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Cargar modelo 3D
        try:
            obj_loaded = ObjModel.load_obj(properties.resources.shirt_obj_path)
            logger.info(f"Modelo cargado: {len(obj_loaded.vertices)} vértices")
        except Exception as e:
            logger.warning(f"Error cargando modelo: {e}")
            self.cap.release()
            return

        # Inicializar MediaPipe
        logger.info(f"MediaPipe...")
        mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        mp_drawing = mp.solutions.drawing_utils
        pose_connections = mp.solutions.pose.POSE_CONNECTIONS

        logger.info(f"MediaPipe inicializado")

        # Crear ventana principal
        self.mirror_window = ARApp(
            cap=self.cap,
            mp_pose=mp_pose,
            mp_drawing=mp_drawing,
            pose_connections=pose_connections,
            obj_loaded=obj_loaded,
        )
        if properties.settings.Fullscreen:
            # Mostrar inmediatamente en fullscreen
            self.mirror_window.showFullScreen()
            # Forzar un redibujado completo
            self.mirror_window.update()
            self.mirror_window.repaint()
            # También forzar un redimensionamiento del video label
            QApplication.processEvents()
        else:
            # Para modo normal, también aseguramos tamaño
            self.mirror_window.showNormal()
            self.mirror_window.resize(properties.WINDOW_WIDTH, properties.WINDOW_HEIGHT)
            self.mirror_window.update()
            self.mirror_window.repaint()
            QApplication.processEvents()

def main():
    app = QApplication(sys.argv)

    controller = ARController()
    controller.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()