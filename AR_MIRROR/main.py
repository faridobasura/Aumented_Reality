import os
import sys
import cv2
import mediapipe as mp
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from guiManager import ARApp
from utils.app_args import args
from utils.objectLoader import ObjModel
from properties.properties import properties
from utils.logger import logger, set_log_level

# Constantes


# Rutas de archivos
TWOD_SHIRT_PATH = os.path.expanduser('~/AR_python/Aumented_Reality/AR_MIRROR/models/Black_T_Shirt_PNG_Clip_Art-3107.png')
SHIRT_OBJ_PATH = os.path.expanduser('~/AR_python/Aumented_Reality/AR_MIRROR/models/obj/new_shirt.obj')

if args.debug:
    set_log_level('debug')
else:
    set_log_level('info')
logger.info("----------------------------------------")
logger.info("Starting AR MIRROR GUI")

def main():
    # Inicializar cámara
    cap = cv2.VideoCapture(properties.CAMERA_ID)
    
    if not cap.isOpened():
        logger.warning(" No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, properties.WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, properties.WINDOW_HEIGHT)
    
    FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Cargar modelo 3D
    try:
        obj_loaded = ObjModel.load_obj(SHIRT_OBJ_PATH)
        logger.info(f"Modelo cargado: {len(obj_loaded.vertices)} vértices")
    except Exception as e:
        logger.warning(f"Error cargando modelo: {e}")
        cap.release()
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
    
    # Crear aplicación Qt
    app = QApplication(sys.argv)
    
    # Crear ventana principal
    window = ARApp(
        cap=cap,
        mp_pose=mp_pose,
        mp_drawing=mp_drawing,
        pose_connections=pose_connections,
        obj_loaded=obj_loaded,
        textures_dir=args.textures_dir,
        twoD_shirth_path=TWOD_SHIRT_PATH
    )
    if properties.settings.Fullscreen:
        # Mostrar inmediatamente en fullscreen
        window.showFullScreen()
        # Forzar un redibujado completo
        window.update()
        window.repaint()
        # También forzar un redimensionamiento del video label
        QApplication.processEvents()
    else:
        # Para modo normal, también aseguramos tamaño
        window.showNormal()
        window.resize(properties.WINDOW_WIDTH, properties.WINDOW_HEIGHT)
        window.update()
        window.repaint()
        QApplication.processEvents()

    # Ejecutar aplicación
    exit_code = app.exec_()
    
    logger.info("Programa finalizado correctamente")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()