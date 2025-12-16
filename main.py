"""
main.py - Punto de entrada principal
Inicializa MediaPipe y la GUI Tkinter
"""

import os
import tkinter as tk
import cv2
import mediapipe as mp

from utils.objectLoader import ObjModel
from utils.app_args import args
from guiManager import ARApp

# Constantes
CAMERA_ID = 0

# Rutas de archivos
shirt_path = os.path.expanduser('~/AR_python/Aumented_Reality/models/Black_T_Shirt_PNG_Clip_Art-3107.png')
obj_path = os.path.expanduser('~/AR_python/Aumented_Reality/models/obj/new_shirt.obj')

def main():
    
    # Inicializar cámara
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    
    # Configurar resolución
    DESIRED_WIDTH = 720
    DESIRED_HEIGHT = 1024
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    
    FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Cargar modelo 3D
    try:
        obj_loaded = ObjModel.load_obj(obj_path)
        print(f"Modelo cargado: {len(obj_loaded.vertices)} vértices")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        cap.release()
        return
    
    # Inicializar MediaPipe
    print(f"\nMediaPipe...")
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    mp_drawing = mp.solutions.drawing_utils
    pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    print(f"MediaPipe inicializado")
    root = tk.Tk()
    
    # Crear aplicación
    app = ARApp(
        root=root,
        cap=cap,
        mp_pose=mp_pose,
        mp_drawing = mp_drawing,
        pose_connections = pose_connections,
        obj_loaded=obj_loaded,
        textures_dir=args.textures_dir,
        shirt_path=shirt_path
    )    
    root.mainloop()
    
    # Limpiar
    print("\nLimpiando recursos...")
    cap.release()
    mp_pose.close()
    print("Programa finalizado correctamente")


if __name__ == "__main__":
    main()