import numpy as np
import math

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def compute_torso_frame(pL, pR, pH):
    """
    pL: hombro izquierdo REAL (debe tener X negativo en tu modelo)
    pR: hombro derecho REAL (debe tener X positivo en tu modelo)
    pH: centro de caderas
    """
    # 1. Centro entre hombros
    translation = (pL + pR) / 2.0
    
    # 2. Vector entre hombros (de IZQUIERDA a DERECHA)
    shoulder_vector = pR - pL  # De izquierdo a derecho
    shoulder_dist = np.linalg.norm(shoulder_vector)
    
    if shoulder_dist < 1e-6:
        return translation, np.eye(3), shoulder_dist
    
    # Eje X: de hombro izquierdo a derecho (normalizado)
    x_axis = shoulder_vector / shoulder_dist
    
    # Eje Y: hacia ARRIBA (perpendicular al vector torso)
    # Vector de hombros a caderas
    torso_vector = pH - translation
    
    # Si el torso_vector apunta hacia abajo (Y negativo en OpenGL),
    # invertirlo para que apunte hacia arriba
    if torso_vector[1] < 0:
        torso_vector = -torso_vector
    
    y_axis = torso_vector / np.linalg.norm(torso_vector)
    
    # Eje Z: producto cruz (perpendicular al plano XY)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Recalcular Y para ortogonalidad perfecta
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 3. Matriz de rotación CORREGIDA para OpenGL
    # OpenGL: Y arriba, X derecha, Z atrás
    rotation_mat = np.column_stack([x_axis, y_axis, z_axis])
    
    # 4. CORRECCIÓN ESPECÍFICA para modelo que mira hacia abajo
    # Rota 90° alrededor del eje X para que mire hacia adelante
    correction_x = np.array([
        [1, 0, 0],
        [0, 0, -1],  # Cambia Y por Z negativo
        [0, 1, 0]    # Cambia Z por Y
    ])
    
    rotation_mat = rotation_mat @ correction_x
    
    return translation, rotation_mat, shoulder_dist