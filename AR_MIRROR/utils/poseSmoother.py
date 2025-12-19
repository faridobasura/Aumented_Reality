import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple

class PoseSmoother:    
    def __init__(self, window_size: int = 5, alpha: float = 0.3):
        """
        Args:
            window_size: Número de frames anteriores para promediar
            alpha: Factor de suavizado exponencial (0-1)
                  - 0.1 = muy suave (lag)
                  - 0.5 = equilibrio
                  - 0.9 = poco suavizado
        """
        self.window_size = window_size
        self.alpha = alpha
        
        # Buffers para histórico
        self.translation_history = deque(maxlen=window_size)
        self.rotation_history = deque(maxlen=window_size)
        self.scale_history = deque(maxlen=window_size)
        self.landmarks_history = deque(maxlen=window_size)
        
        # Valores suavizados anteriores
        self.last_translation = None
        self.last_rotation = None
        self.last_scale = None
        self.last_landmarks = None
        
        self.outlier_threshold = 0.15  # 15% de cambio máximo aceptable
        
    def smooth_landmarks_3d(self, landmarks_3d: Dict) -> Dict:        

        self.landmarks_history.append(landmarks_3d.copy())
        
        # Si no hay histórico suficiente, devolver tal cual
        if len(self.landmarks_history) < 2:
            self.last_landmarks = landmarks_3d.copy()
            return landmarks_3d
        
        smoothed = {}
        
        for key in landmarks_3d.keys():
            if self.last_landmarks and key in self.last_landmarks:

                # Filtro exponencial: new = alpha * current + (1-alpha) * last
                current = np.array(landmarks_3d[key])
                last = np.array(self.last_landmarks[key])
                
                # Detectar outliers con distancia euclidiana
                diff = np.linalg.norm(current - last)
                
                if diff > self.outlier_threshold:
                    # Es un outlier, usar el anterior
                    smoothed[key] = last.tolist()
                else:
                    # Suavizar con histórico
                    smoothed[key] = (self.alpha * current + (1 - self.alpha) * last).tolist()
            else:
                smoothed[key] = landmarks_3d[key]
        
        self.last_landmarks = smoothed.copy()
        return smoothed
    
    def smooth_transform(self, 
                        translation: np.ndarray, 
                        rotation: np.ndarray, 
                        scale: float) -> Tuple[np.ndarray, np.ndarray, float]:
        
        self.translation_history.append(translation.copy())
        self.rotation_history.append(rotation.copy())
        self.scale_history.append(scale)
        
        # Suavizar cada componente
        smoothed_translation = self._smooth_translation(translation)
        smoothed_rotation = self._smooth_rotation(rotation)
        smoothed_scale = self._smooth_scale(scale)
        
        return smoothed_translation, smoothed_rotation, smoothed_scale
    
    def _smooth_translation(self, translation: np.ndarray) -> np.ndarray:
        if self.last_translation is None:
            self.last_translation = translation.copy()
            return translation
        
        diff = np.linalg.norm(translation - self.last_translation)
        
        if diff > 0.5:  # Salto mayor a 50cm
            # Limitar el cambio al 50% del salto
            direction = (translation - self.last_translation) / (diff + 1e-6)
            limited_translation = self.last_translation + direction * 0.25
            translation = limited_translation
        
        # Filtro exponencial
        smoothed = self.alpha * translation + (1 - self.alpha) * self.last_translation
        
        self.last_translation = smoothed.copy()
        return smoothed
    
    def _smooth_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Suaviza matriz de rotación usando SLERP"""
        if self.last_rotation is None:
            self.last_rotation = rotation.copy()
            return rotation
        
        # Convertir a quaterniones para suavizado
        q_current = self._rotation_to_quaternion(rotation)
        q_last = self._rotation_to_quaternion(self.last_rotation)
        
        # SLERP (Spherical Linear Interpolation)
        q_smoothed = self._slerp(q_last, q_current, self.alpha)
        
        # Convertir de vuelta a matriz
        smoothed = self._quaternion_to_rotation(q_smoothed)
        
        self.last_rotation = smoothed.copy()
        return smoothed
    
    def _smooth_scale(self, scale: float) -> float:
        """Suaviza escala"""
        if self.last_scale is None:
            self.last_scale = scale
            return scale
        
        # Detectar cambios bruscos
        ratio = scale / (self.last_scale + 1e-6)
        
        if ratio > 1.2 or ratio < 0.8:  # Cambio mayor al 20%
            # Limitar al 10% de cambio
            scale = self.last_scale * (1.0 + np.clip(ratio - 1.0, -0.1, 0.1))
        
        # Filtro exponencial
        smoothed = self.alpha * scale + (1 - self.alpha) * self.last_scale
        
        self.last_scale = smoothed
        return smoothed
    
    @staticmethod
    def _rotation_to_quaternion(rot_matrix: np.ndarray) -> np.ndarray:
        """Convierte matriz de rotación a quaternión"""
        trace = np.trace(rot_matrix)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
            y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
            z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
        elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
            w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        elif rot_matrix[1, 1] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
            w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
            w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
            x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])
    
    @staticmethod
    def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
        """Convierte quaternión a matriz de rotación"""
        x, y, z, w = q
        
        rot_matrix = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return rot_matrix
    
    @staticmethod
    def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Interpolación esférica lineal entre quaterniones"""
        # Normalizar
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        dot = np.dot(q1, q2)
        
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        dot = np.clip(dot, -1.0, 1.0)
        
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        q3 = (q2 - q1 * dot)
        q3 = q3 / (np.linalg.norm(q3) + 1e-6)
        
        return q1 * np.cos(theta) + q3 * np.sin(theta)
    
    def reset(self):
        """Resetea todos los buffers (para cambio de persona)"""
        self.translation_history.clear()
        self.rotation_history.clear()
        self.scale_history.clear()
        self.landmarks_history.clear()
        
        self.last_translation = None
        self.last_rotation = None
        self.last_scale = None
        self.last_landmarks = None