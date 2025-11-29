import numpy as np
import cv2
import pyrr
from utils.modernGLRenderer import ModernGLRenderer


class Model3DRenderer:
    def __init__(self, obj_loader, simplify_factor=3, use_texture=True,
                 screen_width=640, screen_height=480):

        self.obj_loader = obj_loader
        self.use_texture = use_texture
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Inicializar renderer ModernGL (texturas + geometría)
        self.gl_renderer = ModernGLRenderer(
            obj_loader,
            width=screen_width,
            height=screen_height
        )

        print("✅ ModernGL Renderer inicializado")

    # ---------------------------------------------------------
    # MATRIZ DE MODELO BASADA EN MEDIAPIPE
    # ---------------------------------------------------------
    def compute_model_matrix(self, landmarks, frame_shape,
                             shoulder_left_idx=11,
                             shoulder_right_idx=12,
                             hip_left_idx=23,
                             hip_right_idx=24):
        h, w = frame_shape[:2]

        # Obtener puntos de pose
        ls = landmarks[shoulder_left_idx]
        rs = landmarks[shoulder_right_idx]
        lh = landmarks[hip_left_idx]
        rh = landmarks[hip_right_idx]

        # Centro del torso
        cx = ((ls.x + rs.x + lh.x + rh.x) * 0.25) * w
        cy = ((ls.y + rs.y + lh.y + rh.y) * 0.25) * h

        # Distancia hombro-hombro en píxeles
        dx = (rs.x - ls.x) * w
        dy = (rs.y - ls.y) * h
        shoulder_dist = max(np.sqrt(dx*dx + dy*dy), 1e-6)

        # Escala respecto al tamaño real del modelo
        bbox_min, bbox_max = self.obj_loader.get_bounding_box()
        model_shoulder_dist = abs(bbox_max[0] - bbox_min[0])
        model_shoulder_dist = max(model_shoulder_dist, 1e-6)

        scale = shoulder_dist / model_shoulder_dist

        # Rotación Z a partir de los hombros
        angle_z = np.arctan2(dy, dx)

        # MATRICES PYRR
        S = pyrr.matrix44.create_from_scale([scale, scale, scale], dtype='f4')
        Rz = pyrr.matrix44.create_from_z_rotation(angle_z, dtype='f4')
        T = pyrr.matrix44.create_from_translation([cx, cy, 0], dtype='f4')

        # model_matrix = T * Rz * S
        model_matrix = pyrr.matrix44.multiply(T, pyrr.matrix44.multiply(Rz, S))
        return model_matrix

    # ---------------------------------------------------------
    # RENDER DEL MODELO 3D EN EL FRAME
    # ---------------------------------------------------------
    def render_model(self, frame, model_matrix):
        """Dibuja el modelo texturizado encima del frame con alpha blending."""
        # Render ModernGL a RGBA 640x480 (o tamaño del frame)
        gl_image = self.gl_renderer.render(model_matrix)

        # Separar alpha
        alpha = gl_image[:, :, 3].astype(np.float32) / 255.0

        if alpha.max() <= 0.01:
            # No hay nada visible
            return

        # RGB → BGR para OpenCV
        rgb = gl_image[:, :, :3][:, :, ::-1].astype(np.float32)

        fh, fw = frame.shape[:2]

        # Si dimensiones no coinciden, escalar buffer GL al frame
        if (self.screen_width != fw) or (self.screen_height != fh):
            # Solo emitir warning una vez
            print("⚠️ AVISO: Tamaño del framebuffer ModernGL no coincide con la cámara.")
            print("   Corrige screen_width/screen_height al crear Model3DRenderer.")
            return


        # Alpha blend
        for c in range(3):
            frame[:, :, c] = frame[:, :, c] * (1 - alpha) + rgb[:, :, c] * alpha

        frame[:, :, :] = np.clip(frame, 0, 255).astype(np.uint8)

    def render_on_frame(self, frame, pose_landmarks):
        # pose_landmarks es results.pose_landmarks (tipo NormalizedLandmarkList)
        landmarks = pose_landmarks.landmark
        model_matrix = self.compute_model_matrix(landmarks, frame.shape)
        self.render_model(frame, model_matrix)
        return frame