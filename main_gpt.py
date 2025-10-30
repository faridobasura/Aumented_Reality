"""
superpone una prenda (PNG) sobre el torso de una persona
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import math
import logging

mp_pose = mp.solutions.pose

cv2.namedWindow("Detección de Pose", cv2.WINDOW_NORMAL)

logger = logging.getLogger(__name__)

def normalized_to_pixel_landmark(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def get_body_quad(landmarks, w, h):
    try:
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    except Exception:
        return None


    ls_pt = normalized_to_pixel_landmark(ls, w, h)
    rs_pt = normalized_to_pixel_landmark(rs, w, h)
    lh_pt = normalized_to_pixel_landmark(lh, w, h)
    rh_pt = normalized_to_pixel_landmark(rh, w, h)
    lw_pt = normalized_to_pixel_landmark(lw, w, h)
    rw_pt = normalized_to_pixel_landmark(rw, w, h)

    logger.warning(f"Hombro izquierdo: {ls_pt}\Hombro derecho: {rs_pt}")
    logger.warning(f"Cadera izquierda: {lh_pt}\Cadera derecha: {rh_pt}")
    logger.warning(f"Muñeca izquierda: {lw_pt}\nMuñeca derecha: {rw_pt}")
    
    # Si las caderas parecen invalidas (muy fuera o 0), estimar.
    def is_valid(pt):
        x,y = pt
        return 0 <= x < w and 0 <= y < h

    if not (is_valid(lh_pt) and is_valid(rh_pt)):
        # estimar caderas a una distancia vertical desde los hombros
        shoulder_mid_y = (ls_pt[1] + rs_pt[1]) / 2.0
        shoulder_dist = math.hypot(ls_pt[0]-rs_pt[0], ls_pt[1]-rs_pt[1])
        # heurística: caderas ~ 0.9 * shoulder_dist abajo del medio de hombros
        est_offset = int(shoulder_dist * 0.9)
        lh_pt = (ls_pt[0], int(shoulder_mid_y + est_offset))
        rh_pt = (rs_pt[0], int(shoulder_mid_y + est_offset))

    # Orden de destino para perspectiva: top-left, top-right, bottom-right, bottom-left
    shoulders = True if ((ls_pt is not None) and (rs_pt is not None)) else False
    hips = True if ((lh_pt is not None) and (rh_pt is not None)) else False
    if shoulders and hips:
        dst = np.array([ls_pt, rs_pt, rh_pt, lh_pt], dtype=np.float32)
        return dst

def warp_and_overlay(frame, garment_img, dst_quad, scale=1.0):
    """
    - garment_img: imagen RGBA (h_g, w_g, 4)
    - dst_quad: np.array float32 (4,2) con orden TL, TR, BR, BL en coordenadas del frame
    - scale: factor para escalar prenda antes de mapear (1.0 = tamaño original)
    """
    h_frame, w_frame = frame.shape[:2]
    h_g, w_g = garment_img.shape[:2]

    # src: corners de la prenda (su imagen)
    src = np.array([[0,0], [w_g-1,0], [w_g-1,h_g-1], [0,h_g-1]], dtype=np.float32)

    # opcional: escalar src (centrado)
    if scale != 1.0:
        cx, cy = w_g/2.0, h_g/2.0
        src = (src - [cx, cy]) * scale + [cx, cy]
    
    if dst_quad is None or len(dst_quad) != 4:
        logger.warning(f"⚠️ dst_quad inválido o incompleto: {dst_quad}")
        return frame
    else:

        dst_quad = np.array(dst_quad, dtype=np.float32)

        # Calcular matriz de perspectiva
        
        try:
            M = cv2.getPerspectiveTransform(src, dst_quad)
        except cv2.error as e:
            print("🚨 Error en getPerspectiveTransform:", e)
            return frame
        # Warp de la prenda al tamaño del frame
        warped = cv2.warpPerspective(garment_img, M, (w_frame, h_frame), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        # Separar alpha y rgb
        if warped.shape[2] == 4:
            alpha = warped[:,:,3] / 255.0
            garment_rgb = warped[:,:,:3]
        else:
            # Si no hay canal alpha, crear máscara a partir de no-negro
            garment_rgb = warped
            alpha = (np.mean(warped, axis=2) > 10).astype(np.float32)

        alpha = np.expand_dims(alpha, axis=2)  # (H,W,1)

        # Composición simple: result = alpha*garment + (1-alpha)*frame
        foreground = (alpha * garment_rgb).astype(np.uint8)
        background = ((1.0 - alpha) * frame).astype(np.uint8)
        composed = cv2.add(foreground, background)

        # Donde alpha==0, keep original frame (evita artefactos por rounding)
        mask_zero = (alpha[:,:,0] == 0)
        composed[mask_zero] = frame[mask_zero]

        return composed

def main(args):
    # Cargar imagen de prenda (mantener alfa)
    garment = cv2.imread("/home/faros/AR_python/Aumented_Reality/Black_T_Shirt_PNG_Clip_Art-3107.png", cv2.IMREAD_UNCHANGED)
    if garment is None:
        print("No pude cargar la prenda. Revisa la ruta.")
        return

    # Asegurar canal alfa
    if garment.shape[2] == 3:
        # agregar canal alfa totalmente opaco
        b,g,r = cv2.split(garment)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        garment = cv2.merge([b,g,r,alpha])

    # Fuente (webcam o fichero)
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    # MediaPipe Pose
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        smoothing = 0.7  # suavizado exponencial para quad para evitar jitter
        prev_dst = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            # MediaPipe trabaja en RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                dst = get_body_quad(results.pose_landmarks.landmark, w, h)
                
                if dst is not None:
                    # Suavizado temporal
                    if prev_dst is None:
                        smooth_dst = dst
                    else:
                        smooth_dst = prev_dst * smoothing + dst * (1.0 - smoothing)
                    prev_dst = smooth_dst

                    # ajustar escala según distancia de hombros (mejor encaje)
                    shoulder_width = np.linalg.norm(smooth_dst[0] - smooth_dst[1])
                    # Estimación de tamaño base de la prenda; si quieres puedes ajustar factor
                    # para agrandar o reducir la prenda relativa al ancho de hombros.
                    # Calculamos un scale basado en ancho original de la prenda:
                    garment_width_px = garment.shape[1]
                    scale_factor = (shoulder_width / garment_width_px) * 1.05  # 1.05 pequeño ajuste
                    composed = warp_and_overlay(frame, garment, smooth_dst, scale=scale_factor)
                else:
                    composed = frame
            else:
                composed = frame

            cv2.imshow("Detección de Pose", composed)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # guardar frame
                cv2.imwrite("overlay_snapshot.png", composed)
                print("Guardado snapshot: overlay_snapshot.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Superponer prenda sobre persona con MediaPipe + OpenCV")
    parser.add_argument("--video", required=False, help="Ruta a video. Si no se provee, usa webcam.")
    args = parser.parse_args()
    main(args)
