import cv2
from properties.properties import properties

def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if y >= background.shape[0] or x >= background.shape[1]:
        return background
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)
    overlay_crop = overlay[0:y2 - y1, 0:x2 - x1]

    if overlay_crop.shape[2] < 4:
        return background
    alpha = overlay_crop[:, :, 3] / 255.0
    for c in range(3):
        background[y1:y2, x1:x2, c] = (1 - alpha) * background[y1:y2, x1:x2, c] + alpha * overlay_crop[:, :, c]
    return background

def draw_silhouette(frame, center_x_ratio=0.5, center_y_ratio=0.5, scale=1.0):

        if properties.resources.silhouette is None:
            return frame
        
        silhouette_img = cv2.imread(
            properties.resources.silhouette,
            cv2.IMREAD_UNCHANGED
        )

        if silhouette_img is None:
            return frame
    
        frame_h, frame_w = frame.shape[:2]
    
        # Tamaño base de la silueta (proporcional a la altura del frame)
        base_height = int(frame_h * 0.65 * scale)
        aspect_ratio = silhouette_img.shape[1] / silhouette_img.shape[0]
        base_width = int(base_height * aspect_ratio)
    
        # Redimensionar silueta
        silhouette = cv2.resize(
            silhouette_img,
            (base_width, base_height),
            interpolation=cv2.INTER_AREA
        )
    
        # Centro fijo en pantalla
        center_x = int(frame_w * center_x_ratio)
        center_y = int(frame_h * center_y_ratio)
    
        # Coordenadas finales
        x = center_x - base_width // 2
        y = center_y - base_height // 2
    
        # Overlay con transparencia
        return overlay_transparent(frame, silhouette, x, y)

def opengl_to_transparent_rgba(frame):

    # Asegurar que está en RGB
    if frame.shape[2] == 3:
        rgb = frame
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Crear máscara donde negro = 0 → transparente
    _, alpha = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # Expandir a RGBA
    rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = alpha

    return rgba
