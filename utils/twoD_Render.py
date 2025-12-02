import cv2

def overlay_transparent(background, overlay, x, y):
    """Superpone una imagen RGBA sobre otra BGR."""
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

def opengl_to_transparent_rgba(frame):
    """
    Convierte un render OpenGL (wireframe blanco sobre negro)
    a una imagen RGBA con fondo transparente.
    """
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
