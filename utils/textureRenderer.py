# utils/textureRenderer.py
import numpy as np
from OpenGL.GL import *
import cv2
import os

class TextureRenderer:
    def __init__(self):
        self.textures = {}
        self.active_texture_name = None
        self.active_texture_id = None

    # -----------------------------------------------------
    #  Crear textura OpenGL desde imagen
    # -----------------------------------------------------
    def _create_opengl_texture(self, image):
        image = cv2.flip(image, 0)
        h, w = image.shape[:2]
        has_alpha = (image.shape[2] == 4)

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        if has_alpha:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, image)
        else:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, image)

        glBindTexture(GL_TEXTURE_2D, 0)
        return texture_id

    # -----------------------------------------------------
    #  Cargar textura desde archivo
    # -----------------------------------------------------
    def load_texture(self, name, filepath):
        if not os.path.exists(filepath):
            print(f"❌ No se encontró la textura: {filepath}")
            return False

        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"❌ Error cargando imagen: {filepath}")
            return False

        tex_id = self._create_opengl_texture(img)
        self.textures[name] = tex_id
        return True

    # -----------------------------------------------------
    #  Crear textura sólida (RGB)
    # -----------------------------------------------------
    def create_solid_color_texture(self, name, color_rgb, size=256):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :] = color_rgb
        tex_id = self._create_opengl_texture(img)
        self.textures[name] = tex_id

    # -----------------------------------------------------
    #  Crear textura checkerboard
    # -----------------------------------------------------
    def create_checkerboard_texture(self, name, color1, color2, size=256, block=32):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                if ((x // block) + (y // block)) % 2 == 0:
                    img[y, x] = color1
                else:
                    img[y, x] = color2

        tex_id = self._create_opengl_texture(img)
        self.textures[name] = tex_id

    # -----------------------------------------------------
    # Activar textura
    # -----------------------------------------------------
    def set_active_texture(self, name):
        if name not in self.textures:
            print(f"❌ Textura '{name}' no existe")
            return

        self.active_texture_name = name
        self.active_texture_id = self.textures[name]

    # -----------------------------------------------------
    # Usar textura activa en OpenGL
    # -----------------------------------------------------
    def bind_active_texture(self):
        if self.active_texture_id:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.active_texture_id)

    def unbind_texture(self):
        glBindTexture(GL_TEXTURE_2D, 0)

    # -----------------------------------------------------
    # Liberar recursos
    # -----------------------------------------------------
    def cleanup(self):
        for tex_name, tex_id in self.textures.items():
            glDeleteTextures([tex_id])
        self.textures.clear()
        self.active_texture_id = None
