"""
TextureRenderer - Sistema de gestión de texturas para modelos 3D
"""

import numpy as np
from OpenGL.GL import *
import cv2
import os
from typing import Dict, Optional


class TextureRenderer:
    """Gestor de texturas para modelos 3D - Solo carga desde archivos"""
    
    def __init__(self):
        self.textures: Dict[str, int] = {}
        self.active_texture_name: Optional[str] = None
        self.active_texture_id: Optional[int] = None
        print("🎨 TextureRenderer inicializado")
    
    
    def _create_opengl_texture(self, image: np.ndarray) -> int:
        """
        Crea una textura OpenGL desde un array numpy
        
        Args:
            image: Array numpy (H, W, C) en formato RGB o RGBA
            
        Returns:
            int: ID de textura OpenGL
        """
        # Voltear verticalmente (OpenGL tiene origen abajo-izquierda)
        image = cv2.flip(image, 0)
        
        h, w = image.shape[:2]
        has_alpha = (len(image.shape) == 3 and image.shape[2] == 4)
        
        # Generar textura
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Configurar parámetros
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        # Subir datos
        if has_alpha:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE, image)
        else:
            # Asegurar formato RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                        GL_RGB, GL_UNSIGNED_BYTE, image)
        
        # Generar mipmaps
        glGenerateMipmap(GL_TEXTURE_2D)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
        return texture_id
    
    
    def load_texture(self, name: str, filepath: str) -> bool:
        """
        Carga una textura desde archivo
        
        Args:
            name: Nombre identificador
            filepath: Ruta al archivo
            
        Returns:
            bool: True si se cargó exitosamente
        """
        if not os.path.exists(filepath):
            print(f"❌ No se encontró la textura: {filepath}")
            return False
        
        # Cargar imagen
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"❌ Error cargando imagen: {filepath}")
            return False
        
        # Convertir BGR a RGB
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        
        tex_id = self._create_opengl_texture(img)
        self.textures[name] = tex_id
        
        print(f"✅ Textura '{name}' cargada: {img.shape} desde {os.path.basename(filepath)}")
        return True
    
    
    def set_active_texture(self, name: str) -> bool:
        """Activa una textura específica"""
        if name not in self.textures:
            print(f"❌ Textura '{name}' no existe")
            print(f"   Texturas disponibles: {list(self.textures.keys())}")
            return False
        
        self.active_texture_name = name
        self.active_texture_id = self.textures[name]
        
        print(f"🎨 Textura activa: '{name}' (ID: {self.active_texture_id})")
        return True
    
    
    def bind_active_texture(self):
        """Vincula la textura activa"""
        if self.active_texture_id is not None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.active_texture_id)
            # Nota: En OpenGL moderno, NO se debe usar glEnable(GL_TEXTURE_2D)
    
    
    def unbind_texture(self):
        """Desvincular textura"""
        glBindTexture(GL_TEXTURE_2D, 0)
        # Nota: En OpenGL moderno, NO se debe usar glDisable(GL_TEXTURE_2D)
    
    
    def list_textures(self) -> list:
        """Lista todas las texturas cargadas"""
        return list(self.textures.keys())
    
    
    def remove_texture(self, name: str) -> bool:
        """Elimina una textura"""
        if name not in self.textures:
            return False
        
        tex_id = self.textures[name]
        glDeleteTextures(1, [tex_id])
        
        del self.textures[name]
        
        if self.active_texture_name == name:
            self.active_texture_name = None
            self.active_texture_id = None
        
        print(f"🗑️ Textura '{name}' eliminada")
        return True
    
    
    def cleanup(self):
        """Libera todos los recursos"""
        print("🧹 Limpiando texturas...")
        
        for name in list(self.textures.keys()):
            self.remove_texture(name)
        
        self.textures.clear()
        self.active_texture_id = None
        self.active_texture_name = None
        
        print("✅ Texturas limpiadas")
