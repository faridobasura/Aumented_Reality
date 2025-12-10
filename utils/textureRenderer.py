"""
TextureRenderer - Sistema de gestión de texturas para modelos 3D
"""

import numpy as np
from OpenGL.GL import *
import cv2
import os
from typing import Dict, Optional, Tuple


class TextureRenderer:
    """Gestor de texturas para modelos 3D - Soporta múltiples tipos de texturas"""
    
    def __init__(self):
        self.textures: Dict[str, Dict[str, int]] = {}  # {name: {"diffuse": id, "normal": id}}
        self.active_texture_name: Optional[str] = None
        self.active_texture_set: Optional[Dict[str, int]] = None
        
        print("🎨 TextureRenderer inicializado (soporta texturas difusas y normales)")
    
    def _create_opengl_texture(self, image: np.ndarray, is_normal_map: bool = False) -> int:
        """
        Crea una textura OpenGL desde un array numpy
        
        Args:
            image: Array numpy (H, W, C) en formato RGB o RGBA
            is_normal_map: Si es True, configura parámetros especiales para mapa de normales
            
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
        
        # Configurar parámetros base
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        # Configuraciones especiales para mapas de normales
        if is_normal_map:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
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
    
    def _load_single_texture(self, filepath: str, texture_type: str = "diffuse") -> Optional[int]:
        """Carga una textura individual desde archivo"""
        if not os.path.exists(filepath):
            print(f"❌ No se encontró la textura: {filepath}")
            return None
        
        # Cargar imagen
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"❌ Error cargando imagen: {filepath}")
            return None
        
        # Convertir basado en el tipo de textura
        if texture_type == "normal":
            # Para mapas de normales, mantener RGBA si existe canal alpha
            if len(img.shape) == 3:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            
            print(f"🔧 Cargando mapa de normales: {os.path.basename(filepath)}")
            is_normal_map = True
        else:
            # Para texturas difusas
            if len(img.shape) == 3:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            is_normal_map = False
        
        # Crear textura OpenGL
        tex_id = self._create_opengl_texture(img, is_normal_map)
        
        return tex_id
    
    def load_texture(self, name: str, filepath: str, texture_type: str = "diffuse") -> bool:
        """
        Carga una textura desde archivo con tipo específico
        
        Args:
            name: Nombre identificador del material/textura
            filepath: Ruta al archivo
            texture_type: Tipo de textura ("diffuse", "normal")
            
        Returns:
            bool: True si se cargó exitosamente
        """
        tex_id = self._load_single_texture(filepath, texture_type)
        if tex_id is None:
            return False
        
        # Crear entrada para este material si no existe
        if name not in self.textures:
            self.textures[name] = {}
        
        # Guardar textura
        self.textures[name][texture_type] = tex_id
        
        print(f"✅ Textura '{texture_type}' para '{name}' cargada desde {os.path.basename(filepath)}")
        
        # Si es la primera textura de este material, hacerla activa
        if self.active_texture_name is None:
            self.active_texture_name = name
            self.active_texture_set = self.textures[name]
        
        return True
    
    def load_texture_pair(self, name: str, diffuse_path: str, normal_path: str = None) -> bool:
        """
        Carga un par de texturas (difusa + normal)
        
        Args:
            name: Nombre del material
            diffuse_path: Ruta a textura difusa
            normal_path: Ruta a textura normal (opcional)
            
        Returns:
            bool: True si al menos la difusa se cargó exitosamente
        """
        # Cargar textura difusa
        if not self.load_texture(name, diffuse_path, "diffuse"):
            return False
        
        # Cargar textura normal si existe
        if normal_path and os.path.exists(normal_path):
            self.load_texture(name, normal_path, "normal")
        else:
            print(f"⚠️  No se encontró mapa de normales para '{name}'")
        
        return True
    
    def set_active_texture(self, name: str) -> bool:
        """Activa un conjunto de texturas específico"""
        if name not in self.textures:
            print(f"❌ Textura '{name}' no existe")
            print(f"   Texturas disponibles: {list(self.textures.keys())}")
            return False
        
        self.active_texture_name = name
        self.active_texture_set = self.textures[name]
        
        print(f"🎨 Textura activa: '{name}'")
        if "diffuse" in self.active_texture_set:
            print(f"   ✅ Textura difusa disponible")
        if "normal" in self.active_texture_set:
            print(f"   🌟 Mapa de normales disponible")
        else:
            print(f"   ⚠️  Sin mapa de normales")
        
        return True
    
    def bind_active_textures(self):
        """Vincula todas las texturas activas a sus unidades de textura"""
        if self.active_texture_set is None:
            return
        
        # Vincular textura difusa (unidad 0)
        if "diffuse" in self.active_texture_set:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.active_texture_set["diffuse"])
        
        # Vincular textura normal (unidad 1)
        if "normal" in self.active_texture_set:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.active_texture_set["normal"])
    
    def bind_diffuse_texture(self):
        """Vincula solo la textura difusa activa"""
        if self.active_texture_set and "diffuse" in self.active_texture_set:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.active_texture_set["diffuse"])
    
    def bind_normal_texture(self):
        """Vincula solo la textura normal activa"""
        if self.active_texture_set and "normal" in self.active_texture_set:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.active_texture_set["normal"])
    
    def has_normal_map(self, texture_name: str = None) -> bool:
        """Verifica si una textura tiene mapa de normales"""
        if texture_name is None:
            texture_name = self.active_texture_name
        
        if texture_name in self.textures:
            return "normal" in self.textures[texture_name]
        return False
    
    def unbind_textures(self):
        """Desvincula todas las texturas"""
        # Desvincular de ambas unidades
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # Volver a unidad 0 por defecto
        glActiveTexture(GL_TEXTURE0)
    
    def list_textures(self) -> list:
        """Lista todas las texturas cargadas"""
        return list(self.textures.keys())
    
    def get_texture_info(self, name: str = None) -> dict:
        """Obtiene información detallada de una textura"""
        if name is None:
            name = self.active_texture_name
        
        if name not in self.textures:
            return {}
        
        info = {"name": name, "textures": {}}
        for tex_type, tex_id in self.textures[name].items():
            info["textures"][tex_type] = {
                "id": tex_id,
                "has_texture": True
            }
        
        return info
    
    def remove_texture(self, name: str) -> bool:
        """Elimina todas las texturas de un material"""
        if name not in self.textures:
            return False
        
        # Eliminar texturas OpenGL
        for tex_type, tex_id in self.textures[name].items():
            glDeleteTextures(1, [tex_id])
            print(f"🗑️ Textura '{tex_type}' de '{name}' eliminada")
        
        # Eliminar del diccionario
        del self.textures[name]
        
        # Si era la textura activa, resetear
        if self.active_texture_name == name:
            self.active_texture_name = None
            self.active_texture_set = None
            
            # Si quedan otras texturas, activar la primera
            if self.textures:
                first_name = list(self.textures.keys())[0]
                self.set_active_texture(first_name)
        
        return True
    
    def cleanup(self):
        """Libera todos los recursos"""
        print("🧹 Limpiando texturas...")
        
        for name in list(self.textures.keys()):
            self.remove_texture(name)
        
        self.textures.clear()
        self.active_texture_name = None
        self.active_texture_set = None
        
        print("✅ Texturas limpiadas")