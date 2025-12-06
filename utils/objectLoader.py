import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
from pygame.locals import *
import os
from dataclasses import dataclass

OBJ_PATH = os.path.expanduser('~/AR_python/Aumented_Reality/t_shirt_model/t_shirt.obj')

@dataclass
class ObjModel:
    """Modelo 3D con texturas"""
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray = None  # Coordenadas UV
    face_uvs: np.ndarray = None  # Índices de UV por cara
    
    def __init__(self, vertices, faces, uvs=None, face_uvs=None):
        self.vertices = vertices
        self.faces = faces
        self.uvs = uvs
        self.face_uvs = face_uvs
    
    @staticmethod
    def load_obj(path):
        """Carga archivo OBJ con soporte para texturas"""
        vertices = []
        faces = []
        uvs = []
        face_uvs = []
        
        print(f"📂 Cargando modelo: {os.path.basename(path)}")
        
        with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    # Vértice
                    vertices.append(list(map(float, line.split()[1:4])))
                
                elif line.startswith("vt "):
                    # Coordenada de textura (UV)
                    uv = list(map(float, line.split()[1:3]))
                    uvs.append(uv)
                
                elif line.startswith("f "):
                    # Cara (puede tener formato v/vt/vn o v/vt o v)
                    face = []
                    face_uv = []
                    
                    for v in line.split()[1:]:
                        parts = v.split("/")
                        
                        # Índice de vértice
                        face.append(int(parts[0]) - 1)
                        
                        # Índice de UV (si existe)
                        if len(parts) > 1 and parts[1]:
                            face_uv.append(int(parts[1]) - 1)
                        else:
                            face_uv.append(-1)  # Sin UV
                    
                    faces.append(face)
                    face_uvs.append(face_uv)
        
        # Convertir a arrays numpy
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)
        
        # Normalizar vértices
        vertices = normalize_vertices(vertices)
        
        # Manejar UVs
        if uvs:
            uvs = np.array(uvs, dtype=np.float32)
            face_uvs = np.array(face_uvs, dtype=np.uint32)
            
            # Verificar que todos los vértices tengan UVs
            if len(uvs) == 0:
                print("⚠️ Advertencia: El archivo OBJ no contiene coordenadas UV")
                uvs = None
                face_uvs = None
        else:
            uvs = None
            face_uvs = None
        
        print(f"✅ Modelo cargado: {len(vertices)} vértices, {len(faces)} caras")
        if uvs is not None:
            print(f"   Texturas: {len(uvs)} coordenadas UV")
        
        return ObjModel(
            vertices=vertices,
            faces=faces,
            uvs=uvs,
            face_uvs=face_uvs
        )
    
    def has_texture_coordinates(self):
        """Verifica si el modelo tiene coordenadas UV"""
        return self.uvs is not None and self.face_uvs is not None

def normalize_vertices(vertices):
    """Normaliza vértices para que quepan en un cubo unitario"""
    if len(vertices) == 0:
        return vertices
    
    min_v = vertices.min(axis=0)
    max_v = vertices.max(axis=0)
    
    size = max_v - min_v
    scale = 1.0 / max(size) if max(size) > 0 else 1.0
    
    center = (min_v + max_v) / 2.0
    
    return (vertices - center) * scale

# Funciones auxiliares para compatibilidad
def load_obj():
    """Función de compatibilidad para cargar modelo simple"""
    model = ObjModel.load_obj(OBJ_PATH)
    return model.vertices, model.faces

def init_gl(width, height):
    """Inicializar OpenGL (para compatibilidad)"""
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def render_to_opencv(vertices, faces):
    """Renderizar y convertir a imagen OpenCV (para compatibilidad)"""
    width, height = 800, 600
    
    pygame.init()
    display = (width, height)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    init_gl(width, height)
    
    # Configurar vista
    glLoadIdentity()
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
    glTranslatef(0.0, 0.0, -5)
    
    # Limpiar buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glLineWidth(1.0)
    
    # Dibujar modelo
    glBegin(GL_TRIANGLES)
    glColor3f(0.5, 0.5, 1.0)
    for face in faces:
        for vertex_idx in face:
            if vertex_idx < len(vertices):
                glVertex3fv(vertices[vertex_idx])
    glEnd()
    
    # Capturar framebuffer
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    
    # Convertir a imagen OpenCV
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    image = np.flipud(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    pygame.quit()
    return image

if __name__ == '__main__':
    # Cargar y mostrar modelo
    vertices, faces = load_obj()
    img = render_to_opencv(vertices, faces)
    
    cv2.imshow('Modelo 3D', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
