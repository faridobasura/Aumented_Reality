import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
from pygame.locals import *
import cv2
import numpy as np
import os

OBJ_PATH = os.path.expanduser('~/AR_python/Aumented_Reality/t_shirt_model/t_shirt.obj')

from dataclasses import dataclass

@dataclass
class ObjModel:
    vertices: np.ndarray
    faces: np.ndarray

    @staticmethod
    def load_obj(path):
        vertices = []
        faces = []

        with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    vertices.append(list(map(float, line.split()[1:4])))
                elif line.startswith("f "):
                    faces.append([int(v.split("/")[0]) - 1 for v in line.split()[1:]])

        vertices = normalize_vertices(vertices)

        return ObjModel(
            vertices=vertices.astype(np.float32),
            faces=np.array(faces, dtype=np.uint32)
        )

    
def normalize_vertices(vertices):
    vertices = np.array(vertices, dtype=np.float32)    # ← CONVERSIÓN NECESARIA

    min_v = vertices.min(axis=0)
    max_v = vertices.max(axis=0)

    size = max_v - min_v
    scale = 1.0 / max(size)

    center = (min_v + max_v) / 2.0

    return (vertices - center) * scale


def load_obj():
    """Cargar archivo OBJ"""
    vertices = []
    faces = []
    
    with open(OBJ_PATH, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:4]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]]
                faces.append(face)
    
    return np.array(vertices), np.array(faces)

def init_gl(width, height):
    """Inicializar OpenGL"""
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
    """Renderizar y convertir a imagen OpenCV"""
    width, height = 800, 600
    
    # Inicializar Pygame con OpenGL
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

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # Wireframe
    glLineWidth(1.0)  # Grosor de línea

    
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
    image = np.flipud(image)  # Voltear verticalmente
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    pygame.quit()
    return image

# Uso principal
def main():
    # Cargar modelo
    vertices, faces = load_obj()
    
    # Renderizar a imagen OpenCV
    img = render_to_opencv(vertices, faces)
    
    # Mostrar con OpenCV
    cv2.imshow('Modelo 3D', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()