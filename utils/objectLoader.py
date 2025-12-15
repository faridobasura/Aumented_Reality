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
    """Modelo 3D con texturas y normales"""
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray = None  # Coordenadas UV
    normals: np.ndarray = None  # Normales de vértice
    face_uvs: np.ndarray = None  # Índices de UV por cara
    face_normals: np.ndarray = None  # Índices de normales por cara
    
    def __init__(self, vertices, faces, uvs=None, normals=None, face_uvs=None, face_normals=None):
        self.vertices = vertices
        self.faces = faces
        self.uvs = uvs
        self.normals = normals
        self.face_uvs = face_uvs
        self.face_normals = face_normals
    
    @staticmethod
    def load_obj(path):
        """Carga archivo OBJ con soporte para texturas y normales"""
        vertices = []
        faces = []
        uvs = []
        normals = []
        face_uvs = []
        face_normals = []
        
        print(f"📂 Cargando modelo: {os.path.basename(path)}")
        
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                
                if line.startswith("v "):
                    # Vértice (x, y, z)
                    parts = line.split()
                    vertices.append([
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3])
                    ])
                
                elif line.startswith("vt "):
                    # Coordenada de textura (u, v)
                    parts = line.split()
                    u = float(parts[1])
                    v = float(parts[2])  # ← CAMBIA ESTO
                    uvs.append([u, v])
                
                elif line.startswith("vn "):
                    # Normal (nx, ny, nz)
                    parts = line.split()
                    normals.append([
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3])
                    ])
                
                elif line.startswith("f "):
                    # Cara (formato: v/vt/vn o v/vt o v)
                    face_vert = []
                    face_uv = []
                    face_norm = []
                    
                    for v_data in line.split()[1:]:
                        # Dividir por slashes
                        indices = v_data.split("/")
                        
                        # Índice de vértice (obligatorio, 1-based)
                        vert_idx = int(indices[0]) - 1
                        face_vert.append(vert_idx)
                        
                        # Índice de UV (opcional)
                        uv_idx = -1
                        if len(indices) > 1 and indices[1]:
                            uv_idx = int(indices[1]) - 1
                        face_uv.append(uv_idx)
                        
                        # Índice de normal (opcional)
                        norm_idx = -1
                        if len(indices) > 2 and indices[2]:
                            norm_idx = int(indices[2]) - 1
                        face_norm.append(norm_idx)
                    
                    faces.append(face_vert)
                    face_uvs.append(face_uv)
                    face_normals.append(face_norm)
        
        # Convertir a arrays numpy
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        # Normalizar vértices
        vertices = ObjModel._normalize_vertices(vertices)
        
        # Procesar UVs si existen
        if uvs:
            uvs = np.array(uvs, dtype=np.float32)
            face_uvs = np.array(face_uvs, dtype=np.int32)
        else:
            uvs = None
            face_uvs = None
        
        # Procesar normales si existen
        if normals:
            normals = np.array(normals, dtype=np.float32)
            # Normalizar normales (asegurar longitud 1)
            for i in range(len(normals)):
                length = np.linalg.norm(normals[i])
                if length > 0:
                    normals[i] = normals[i] / length
            face_normals = np.array(face_normals, dtype=np.int32)
        else:
            normals = None
            face_normals = None
        
        print(f"✅ Modelo cargado: {len(vertices)} vértices, {len(faces)} caras")
        
        if uvs is not None:
            print(f"   Texturas: {len(uvs)} coordenadas UV")
        else:
            print(f"   Texturas: NO DISPONIBLES")
            
        if normals is not None:
            print(f"   Normales: {len(normals)} normales de vértice")
        else:
            print(f"   Normales: NO DISPONIBLES - se generarán automáticamente")
        
        return ObjModel(
            vertices=vertices,
            faces=faces,
            uvs=uvs,
            normals=normals,
            face_uvs=face_uvs,
            face_normals=face_normals
        )
    
    @staticmethod
    def _normalize_vertices(vertices):
        """Normaliza vértices para que quepan en un cubo unitario centrado"""
        if len(vertices) == 0:
            return vertices
        
        min_v = vertices.min(axis=0)
        max_v = vertices.max(axis=0)
        
        size = max_v - min_v
        max_size = np.max(size)
        
        if max_size > 0:
            # Escalar para caber en un cubo de lado 1
            scale = 1.0 / max_size
        else:
            scale = 1.0
        
        # Calcular centro y trasladar al origen
        center = (min_v + max_v) / 2.0
        
        # Aplicar transformación: centrar -> escalar
        normalized = (vertices - center) * scale
        
        return normalized
    
    def has_texture_coordinates(self):
        """Verifica si el modelo tiene coordenadas UV"""
        return (self.uvs is not None and len(self.uvs) > 0 and
                self.face_uvs is not None)
    
    def has_normals(self):
        """Verifica si el modelo tiene normales"""
        return (self.normals is not None and len(self.normals) > 0 and
                self.face_normals is not None)
    
    def generate_normals(self):
        """Genera normales planas si el modelo no las tiene"""
        if self.has_normals():
            return
        
        print("🔧 Generando normales planas...")
        
        # Inicializar acumuladores de normales
        vertex_normals = np.zeros((len(self.vertices), 3), dtype=np.float32)
        vertex_count = np.zeros(len(self.vertices), dtype=np.int32)
        
        # Calcular normales por cara
        for face in self.faces:
            if len(face) >= 3:
                # Obtener vértices del triángulo
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[1]]
                v2 = self.vertices[face[2]]
                
                # Calcular normal del triángulo
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                
                length = np.linalg.norm(normal)
                if length > 0:
                    normal = normal / length
                
                # Acumular normal para cada vértice
                for vertex_idx in face:
                    vertex_normals[vertex_idx] += normal
                    vertex_count[vertex_idx] += 1
        
        # Normalizar normales de vértice
        for i in range(len(vertex_normals)):
            if vertex_count[i] > 0:
                vertex_normals[i] = vertex_normals[i] / vertex_count[i]
                # Normalizar
                length = np.linalg.norm(vertex_normals[i])
                if length > 0:
                    vertex_normals[i] = vertex_normals[i] / length
        
        self.normals = vertex_normals
        
        # Crear índices de normales (mismos que vértices)
        self.face_normals = self.faces.copy()
        
        print(f"✅ Normales generadas: {len(self.normals)} normales")
    
    def get_vertex_count(self):
        """Obtiene número total de vértices después de expandir caras"""
        total = 0
        for face in self.faces:
            if len(face) == 3:
                total += 3
            elif len(face) == 4:
                total += 6  # 2 triángulos
        return total
    
    def print_info(self):
        """Imprime información detallada del modelo"""
        print("\n📊 INFORMACIÓN DEL MODELO:")
        print(f"   Vértices: {len(self.vertices)}")
        print(f"   Caras: {len(self.faces)}")
        print(f"   Coordenadas UV: {'SÍ' if self.has_texture_coordinates() else 'NO'}")
        print(f"   Normales: {'SÍ' if self.has_normals() else 'NO'}")
        
        # Estadísticas de caras
        tri_count = 0
        quad_count = 0
        other_count = 0
        
        for face in self.faces:
            if len(face) == 3:
                tri_count += 1
            elif len(face) == 4:
                quad_count += 1
            else:
                other_count += 1
        
        print(f"   Triángulos: {tri_count}")
        print(f"   Cuadriláteros: {quad_count}")
        print(f"   Otros: {other_count}")
        
        if self.uvs is not None:
            print(f"   UVs únicas: {len(self.uvs)}")
        
        if self.normals is not None:
            print(f"   Normales únicas: {len(self.normals)}")

# Funciones auxiliares para compatibilidad (opcionales)
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
    model = ObjModel.load_obj(OBJ_PATH)
    model.print_info()
    
    # Generar normales si no existen
    if not model.has_normals():
        model.generate_normals()
    
    # Mostrar algunas normales de ejemplo
    if model.has_normals():
        print("\n📐 Normales de ejemplo (primeras 5):")
        for i in range(min(5, len(model.normals))):
            print(f"   Normal {i}: [{model.normals[i][0]:.3f}, {model.normals[i][1]:.3f}, {model.normals[i][2]:.3f}]")
    
    # Renderizar y mostrar
    img = render_to_opencv(model.vertices, model.faces)
    
    cv2.imshow('Modelo 3D', img)
    print("\n👁️ Presiona cualquier tecla para cerrar la ventana...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
