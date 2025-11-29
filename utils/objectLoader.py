import cv2
import os
import numpy as np
from typing import List, Tuple, Optional

class ObjectLoader:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.tex_coords = []
        self.normals = []
        self.texture = None
        self.texture_cv = None  # ← NUEVO: textura para OpenCV
        self.material_file = None
        self.texture_loaded = False

        self.anchor_points = {
            'left_shoulder': None,
            'right_shoulder': None, 
            'left_hip': None,
            'right_hip': None
        }
    
    def load_obj(self, filepath: str) -> bool:
        """
        Carga un archivo .obj y su material .mtl si existe
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Buscar archivo .mtl
            mtl_file = self._find_mtl_file(lines, filepath)
            if mtl_file:
                self._load_mtl(mtl_file)
            
            # Parsear geometría
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if not parts:
                    continue
                    
                # Vértices
                if parts[0] == 'v':
                    x, y, z = map(float, parts[1:4])
                    self.vertices.append([x, y, z])
                    
                # Coordenadas de textura
                elif parts[0] == 'vt':
                    u, v = map(float, parts[1:3])
                    self.tex_coords.append([u, v])
                    
                # Normales
                elif parts[0] == 'vn':
                    nx, ny, nz = map(float, parts[1:4])
                    self.normals.append([nx, ny, nz])
                    
                # Caras
                elif parts[0] == 'f':
                    face_vertices = []
                    for part in parts[1:]:
                        indices = part.split('/')
                        v_idx = int(indices[0]) - 1  # OBJ es 1-indexed
                        
                        vt_idx = -1
                        if len(indices) > 1 and indices[1]:
                            vt_idx = int(indices[1]) - 1
                            
                        vn_idx = -1  
                        if len(indices) > 2 and indices[2]:
                            vn_idx = int(indices[2]) - 1
                            
                        face_vertices.append([v_idx, vt_idx, vn_idx])
                    
                    self.faces.append(face_vertices)
            
            # Identificar puntos de anclaje automáticamente
            self._auto_detect_anchors()
            
            print(f"Modelo cargado: {len(self.vertices)} vértices, {len(self.faces)} caras")
            return True
            
        except Exception as e:
            print(f"Error cargando OBJ: {e}")
            return False
    
    def _find_mtl_file(self, lines: List[str], obj_path: str) -> Optional[str]:
        """Busca y retorna la ruta del archivo .mtl"""
        for line in lines:
            if line.startswith('mtllib'):
                mtl_filename = line.split()[1]
                import os
                obj_dir = os.path.dirname(obj_path)
                return os.path.join(obj_dir, mtl_filename)
        return None
    
    def _load_mtl(self, mtl_path: str):
        """Carga el archivo de materiales y texturas"""
        try:
            print(f"📖 Analizando MTL: {mtl_path}")

            if not os.path.exists(mtl_path):
                print(f"❌ Archivo MTL no existe: {mtl_path}")
                return

            texture_files = []
            material_name = None
            mtl_dir = os.path.dirname(mtl_path)

            with open(mtl_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'newmtl':
                        material_name = parts[1]
                        print(f"   📝 Material: {material_name}")

                    elif parts[0] == 'map_Kd':
                        if len(parts) > 1:
                            texture_file = parts[1]
                            texture_files.append(texture_file)
                            print(f"   🎨 Textura referenciada: '{texture_file}'")

            # Intentar cargar texturas con OpenCV
            if texture_files:
                for texture_file in texture_files:
                    texture_path = os.path.join(mtl_dir, texture_file)

                    print(f"🔄 Intentando cargar textura: {texture_path}")
                    print(f"   📄 Existe: {os.path.exists(texture_path)}")

                    if os.path.exists(texture_path):
                        if self.load_texture_opencv(texture_path):  # Usar OpenCV
                            self.texture_loaded = True
                            print(f"   🎉 Textura cargada exitosamente con OpenCV!")
                            return
                        else:
                            print(f"   ❌ Error al cargar la textura con OpenCV")

                    # Intentar alternativas...
                    # (mantén el código de alternativas que ya tenías)

            print("❌ No se pudo cargar ninguna textura")

        except Exception as e:
            print(f"❌ Error cargando MTL: {e}")
            import traceback
            traceback.print_exc()
    
    def load_texture_opencv(self, image_path: str, max_size=1024) -> bool:
        """Carga una textura usando OpenCV y la redimensiona si es muy grande"""
        try:
            print(f"🖼️ Cargando textura con OpenCV: {image_path}")
            
            # Verificar si el archivo existe
            if not os.path.exists(image_path):
                print(f"❌ Archivo de textura no existe: {image_path}")
                return False
            
            # Cargar con OpenCV
            texture_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if texture_bgr is None:
                print(f"❌ OpenCV no pudo cargar la textura: {image_path}")
                return False
            
            # Redimensionar si es muy grande
            height, width = texture_bgr.shape[:2]
            print(f"   Tamaño original: {width}x{height}")
            
            if width > max_size or height > max_size:
                # Calcular nuevo tamaño manteniendo aspect ratio
                if width > height:
                    new_width = max_size
                    new_height = int((max_size / width) * height)
                else:
                    new_height = max_size
                    new_width = int((max_size / height) * width)
                
                print(f"   Redimensionando a: {new_width}x{new_height}")
                texture_bgr = cv2.resize(texture_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convertir BGR to RGB para consistencia
            if texture_bgr.shape[2] == 3:  # RGB
                self.texture_cv = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
            elif texture_bgr.shape[2] == 4:  # RGBA
                self.texture_cv = cv2.cvtColor(texture_bgr, cv2.COLOR_BGRA2RGBA)
            else:
                self.texture_cv = texture_bgr
            
            print(f"✅ Textura cargada con OpenCV: {image_path}")
            print(f"   Tamaño final: {self.texture_cv.shape[1]}x{self.texture_cv.shape[0]}")
            print(f"   Canales: {self.texture_cv.shape[2]}")
            
            self.texture_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error cargando textura con OpenCV: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _auto_detect_anchors(self):
        """Detecta automáticamente puntos de anclaje basado en geometría"""
        if not self.vertices:
            return
            
        vertices_array = np.array(self.vertices)
        
        # Encontrar puntos extremos en X (hombros)
        left_idx = np.argmin(vertices_array[:, 0])  # Más a la izquierda
        right_idx = np.argmax(vertices_array[:, 0]) # Más a la derecha
        
        # Encontrar puntos extremos en Y (caderas vs hombros)
        top_indices = np.argsort(vertices_array[:, 1])[:10]  # 10 puntos más arriba
        bottom_indices = np.argsort(vertices_array[:, 1])[-10:]  # 10 puntos más abajo
        
        # Buscar puntos de cadera en la parte inferior
        left_hip_candidates = [idx for idx in bottom_indices if vertices_array[idx, 0] < 0]
        right_hip_candidates = [idx for idx in bottom_indices if vertices_array[idx, 0] > 0]
        
        # Asignar puntos de anclaje
        self.anchor_points['left_shoulder'] = left_idx
        self.anchor_points['right_shoulder'] = right_idx
        self.anchor_points['left_hip'] = left_hip_candidates[0] if left_hip_candidates else bottom_indices[0]
        self.anchor_points['right_hip'] = right_hip_candidates[0] if right_hip_candidates else bottom_indices[-1]
        
        print("Puntos de anclaje detectados:")
        for name, idx in self.anchor_points.items():
            if idx is not None:
                print(f"  {name}: {self.vertices[idx]}")
    
    def set_anchor_points(self, left_shoulder: int, right_shoulder: int, 
                         left_hip: int, right_hip: int):
        """Configura manualmente los puntos de anclaje por índice de vértice"""
        self.anchor_points['left_shoulder'] = left_shoulder
        self.anchor_points['right_shoulder'] = right_shoulder
        self.anchor_points['left_hip'] = left_hip
        self.anchor_points['right_hip'] = right_hip
    
    def get_anchor_positions(self):
        """Retorna las posiciones 3D de los puntos de anclaje"""
        anchors_3d = {}
        for name, idx in self.anchor_points.items():
            if idx is not None and idx < len(self.vertices):
                anchors_3d[name] = self.vertices[idx]
        return anchors_3d
    
    def get_bounding_box(self) -> Tuple[List[float], List[float]]:
        """Calcula la bounding box del modelo"""
        if not self.vertices:
            return [0, 0, 0], [0, 0, 0]
            
        vertices_array = np.array(self.vertices)
        min_coords = np.min(vertices_array, axis=0)
        max_coords = np.max(vertices_array, axis=0)
        
        return min_coords.tolist(), max_coords.tolist()
    
    def get_model_info(self) -> dict:
        """Retorna información del modelo cargado"""
        min_coords, max_coords = self.get_bounding_box()

        # Verificar coordenadas de textura
        faces_with_texcoords = 0
        for face in self.faces:
            for vertex_data in face:
                if len(vertex_data) > 1 and vertex_data[1] != -1:
                    faces_with_texcoords += 1
                    break
                
        info = {
            'vertices': len(self.vertices),
            'faces': len(self.faces),
            'texture_coords': len(self.tex_coords),  # Esta línea debe estar presente
            'faces_with_texture': faces_with_texcoords,
            'texture_loaded': self.texture is not None,
            'bounding_box': {
                'min': min_coords,
                'max': max_coords,
                'size': [max_coords[i] - min_coords[i] for i in range(3)]
            },
            'anchor_points': self.get_anchor_positions()
        }

        # Agregar tamaño de textura solo si existe
        if self.texture:
            info['texture_size'] = self.texture.get_size()

        return info

    def get_simple_faces(self):
        """Retorna solo los índices de vértices para las caras (para dibujo simplificado)"""
        simple_faces = []
        for face in self.faces:
            simple_face = []
            for vertex_data in face:
                simple_face.append([vertex_data[0], -1, -1])  # Solo mantener índice de vértice
            simple_faces.append(simple_face)
        return simple_faces
