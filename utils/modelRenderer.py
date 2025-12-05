import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import os
import ctypes
import sys

Y_OFFSET = -0.3 
X_OFFSET = -0.05  
class ModelRenderer:
    """Clase corregida - solo los métodos que cambian"""
    
    def __init__(self, width=512, height=512, obj=None):
        """Constructor corregido"""
        self.width = width
        self.height = height
        self.obj_model = obj
        self.vertex_count = 0
        
        self.ANCHOR_VERTEX_IDS = {
            "right_shoulder": 2557,
            "left_shoulder": 2854,
            "right_hip": 2511,
            "left_hip": 3275,
        }
        
        # Transformaciones
        self.model_translation = np.zeros(3, dtype=np.float32)
        self.model_rotation = np.eye(3, dtype=np.float32)
        self.model_scale = 1.0
        self.render_mode = "wireframe"
        
        if obj:
            
            self.model_shoulder_dist = self._calculate_shoulder_distance()
            self.model_anchor_offset = self._calculate_anchor_offset()
            self.debug_draw_anchor_vertices()
            print(f"✅ Modelo cargado: {len(obj.vertices)} vértices")
        else:
            self.model_shoulder_dist = 0.0
            self.model_anchor_offset = np.zeros(3)
        
        # Inicializar OpenGL
        self._init_opengl()
        self._compile_shaders()
        self._init_gl_objects()
        self._upload_mesh()
    
    def _load_shader_file(self, filepath):
        """Carga un shader desde archivo"""
        try:
            with open(filepath, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el shader {filepath}")
            # Fallback a shaders embebidos
            if "vertex" in filepath:
                return """#version 330 core
                    layout(location = 0) in vec3 position;
                    layout(location = 1) in vec3 color;
                    out vec3 vertexColor;
                    uniform mat4 model;
                    uniform mat4 view;
                    uniform mat4 projection;
                    void main() {
                        gl_Position = projection * view * model * vec4(position, 1.0);
                        vertexColor = color;
                    }"""
            else:
                return """#version 330 core
                    in vec3 vertexColor;
                    out vec4 FragColor;
                    void main() {
                        FragColor = vec4(vertexColor, 1.0);
                    }"""
    
    def _compile_shaders(self):
        """Compila shaders desde archivos externos"""
        # Determinar ruta base
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # ej: /home/usuario
        shader_dir = os.path.join(parent_dir, "shaders")
        
        # Cargar shaders desde archivos
        vertex_source = self._load_shader_file(os.path.join(shader_dir, "vertex.glsl"))
        fragment_source = self._load_shader_file(os.path.join(shader_dir, "fragment.glsl"))
        
        # Crear y compilar vertex shader
        self.vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vertex_shader, vertex_source)
        glCompileShader(self.vertex_shader)
        
        # Verificar compilación
        if not glGetShaderiv(self.vertex_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(self.vertex_shader)
            raise RuntimeError(f"Error compilando vertex shader:\n{error}")
        
        # Crear y compilar fragment shader
        self.fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fragment_shader, fragment_source)
        glCompileShader(self.fragment_shader)
        
        if not glGetShaderiv(self.fragment_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(self.fragment_shader)
            raise RuntimeError(f"Error compilando fragment shader:\n{error}")
        
        # Crear programa
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, self.vertex_shader)
        glAttachShader(self.shader_program, self.fragment_shader)
        glLinkProgram(self.shader_program)
        
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program)
            raise RuntimeError(f"Error linkando shader program:\n{error}")
        
        # Obtener ubicaciones de uniformes
        self.model_loc = glGetUniformLocation(self.shader_program, "model")
        self.view_loc = glGetUniformLocation(self.shader_program, "view")
        self.proj_loc = glGetUniformLocation(self.shader_program, "projection")
        
        print("✅ Shaders compilados exitosamente desde archivos externos")

    def _init_opengl(self):
        """Inicializa contexto OpenGL con GLFW"""
        if not glfw.init():
            raise RuntimeError("No se pudo inicializar GLFW")

        # Configurar ventana completamente oculta
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # Crear ventana pequeña y oculta
        self.window = glfw.create_window(1, 1, "Hidden Render Context", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("No se pudo crear ventana GLFW oculta")

        glfw.make_context_current(self.window)

        # Ocultar completamente la ventana
        glfw.hide_window(self.window)

        # Verificar que esté oculta
        print(f"✅ Contexto OpenGL creado (ventana oculta)")
        print(f"   OpenGL version: {glGetString(GL_VERSION).decode()}")
        print(f"   GLSL version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}")
    
    def _calculate_shoulder_distance(self):
        """Calcula distancia entre hombros"""
        if self.obj_model is None:
            return 0.0
        
        left_idx = self.ANCHOR_VERTEX_IDS["left_shoulder"]
        right_idx = self.ANCHOR_VERTEX_IDS["right_shoulder"]
        
        if (left_idx >= len(self.obj_model.vertices) or 
            right_idx >= len(self.obj_model.vertices)):
            print(f"⚠️  Índices fuera de rango")
            return 0.0
        
        left = np.array(self.obj_model.vertices[left_idx])
        right = np.array(self.obj_model.vertices[right_idx])
        
        distance = np.linalg.norm(right - left)
    
        # Si la distancia es muy pequeña, escalar el modelo
        SCALE_FACTOR = 2.0  # Aumentar este valor para hacer el modelo más grande
        if distance < 0.3:  # Si es menor a 30cm en espacio 3D
            distance *= SCALE_FACTOR
            print(f"⚠️  Modelo muy pequeño, escalando por {SCALE_FACTOR}")
    
        return distance
    
    def _calculate_anchor_offset(self):
        """Calcula offset del centro del torso"""
        if self.obj_model is None:
            return np.zeros(3)
        
        # Obtener vértices ancla
        ls = np.array(self.obj_model.vertices[self.ANCHOR_VERTEX_IDS["left_shoulder"]])
        rs = np.array(self.obj_model.vertices[self.ANCHOR_VERTEX_IDS["right_shoulder"]])
        
        # Centro entre hombros
        return (ls + rs) / 2.0

    def debug_draw_anchor_vertices(self):
        """Versión simplificada - solo marca los vértices"""
        if self.obj_model is None:
            return

        print("\n=== VÉRTICES ANCLA ===")
        for name, vertex_id in self.ANCHOR_VERTEX_IDS.items():
            if vertex_id < len(self.obj_model.vertices):
                vertex = self.obj_model.vertices[vertex_id]
                print(f"{name}: vértice {vertex_id} -> {vertex}")
            else:
                print(f"ERROR: {name} índice {vertex_id} fuera de rango")
        print("=====================\n")

        # No dibujar nada, solo imprimir información
        return

    
    def set_model_transform(self, translation, rotation_mat, scale):
        """Asegura que la traslación sea un vector plano"""
        # Aplanar la traslación si es necesario
        if isinstance(translation, np.ndarray) and translation.ndim > 1:
            translation = translation.flatten()

        self.model_translation = np.array(translation, dtype=np.float32)
        self.model_rotation = np.array(rotation_mat, dtype=np.float32)
        self.model_scale = float(scale)

        # Debug detallado
        print(f"\n🎯 set_model_transform DEBUG:")
        print(f"   Translation: {self.model_translation}")
        print(f"   Translation magnitude: {np.linalg.norm(self.model_translation):.4f}")
        print(f"   Scale: {self.model_scale}")
        print(f"   Rotation matrix:")
        for i in range(3):
            print(f"     [{rotation_mat[i,0]:.3f}, {rotation_mat[i,1]:.3f}, {rotation_mat[i,2]:.3f}]")

    def set_render_mode(self, mode):
        self.render_mode = mode
    
    def _init_gl_objects(self):
        """Inicializa buffers OpenGL"""
        # Framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        
        # Textura
        self.color_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 
                     0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                               GL_TEXTURE_2D, self.color_tex, 0)
        
        # Depth buffer
        self.depth_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 
                              self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 
                                  GL_RENDERBUFFER, self.depth_rbo)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def _upload_mesh(self):
        """Sube malla a GPU con posiciones y colores"""
        if self.obj_model is None:
            return

        # Expandir caras para obtener posiciones
        positions = self._expand_faces()
        self.vertex_count = len(positions)

        # Generar colores para cada vértice
        # Por ejemplo: todos grises para el modelo
        colors = np.full((self.vertex_count, 3), 0.7, dtype=np.float32)  # Gris

        # Intercalar posiciones y colores en un solo array
        # Formato: [x, y, z, r, g, b, x, y, z, r, g, b, ...]
        interleaved = np.zeros((self.vertex_count, 6), dtype=np.float32)
        interleaved[:, 0:3] = positions  # Posiciones (x, y, z)
        interleaved[:, 3:6] = colors      # Colores (r, g, b)

        # Aplanar el array
        vertex_data = interleaved.flatten()

        # VAO/VBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        # STRIDE = 6 floats * 4 bytes cada uno = 24 bytes
        stride = 6 * 4  # 6 floats (3 posición + 3 color) * 4 bytes

        # Atributo 0: posición (3 floats)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)

        # Atributo 1: color (3 floats) - offset 12 bytes (3 floats * 4 bytes)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glBindVertexArray(0)

        print(f"✅ Malla subida a GPU: {self.vertex_count} vértices")
    
    def _expand_faces(self):
        """Convierte caras a triángulos - devuelve array 2D"""
        tri_verts = []
        for face in self.obj_model.faces:
            if len(face) == 3:
                tri_verts.extend([
                    self.obj_model.vertices[face[0]],
                    self.obj_model.vertices[face[1]],
                    self.obj_model.vertices[face[2]]
                ])
            elif len(face) == 4:
                tri_verts.extend([
                    self.obj_model.vertices[face[0]],
                    self.obj_model.vertices[face[1]],
                    self.obj_model.vertices[face[2]],
                    self.obj_model.vertices[face[0]],
                    self.obj_model.vertices[face[2]],
                    self.obj_model.vertices[face[3]]
                ])
        
        # Convertir a array numpy 2D
        return np.array(tri_verts, dtype=np.float32).reshape(-1, 3)

    def _apply_transform(self):
        if self.model_translation.ndim > 1:
            self.model_translation = self.model_translation.flatten()

        # 1. ESCALA
        scale_mat = np.eye(4, dtype=np.float32)
        scale_mat[:3, :3] *= self.model_scale

        # 2. ROTACIÓN
        rot_mat = np.eye(4, dtype=np.float32)
        rot_mat[:3, :3] = self.model_rotation

        # 3. TRASLACIÓN
        trans_mat = np.eye(4, dtype=np.float32)
        trans_mat[:3, 3] = self.model_translation

        # Orden correcto T * R * S
        self.model_matrix = trans_mat @ rot_mat @ scale_mat

        # Vista simple - cámara mirando hacia -Z
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.view_matrix[2, 3] = -3.0  # Cámara en Z = -3.0

        # 6. PROYECCIÓN
        fov = 45.0  # Reducido de 60° a 45° para menos distorsión
        aspect = self.width / max(self.height, 1)
        near = 0.1
        far = 100.0
        self.proj_matrix = self._perspective_projection(fov, aspect, near, far)

    def _perspective_projection(self, fov, aspect, near, far):
        """Crea matriz de proyección en perspectiva"""
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        
        return proj
    
    # Debug opcional
    # print(f"📐 Model matrix:\n{self.model_matrix}")
    
    def set_viewport(self, w, h):
        self.width = w
        self.height = h
        self._init_gl_objects()
    
    def _draw_model(self):
        """Dibuja el modelo con shaders"""
        glUseProgram(self.shader_program)

        # Pasar matrices a shaders
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.model_matrix.T)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view_matrix.T)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.proj_matrix.T)

        glBindVertexArray(self.vao)

        # Configurar modo de renderizado
        if self.render_mode == "wireframe":
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(1.0)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)
        glUseProgram(0)
    
    def render_to_image(self):
        """Renderiza a imagen usando shaders"""
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)       
        glEnable(GL_DEPTH_TEST)
        glClearColor(0, 0, 0, 0)  # Fondo transparente
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)    

        # 1. ACTUALIZAR transformaciones antes de renderizar
        self._apply_transform()

        aspect = self.width / max(self.height, 1)
        fov = 45.0
        near, far = 0.1, 100.0      

        # Proyección ortográfica ajustada
        self.proj_matrix = self._perspective_projection(fov, aspect, near, far)

        # 3. Mover la cámara ligeramente para mejor vista
        self.view_matrix = np.eye(4)
        self.view_matrix[2, 3] = -2.5  # Alejar cámara
        
        # 4. Renderizar modelo principal
        self._draw_model()
        
        # 5. Dibujar vértices ancla para debug
        self.draw_anchor_vertices_with_labels()
        
        # 6. Leer píxeles
        buffer = glReadPixels(0, 0, self.width, self.height,
                              GL_RGBA, GL_UNSIGNED_BYTE)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # 7. Convertir a numpy
        img = np.frombuffer(buffer, dtype=np.uint8)\
                .reshape(self.height, self.width, 4)
        
        img = cv2.flip(img, 0)  # Flip vertical porque OpenGL tiene origen abajo
        return img
    
    def draw_anchor_vertices_with_labels(self):
        """Dibuja vértices ancla con colores y etiquetas"""
        if self.obj_model is None:
            return

        # Asegurar que tenemos matrices válidas
        if not hasattr(self, 'model_matrix') or self.model_matrix is None:
            self._apply_transform()

        # Usar el mismo shader que el modelo principal
        glUseProgram(self.shader_program)

        # Configurar matrices
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.model_matrix.T)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view_matrix.T)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.proj_matrix.T)

        # Colores para los puntos ancla
        anchor_colors = {
            "right_shoulder": (1.0, 0.0, 0.0),    # Rojo
            "left_shoulder": (0.0, 1.0, 0.0),   # Verde
            "right_hip": (0.0, 0.0, 1.0),         # Azul
            "left_hip": (1.0, 1.0, 0.0)         # Amarillo
        }

        # Preparar datos para todos los puntos ancla
        anchor_data = []
        for name, vertex_id in self.ANCHOR_VERTEX_IDS.items():
            if vertex_id < len(self.obj_model.vertices):
                vertex = self.obj_model.vertices[vertex_id]
                color = anchor_colors.get(name, (1.0, 1.0, 1.0))

                # Añadir posición y color
                anchor_data.extend(vertex)   # x, y, z
                anchor_data.extend(color)    # r, g, b

        if not anchor_data:
            return

        # Convertir a numpy
        anchor_array = np.array(anchor_data, dtype=np.float32)
        num_points = len(anchor_data) // 6  # 6 valores por punto (3 pos + 3 color)

        # Crear VAO/VBO temporal
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)

        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, anchor_array.nbytes, anchor_array, GL_STATIC_DRAW)

        stride = 6 * 4  # 6 floats * 4 bytes

        # Atributo 0: posición
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)

        # Atributo 1: color
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        # Dibujar puntos grandes
        glPointSize(20.0)
        glDrawArrays(GL_POINTS, 0, num_points)
        glPointSize(1.0)

        # Limpiar
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])

        glUseProgram(0)
    
    def align_model_with_landmarks(self, landmarks_3d, scale_multiplier=None):
        """
        Alinea el modelo para que sus vértices ancla coincidan con landmarks 3D
        """
        if self.obj_model is None:
            print("❌ No hay modelo cargado")
            return None, None, None

        corrected_landmarks = {}
        print("\n🎯 CONVERSIÓN COORDENADAS MEDIAPIPE -> OPENGL:")

        for name, point in landmarks_3d.items():
            corrected_point = np.array(point, dtype=np.float32)
            
            # ✅ Invertir Y (MediaPipe usa Y hacia abajo, OpenGL hacia arriba)
            corrected_point[1] = -corrected_point[1]
            
            # ✅ Invertir Z (para que el modelo mire hacia la cámara)
            corrected_point[2] = -corrected_point[2]
            
            corrected_landmarks[name] = corrected_point
            print(f"   {name}: {point} -> {corrected_point}")

        # Preparar arrays de puntos
        model_points = []
        target_points = []
        
        for name in self.ANCHOR_VERTEX_IDS.keys():
            if name in corrected_landmarks:
                vertex_id = self.ANCHOR_VERTEX_IDS[name]
                if vertex_id < len(self.obj_model.vertices):
                    model_points.append(np.array(self.obj_model.vertices[vertex_id]))
                    target_points.append(corrected_landmarks[name])
        
        if len(model_points) < 2:
            print("❌ No hay suficientes puntos para alinear")
            return None, None, None
        
        model_points_arr = np.array(model_points)
        target_points_arr = np.array(target_points)
        
        print(f"✅ Alineando {len(model_points_arr)} puntos")

        # Calcular centroides
        model_centroid = np.mean(model_points_arr, axis=0)
        target_centroid = np.mean(target_points_arr, axis=0)
        
        # Centrar puntos
        model_centered = model_points_arr - model_centroid
        target_centered = target_points_arr - target_centroid
        
        # Rotación óptima usando SVD (Procrustes)
        H = model_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        rotation = Vt.T @ U.T
        
        # Asegurar rotación propia (det=1)
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = Vt.T @ U.T
        
        # Aplicar rotación 180° en Y SOLO UNA VEZ
        rot_180_y = np.array([
            [-1.0, 0.0,  0.0],
            [ 0.0, 1.0,  0.0],
            [ 0.0, 0.0, -1.0]
        ], dtype=np.float32)
        
        rotation = rotation
        
        print(f"\n🎯 ROTACIÓN CORREGIDA:")
        for i in range(3):
            print(f"   [{rotation[i,0]:.3f}, {rotation[i,1]:.3f}, {rotation[i,2]:.3f}]")

        # Calcular escala 
        # Basada en distancia de hombros (más precisa)
        model_shoulder_dist = np.linalg.norm(model_points[1] - model_points[0])
        target_shoulder_dist = np.linalg.norm(target_points[1] - target_points[0])
        
        if model_shoulder_dist > 0 and target_shoulder_dist > 0:
            scale_base = target_shoulder_dist / model_shoulder_dist
            
            # Factor de ajuste empírico (ajusta según tu modelo)
            SCALE_ADJUSTMENT = 1.8  # ✅ Cambia este valor entre 1.0 y 3.0
            scale = scale_base * SCALE_ADJUSTMENT
            
            print(f"\n📊 ESCALA CALCULADA:")
            print(f"   Distancia modelo: {model_shoulder_dist:.4f}")
            print(f"   Distancia target: {target_shoulder_dist:.4f}")
            print(f"   Escala base: {scale_base:.4f}")
            print(f"   Factor ajuste: {SCALE_ADJUSTMENT}")
            print(f"   Escala final: {scale:.4f}")
            
            # Aplicar multiplicador externo si existe
            if scale_multiplier is not None:
                scale *= scale_multiplier
                print(f"   Con multiplicador: {scale:.4f}")
        else:
            scale = 1.0
            if scale_multiplier is not None:
                scale *= scale_multiplier
        
        # Traslación 
        # T = T_target - s * (R * T_model)
        translation = target_centroid - scale * (rotation @ model_centroid)

        translation[0] += X_OFFSET
        translation[1] += Y_OFFSET
        
        print(f"\n📍 TRASLACIÓN:")
        print(f"   Target centroid: {target_centroid}")
        print(f"   Model centroid (rotado y escalado): {scale * (rotation @ model_centroid)}")
        print(f"   Traslación final: {translation}")
        
        # Aplicar transformación
        self.model_rotation = rotation
        self.model_scale = scale
        self.model_translation = translation
        
        # Verificar calidad de alineación
        avg_error = self.verify_alignment_quality(corrected_landmarks)
        
        return rotation, scale, translation
    
    def _calculate_average_distance(self, points):
        """Calcula la distancia promedio entre puntos"""
        if len(points) < 2:
            return 0.0
        
        # Calcular todas las distancias entre pares de puntos
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def verify_alignment_quality(self, landmarks_3d_corrected):
        """Verificación de calidad de alineación"""
        errors = {}

        for name in self.ANCHOR_VERTEX_IDS.keys():
            if name in landmarks_3d_corrected:
                vertex_id = self.ANCHOR_VERTEX_IDS[name]
                if vertex_id < len(self.obj_model.vertices):
                    # Punto del modelo
                    model_point = np.array(self.obj_model.vertices[vertex_id])

                    # Aplicar transformación CORRECTA: s * R * p + T
                    transformed_point = self.model_scale * (self.model_rotation @ model_point) + self.model_translation

                    # Punto objetivo
                    target_point = np.array(landmarks_3d_corrected[name])

                    # Calcular error
                    error = np.linalg.norm(transformed_point - target_point)
                    errors[name] = error

        if errors:
            avg_error = np.mean(list(errors.values()))
            max_error = max(errors.values())

            print(f"\n🔍 ERROR DE ALINEACIÓN:")
            for name, error in errors.items():
                status = "✅" if error < 0.05 else "⚠️" if error < 0.1 else "❌"
                print(f"   {status} {name}: {error:.4f}")
            print(f"   Promedio: {avg_error:.4f}, Máximo: {max_error:.4f}")
            
            if avg_error < 0.05:
                print("   🎉 ¡Alineación EXCELENTE!")
            elif avg_error < 0.1:
                print("   👍 Alineación BUENA")
            else:
                print("   ⚠️ Alineación necesita mejora - ajusta SCALE_ADJUSTMENT")

            return avg_error
        return 0.0
    

    def cleanup(self):
        """Limpia recursos OpenGL y GLFW"""
        print("🧹 Limpiando recursos OpenGL...")
        
        # Limpiar recursos OpenGL
        if hasattr(self, 'vao'):
            glDeleteVertexArrays(1, [self.vao])
        if hasattr(self, 'vbo'):
            glDeleteBuffers(1, [self.vbo])
        if hasattr(self, 'shader_program'):
            glDeleteProgram(self.shader_program)
        if hasattr(self, 'fbo'):
            glDeleteFramebuffers(1, [self.fbo])
        if hasattr(self, 'color_tex'):
            glDeleteTextures(1, [self.color_tex])
        if hasattr(self, 'depth_rbo'):
            glDeleteRenderbuffers(1, [self.depth_rbo])
        
        # Destruir ventana GLFW
        if hasattr(self, 'window') and self.window:
            glfw.destroy_window(self.window)
        
        # Terminar GLFW
        glfw.terminate()
        print("✅ Recursos limpiados correctamente")
    
    def debug_print_vectors(self, rotation):
        """Imprime los vectores de dirección de la rotación"""
        print(f"\n🧭 VECTORES DE DIRECCIÓN:")
        
        # Vectores base
        forward = rotation @ np.array([0, 0, 1])  # Z positivo es hacia adelante en Blender
        right = rotation @ np.array([1, 0, 0])    # X positivo es hacia la derecha
        up = rotation @ np.array([0, 1, 0])       # Y positivo es hacia arriba
        
        print(f"   Forward (Z): {forward}")
        print(f"   Right (X): {right}")
        print(f"   Up (Y): {up}")
        
        return forward, right, up

    def calculate_manual_alignment(self, model_points_arr, target_points_arr):
        """
        Alineación MANUAL basada en vectores del torso
        Esto evita los problemas de Procrustes con la orientación
        """
        print("\n🎯 ALINEACIÓN MANUAL (basada en vectores)")

        # 1. Extraer puntos específicos
        # Suponiendo que los puntos están en este orden: [left_shoulder, right_shoulder, left_hip, right_hip]
        if len(model_points_arr) >= 4 and len(target_points_arr) >= 4:
            # Puntos del modelo
            m_ls = model_points_arr[0]  # left shoulder
            m_rs = model_points_arr[1]  # right shoulder
            m_lh = model_points_arr[2]  # left hip
            m_rh = model_points_arr[3]  # right hip

            # Puntos objetivo
            t_ls = target_points_arr[0]
            t_rs = target_points_arr[1]
            t_lh = target_points_arr[2]
            t_rh = target_points_arr[3]

            # 2. Calcular vectores para el modelo
            m_right = m_rs - m_ls  # De izquierda a derecha
            m_down = (m_lh + m_rh)/2 - (m_ls + m_rs)/2  # De hombros a caderas
            m_forward = np.cross(m_right, m_down)  # Forward es perpendicular

            # Normalizar
            m_right = m_right / np.linalg.norm(m_right)
            m_down = m_down / np.linalg.norm(m_down)
            m_forward = m_forward / np.linalg.norm(m_forward)

            # 3. Calcular vectores para los landmarks
            t_right = t_rs - t_ls
            t_down = (t_lh + t_rh)/2 - (t_ls + t_rs)/2
            t_forward = np.cross(t_right, t_down)

            # Normalizar
            t_right = t_right / np.linalg.norm(t_right)
            t_down = t_down / np.linalg.norm(t_down)
            t_forward = t_forward / np.linalg.norm(t_forward)

            # 4. Crear matrices de rotación
            m_basis = np.column_stack([m_right, m_down, m_forward])
            t_basis = np.column_stack([t_right, t_down, t_forward])

            # 5. Calcular rotación que convierte m_basis a t_basis
            rotation = t_basis @ np.linalg.inv(m_basis)

            print(f"   Rotación calculada (manual)")
            return rotation

        return np.eye(3)  # Matriz identidad si falla

    
def debug_print_matrix(name, matrix):
    """Función de debug para matrices"""
    print(f"\n{name}:")
    if matrix.shape == (3, 3):
        for i in range(3):
            print(f"  [{matrix[i,0]:.3f}, {matrix[i,1]:.3f}, {matrix[i,2]:.3f}]")
    elif matrix.shape == (4, 4):
        for i in range(4):
            print(f"  [{matrix[i,0]:.3f}, {matrix[i,1]:.3f}, {matrix[i,2]:.3f}, {matrix[i,3]:.3f}]")
    else:
        print(f"  {matrix}")