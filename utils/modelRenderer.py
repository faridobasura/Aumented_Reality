import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import os
import ctypes

class ModelRenderer:
    def __init__(self, width=512, height=512, obj=None):
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
        
        # Calcular propiedades
        if obj:
            # Verificar orientación del modelo
            self.check_model_orientation()

            self.model_shoulder_dist = self._calculate_shoulder_distance()
            self.model_anchor_offset = self._calculate_anchor_offset()

            # DEBUG: Imprimir información de vértices ancla
            self.debug_draw_anchor_vertices()

            print(f"✅ Modelo cargado: {len(obj.vertices)} vértices")
            print(f"✅ Distancia entre hombros: {self.model_shoulder_dist:.4f}")
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

        # Debug: verificar formas
        print(f"🎯 set_model_transform:")
        print(f"   Translation shape: {self.model_translation.shape}")
        print(f"   Rotation shape: {self.model_rotation.shape}")
        print(f"   Scale: {self.model_scale}")
    
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
        """Crea matrices de transformación"""
        # Asegurar que la traslación sea vector plano
        if self.model_translation.ndim > 1:
            self.model_translation = self.model_translation.flatten()

        self.model_matrix = np.eye(4, dtype=np.float32)

        scale_mat = np.eye(4)
        scale_mat[:3, :3] *= self.model_scale

        # Rotación 180° en X
        # Ángulo de 180 grados en radianes = π
        angle = np.pi  # 180 grados
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        rot_180_x = np.array([
            [1.0,  0.0,   0.0, 0.0],
            [0.0,  cos_a, -sin_a, 0.0],
            [0.0,  sin_a,  cos_a, 0.0],
            [0.0,  0.0,   0.0, 1.0]
        ], dtype=np.float32)

        # Para 180°, cos(π) = -1, sin(π) = 0
        # Así que la matriz se simplifica a:
        rot_180_x = np.array([
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0]
        ], dtype=np.float32)

        rot_initial = rot_180_x

        # Rotación del usuario
        user_rot_mat = np.eye(4)
        user_rot_mat[:3, :3] = self.model_rotation

        # Traslación - asegurar que es array 1D
        trans_mat = np.eye(4)
        trans_mat[:3, 3] = self.model_translation.reshape(3)  # Asegurar forma (3,)

        self.model_matrix = trans_mat @ user_rot_mat @ rot_initial @ scale_mat

        # Matriz de vista (cámara)
        self.view_matrix = np.eye(4)

        # Mover la cámara hacia atrás para ver mejor
        self.view_matrix[2, 3] = -3.0  # Alejar cámara

        # Matriz de proyección en perspectiva (más natural)
        fov = 60.0  # Campo de visión en grados
        aspect = self.width / max(self.height, 1)
        near = 0.1
        far = 100.0

        # Proyección perspectiva
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
            "left_shoulder": (1.0, 0.0, 0.0),    # Rojo
            "right_shoulder": (0.0, 1.0, 0.0),   # Verde
            "left_hip": (0.0, 0.0, 1.0),         # Azul
            "right_hip": (1.0, 1.0, 0.0)         # Amarillo
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
    
        # 1. Obtener vértices ancla del modelo
        model_points = []
        target_points = []
        
        # Variables para cálculos de hombros
        model_left_shoulder = None
        model_right_shoulder = None
        target_left_shoulder = None
        target_right_shoulder = None
    
        for name in self.ANCHOR_VERTEX_IDS.keys():
            if name in landmarks_3d:
                vertex_id = self.ANCHOR_VERTEX_IDS[name]
                if vertex_id < len(self.obj_model.vertices):
                    # Punto del modelo
                    model_point = np.array(self.obj_model.vertices[vertex_id])
                    model_points.append(model_point)
    
                    # Punto objetivo (landmark)
                    target_point = np.array(landmarks_3d[name])
                    target_points.append(target_point)
                    
                    # Guardar puntos de hombros para cálculo de distancia
                    if name == "left_shoulder":
                        model_left_shoulder = model_point
                        target_left_shoulder = target_point
                    elif name == "right_shoulder":
                        model_right_shoulder = model_point
                        target_right_shoulder = target_point
    
        if len(model_points) < 2:
            print("❌ No hay suficientes puntos para alinear")
            return None, None, None
    
        # Convertir a arrays numpy con forma (N, 3)
        model_points_arr = np.array(model_points)  # (N, 3)
        target_points_arr = np.array(target_points)  # (N, 3)
    
        print(f"✅ Alineando {len(model_points_arr)} puntos")
        print(f"   Forma model_points: {model_points_arr.shape}")
        print(f"   Forma target_points: {target_points_arr.shape}")
    
        # 2. Calcular centroides
        model_centroid = np.mean(model_points_arr, axis=0)  # (3,)
        target_centroid = np.mean(target_points_arr, axis=0)  # (3,)
    
        # 3. Centrar puntos
        model_centered = model_points_arr - model_centroid  # (N, 3)
        target_centered = target_points_arr - target_centroid  # (N, 3)
    
        # 4. Transponer para tener forma (3, N) para Procrustes
        model_centered_T = model_centered.T  # (3, N)
        target_centered_T = target_centered.T  # (3, N)
    
        # 5. Calcular rotación óptima (Procrustes)
        H = model_centered_T @ target_centered_T.T  # (3, N) @ (N, 3) = (3, 3)
        U, S, Vt = np.linalg.svd(H)
        rotation = Vt.T @ U.T
    
        # Asegurar que es una rotación propia (det=1)
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = Vt.T @ U.T
    
        print(f"   Rotación shape: {rotation.shape}")
    
        # 6. Calcular distancias de hombros
        if model_left_shoulder is not None and model_right_shoulder is not None:
            model_shoulder_dist = np.linalg.norm(model_right_shoulder - model_left_shoulder)
        else:
            # Si no tenemos hombros, usar distancia promedio entre todos los puntos
            model_shoulder_dist = self._calculate_average_distance(model_points_arr)
        
        if target_left_shoulder is not None and target_right_shoulder is not None:
            target_shoulder_dist = np.linalg.norm(target_right_shoulder - target_left_shoulder)
        else:
            # Si no tenemos hombros, usar distancia promedio entre todos los puntos
            target_shoulder_dist = self._calculate_average_distance(target_points_arr)
    
        # 7. Calcular escala base usando Procrustes
        model_norm = np.trace(model_centered_T @ model_centered_T.T)  # traza de (3,3)
        if model_norm > 0:
            # target_centered_T.T @ rotation @ model_centered_T
            # (3, N) @ (3, N).T -> ya no, mejor usar la forma estándar
            scale_base = np.trace(target_centered_T.T @ rotation @ model_centered_T) / model_norm
        else:
            scale_base = 1.0
    
        print(f"   Escala base (Procrustes): {scale_base:.4f}")
    
        # 8. Calcular multiplicador de escala basado en distancias de hombros
        if model_shoulder_dist > 0 and target_shoulder_dist > 0:
            # Calcular factor basado en la relación de distancias
            distance_ratio = target_shoulder_dist / model_shoulder_dist
            
            # FACTOR DE AJUSTE - Aumentar este valor si el modelo sigue pequeño
            ADJUSTMENT_FACTOR = 3.5  # <-- Ajusta este valor según necesites
            
            if scale_multiplier is None:
                scale_multiplier = distance_ratio * ADJUSTMENT_FACTOR
            else:
                print(f"   Usando multiplicador proporcionado: {scale_multiplier}")
            
            print(f"📏 Distancia hombros modelo: {model_shoulder_dist:.4f}")
            print(f"📏 Distancia hombros landmarks: {target_shoulder_dist:.4f}")
            print(f"📐 Relación de distancia: {distance_ratio:.4f}")
            print(f"🎯 Multiplicador calculado: {scale_multiplier:.4f}")
        else:
            if scale_multiplier is None:
                scale_multiplier = 2.5  # Valor por defecto más grande
            print(f"⚠️  No se pudieron calcular distancias, usando multiplicador: {scale_multiplier}")
        
        # 9. ESCALA FINAL (combinar escala base con multiplicador)
        scale = scale_base * scale_multiplier
        
        # 10. Verificar que la escala sea razonable
        if scale > 5.0:
            print(f"⚠️  Escala muy grande ({scale:.2f}), limitando a 5.0")
            scale = 5.0
        elif scale < 0.2:
            print(f"⚠️  Escala muy pequeña ({scale:.2f}), aumentando a 0.5")
            scale = 0.5
    
        # 11. Calcular traslación
        # Para calcular la traslación, necesitamos aplicar la rotación y escala al centroide del modelo
        # y ver la diferencia con el centroide objetivo
        translation = target_centroid - scale * (rotation @ model_centroid)
    
        # 12. Aplicar transformación
        self.model_rotation = rotation
        self.model_scale = scale
        self.model_translation = translation
    
        print(f"✅ Transformación final:")
        print(f"   Escala base (Procrustes): {scale_base:.4f}")
        print(f"   Multiplicador: {scale_multiplier:.4f}")
        print(f"   Escala final: {scale:.4f}")
        print(f"   Traslación: {translation}")
    
        # Verificar alineación
        self.verify_alignment_quality(landmarks_3d)
    
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

    def check_model_orientation(self):
        """Verifica y corrige la orientación inicial del modelo"""
        if self.obj_model is None:
            return

        # Verificar posición de hombros
        ls_idx = self.ANCHOR_VERTEX_IDS["left_shoulder"]
        rs_idx = self.ANCHOR_VERTEX_IDS["right_shoulder"]

        if ls_idx < len(self.obj_model.vertices) and rs_idx < len(self.obj_model.vertices):
            left_shoulder = np.array(self.obj_model.vertices[ls_idx])
            right_shoulder = np.array(self.obj_model.vertices[rs_idx])

            # Verificar si los hombros están invertidos (X positivo debería ser derecha)
            if left_shoulder[0] > right_shoulder[0]:
                print("⚠️  Modelo tiene hombros invertidos, aplicando rotación de 180° en Y")
                # Aplicar rotación de 180° en Y solo si es necesario
                rot_180_y = np.array([
                    [-1.0, 0.0,  0.0],
                    [ 0.0, 1.0,  0.0],
                    [ 0.0, 0.0, -1.0]
                ], dtype=np.float32)

                # Rotar todos los vértices del modelo
                for i in range(len(self.obj_model.vertices)):
                    self.obj_model.vertices[i] = rot_180_y @ np.array(self.obj_model.vertices[i])

    def verify_alignment_quality(self, landmarks_3d):
        """
        Verifica qué tan bien están alineados los puntos después de la transformación
        """
        errors = {}

        for name in self.ANCHOR_VERTEX_IDS.keys():
            if name in landmarks_3d:
                vertex_id = self.ANCHOR_VERTEX_IDS[name]
                if vertex_id < len(self.obj_model.vertices):
                    # Punto del modelo
                    model_point = np.array(self.obj_model.vertices[vertex_id])

                    # Aplicar transformación actual
                    transformed_point = self.model_scale * self.model_rotation @ model_point + self.model_translation

                    # Punto objetivo
                    target_point = np.array(landmarks_3d[name])

                    # Calcular error
                    error = np.linalg.norm(transformed_point - target_point)
                    errors[name] = error

        if errors:
            avg_error = np.mean(list(errors.values()))
            max_error = max(errors.values())

            print(f"📏 Error de alineación:")
            for name, error in errors.items():
                print(f"   {name}: {error:.4f}")
            print(f"   Promedio: {avg_error:.4f}, Máximo: {max_error:.4f}")

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