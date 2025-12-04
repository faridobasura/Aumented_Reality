import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import os
import ctypes
import sys

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
        """Transformación FINAL corregida - Cámara simple mirando hacia -Z"""
        if self.model_translation.ndim > 1:
            self.model_translation = self.model_translation.flatten()

        print(f"\n=== DEBUG _apply_transform ===")

        # 1. ESCALA
        scale_mat = np.eye(4)
        scale_mat[:3, :3] *= self.model_scale

        # 2. ROTACIÓN (viene de MediaPipe, ya corregida)
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = self.model_rotation

        # 3. TRASLACIÓN
        trans_mat = np.eye(4)
        trans_mat[:3, 3] = self.model_translation.reshape(3)

        # 4. ORDEN: Traslación * Rotación * Escala
        self.model_matrix = trans_mat @ rot_mat @ scale_mat

        # 5. VISTA - CORREGIDA: Cámara simple mirando hacia -Z (estándar OpenGL)
        # NO aplicar rotación extra a la cámara
        self.view_matrix = np.eye(4)
        self.view_matrix[2, 3] = -5.0  # Cámara en Z = -3.0, mirando hacia -Z (origen)

        # 6. PROYECCIÓN
        fov = 60.0
        aspect = self.width / max(self.height, 1)
        near = 0.1
        far = 100.0
        self.proj_matrix = self._perspective_projection(fov, aspect, near, far)

        print(f"Traslación: {self.model_translation}")
        print(f"Escala: {self.model_scale}")

        # DEBUG: Calcular posición final de un vértice
        test_vertex = np.array([0, 0, 0, 1])  # Origen
        final_pos = self.proj_matrix @ self.view_matrix @ self.model_matrix @ test_vertex
        print(f"Posición del origen en clip space: {final_pos}")
        print("================================\n")

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
        CON CORRECCIÓN DE COORDENADAS
        """
        if self.obj_model is None:
            print("❌ No hay modelo cargado")
            return None, None, None

        # 1. Convertir landmarks de MediaPipe a nuestro sistema de coordenadas
        # MediaPipe: Y abajo -> Nuestro sistema: Y arriba
        corrected_landmarks = {}
        print("\n🎯 CONVERSIÓN COORDENADAS MEDIAPIPE -> OPENGL:")

        for name, point in landmarks_3d.items():
            corrected_point = np.array(point, dtype=np.float32)

            # CRÍTICO: Invertir Y para que sea hacia arriba
            corrected_point[1] = -corrected_point[1]

            # CRÍTICO: Invertir Z para que sea compatible con cámara OpenGL
            # OpenGL mira hacia -Z, así que invertimos para que el modelo sea visible
            corrected_point[2] = -corrected_point[2]

            corrected_landmarks[name] = corrected_point
            print(f"   {name}: {point} -> {corrected_point}")

        # 2. Continuar con el proceso de alineación usando los landmarks corregidos
        model_points = []
        target_points = []

        # Variables para cálculos de hombros
        model_left_shoulder = None
        model_right_shoulder = None
        target_left_shoulder = None
        target_right_shoulder = None
    
        for name in self.ANCHOR_VERTEX_IDS.keys():
            if name in corrected_landmarks:
                vertex_id = self.ANCHOR_VERTEX_IDS[name]
                if vertex_id < len(self.obj_model.vertices):
                    # Punto del modelo
                    model_point = np.array(self.obj_model.vertices[vertex_id])
                    model_points.append(model_point)
    
                    # Punto objetivo (landmark)
                    target_point = np.array(corrected_landmarks[name])
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
        # Asegurar que es una rotación propia (det=1)
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = Vt.T @ U.T

        # 6.2 Segundo: Rotar 180° en Y para que mire hacia adelante (no hacia atrás)
        rot_180_y = np.array([
            [-1.0, 0.0,  0.0],
            [ 0.0, 1.0,  0.0],
            [ 0.0, 0.0, -1.0]
        ], dtype=np.float32)

        # 6.3 Combinar ambas correcciones: Primero X, luego Y
        rotation_corrected = rot_180_y @ rotation

        print(f"\n🎯 ROTACIÓN CORREGIDA (180° Y):")
        for i in range(3):
            print(f"   [{rotation_corrected[i,0]:.3f}, {rotation_corrected[i,1]:.3f}, {rotation_corrected[i,2]:.3f}]")

        rotation = rotation_corrected

         # 7. **CALCULAR DISTANCIAS DESPUÉS DE LA ROTACIÓN**
        # Esto es importante porque la rotación afecta las posiciones relativas
        if model_left_shoulder is not None and model_right_shoulder is not None:
            # Calcular hombros rotados
            left_shoulder_rotated = rotation @ model_left_shoulder
            right_shoulder_rotated = rotation @ model_right_shoulder
            model_shoulder_dist = np.linalg.norm(right_shoulder_rotated - left_shoulder_rotated)
        else:
            model_shoulder_dist = self._calculate_average_distance(model_points_arr)

        if target_left_shoulder is not None and target_right_shoulder is not None:
            target_shoulder_dist = np.linalg.norm(target_right_shoulder - target_left_shoulder)
        else:
            target_shoulder_dist = self._calculate_average_distance(target_points_arr)

        print(f"\n📏 DISTANCIAS DESPUÉS DE ROTACIÓN:")
        print(f"   Modelo (rotado): {model_shoulder_dist:.4f}")
        print(f"   Landmarks: {target_shoulder_dist:.4f}")
    
        # 8. Calcular escala base usando Procrustes
        model_norm = np.trace(model_centered_T @ model_centered_T.T)
        if model_norm > 0:
            scale_base = np.trace(target_centered_T.T @ rotation @ model_centered_T) / model_norm
        else:
            scale_base = 1.0

        print(f"\n📊 ESCALA PROCRUSTES: {scale_base:.4f}")

        # 9. **CALCULAR ESCALA CON AJUSTE DE TAMAÑO**
        if model_shoulder_dist > 0 and target_shoulder_dist > 0:
            # Calcular factor basado en la relación de distancias
            distance_ratio = target_shoulder_dist / model_shoulder_dist

            # FACTOR DE AJUSTE - Incrementar para que ocupe más espacio
            # Si el modelo está muy pequeño, prueba con valores más altos (5.0-8.0)
            ADJUSTMENT_FACTOR = 6.0  # <-- Aumenta este valor si sigue pequeño

            if scale_multiplier is None:
                scale_multiplier = distance_ratio * ADJUSTMENT_FACTOR

            print(f"📐 Relación de distancia: {distance_ratio:.4f}")
            print(f"🎯 Multiplicador calculado: {scale_multiplier:.4f}")
        else:
            if scale_multiplier is None:
                scale_multiplier = 4.0  # Valor por defecto más grande
            print(f"⚠️  Usando multiplicador por defecto: {scale_multiplier}")

        # 10. ESCALA FINAL
        scale = scale_base * scale_multiplier

        # 11. Verificar que la escala sea razonable
        if scale > 8.0:
            print(f"⚠️  Escala muy grande ({scale:.2f}), limitando a 8.0")
            scale = 8.0
        elif scale < 0.5:
            print(f"⚠️  Escala muy pequeña ({scale:.2f}), aumentando a 1.0")
            scale = 1.0

        # 12. Calcular traslación
        translation = target_centroid - scale * (rotation @ model_centroid)

        # 13. Aplicar transformación
        self.model_rotation = rotation
        self.model_scale = scale
        self.model_translation = translation

        print(f"\n✅ TRANSFORMACIÓN FINAL:")
        print(f"   Escala: {scale:.4f} (base: {scale_base:.4f} * mult: {scale_multiplier:.4f})")
        print(f"   Traslación: {translation}")

        # 14. Verificar alineación con los landmarks CORREGIDOS
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

    def check_model_orientation(self):
        """Solo corregir hombros invertidos - NO aplicar rotación fija"""
        if self.obj_model is None:
            return

        print("🔄 Verificando orientación del modelo...")

        # Solo verificar si los hombros están invertidos
        ls_idx = self.ANCHOR_VERTEX_IDS["left_shoulder"]
        rs_idx = self.ANCHOR_VERTEX_IDS["right_shoulder"]

        if ls_idx < len(self.obj_model.vertices) and rs_idx < len(self.obj_model.vertices):
            left_shoulder = np.array(self.obj_model.vertices[ls_idx])
            right_shoulder = np.array(self.obj_model.vertices[rs_idx])

            # Verificar si los hombros están invertidos
            if left_shoulder[0] > right_shoulder[0]:
                print("⚠️  Modelo tiene hombros invertidos, aplicando rotación de 180° en Y")
                rot_180_y = np.array([
                    [-1.0, 0.0,  0.0],
                    [ 0.0, 1.0,  0.0],
                    [ 0.0, 0.0, -1.0]
                ], dtype=np.float32)

                for i in range(len(self.obj_model.vertices)):
                    self.obj_model.vertices[i] = rot_180_y @ np.array(self.obj_model.vertices[i])

            # Debug después de la corrección
            left_after = np.array(self.obj_model.vertices[ls_idx])
            right_after = np.array(self.obj_model.vertices[rs_idx])
            print(f"✅ Después de corrección:")
            print(f"   Izquierda: X={left_after[0]:.3f}, Z={left_after[2]:.3f}")
            print(f"   Derecha: X={right_after[0]:.3f}, Z={right_after[2]:.3f}")

    def verify_alignment_quality(self, landmarks_3d_corrected):
        """
        Verifica qué tan bien están alineados los puntos después de la transformación
        Usa landmarks ya corregidos (Y invertido, Z invertido)
        """
        errors = {}

        for name in self.ANCHOR_VERTEX_IDS.keys():
            if name in landmarks_3d_corrected:
                vertex_id = self.ANCHOR_VERTEX_IDS[name]
                if vertex_id < len(self.obj_model.vertices):
                    # Punto del modelo
                    model_point = np.array(self.obj_model.vertices[vertex_id])

                    # Aplicar transformación actual
                    transformed_point = self.model_scale * self.model_rotation @ model_point + self.model_translation

                    # Punto objetivo (YA CORREGIDO)
                    target_point = np.array(landmarks_3d_corrected[name])

                    # Calcular error
                    error = np.linalg.norm(transformed_point - target_point)
                    errors[name] = error

        if errors:
            avg_error = np.mean(list(errors.values()))
            max_error = max(errors.values())

            print(f"\n📏 ERROR DE ALINEACIÓN (usando landmarks corregidos):")
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