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
            self.model_shoulder_dist = self._calculate_shoulder_distance()
            self.model_anchor_offset = self._calculate_anchor_offset()
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

        # Configurar ventana oculta CON CORE PROFILE
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)  # ¡IMPORTANTE!
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)  # ¡IMPORTANTE para macOS/Linux!

        self.window = glfw.create_window(1, 1, "Hidden", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("No se pudo crear ventana GLFW")

        glfw.make_context_current(self.window)

        # Verificar que OpenGL 3.3 está disponible
        print(f"✅ OpenGL version: {glGetString(GL_VERSION).decode()}")
        print(f"✅ GLSL version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}")
    
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
        self.model_translation = np.array(translation, dtype=np.float32)
        self.model_rotation = np.array(rotation_mat, dtype=np.float32)
        self.model_scale = float(scale)
    
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
        self.model_matrix = np.eye(4, dtype=np.float32)

        scale_mat = np.eye(4)
        scale_mat[:3, :3] *= self.model_scale

        rot_180_y = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0, 0.0],
            [ 0.0, 0.0, -1.0, 0.0],
            [ 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        rot_180_x = np.array([
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0]
        ], dtype=np.float32)

        rot_mat = rot_180_y @ rot_180_x

        user_rot_mat = np.eye(4)
        user_rot_mat[:3, :3] = self.model_rotation

        # TRASLACIÓN
        trans_mat = np.eye(4)
        trans_mat[:3, 3] = self.model_translation

        self.model_matrix = trans_mat @ user_rot_mat @ rot_mat @ scale_mat

        self.view_matrix = np.eye(4)

        self.proj_matrix = np.eye(4)
        self.proj_matrix[0, 0] = 2.0 / 1.0
        self.proj_matrix[1, 1] = 2.0 / 1.0
        self.proj_matrix[2, 2] = -2.0 / 10.0
        self.proj_matrix[3, 2] = -1.0
    
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
        self._apply_transform()  # ¡ESTO ES IMPORTANTE!     
        # 2. Actualizar matriz de proyección según viewport
        aspect = self.width / max(self.height, 1)
        near, far = 0.1, 100.0      
        # Proyección ortográfica ajustada
        self.proj_matrix = np.eye(4)
        if aspect >= 1.0:
            self.proj_matrix[0, 0] = 1.0 / aspect
            self.proj_matrix[1, 1] = 1.0
        else:
            self.proj_matrix[0, 0] = 1.0
            self.proj_matrix[1, 1] = aspect     
        self.proj_matrix[2, 2] = -2.0 / (far - near)
        self.proj_matrix[3, 2] = -(far + near) / (far - near)       
        # 3. Renderizar modelo principal
        self._draw_model()      
        # 4. ¡NUEVO! Dibujar vértices ancla con colores
        self.draw_anchor_vertices_with_labels()  # O usa debug_draw_anchors_simple()        
        # 5. Leer píxeles
        buffer = glReadPixels(0, 0, self.width, self.height,
                              GL_RGBA, GL_UNSIGNED_BYTE)        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)        
        # 6. Convertir a numpy
        img = np.frombuffer(buffer, dtype=np.uint8)\
                .reshape(self.height, self.width, 4)        
        img = cv2.flip(img, 0)
        return img  
    
    def draw_anchor_vertices_with_labels(self):
        """Dibuja vértices ancla con colores y etiquetas"""
        if self.obj_model is None:
            return

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