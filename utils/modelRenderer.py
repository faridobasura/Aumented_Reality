import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

# Vertex Shader simple
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

# Fragment Shader simple
FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);  // Color blanco
}
"""


class ModelRenderer:
    def __init__(self, width=512, height=512, obj=None):
        self.width = width
        self.height = height
        self.obj_model = obj
        self.vertex_count = 0
        
        # VÉRTICES CORRECTOS
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
        
        # Inicializar OpenGL sin GLUT
        self._init_opengl()
        self._compile_shaders()  # Nuevo: compilar shaders

        self._init_gl_objects()
        self._upload_mesh()
    
    def _compile_shaders(self):
        """Compila shaders OpenGL"""
        # Crear y compilar vertex shader
        self.vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vertex_shader, VERTEX_SHADER)
        glCompileShader(self.vertex_shader)
        
        # Verificar compilación
        if not glGetShaderiv(self.vertex_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(self.vertex_shader)
            raise RuntimeError(f"Error compilando vertex shader: {error}")
        
        # Crear y compilar fragment shader
        self.fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fragment_shader, FRAGMENT_SHADER)
        glCompileShader(self.fragment_shader)
        
        if not glGetShaderiv(self.fragment_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(self.fragment_shader)
            raise RuntimeError(f"Error compilando fragment shader: {error}")
        
        # Crear programa
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, self.vertex_shader)
        glAttachShader(self.shader_program, self.fragment_shader)
        glLinkProgram(self.shader_program)
        
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program)
            raise RuntimeError(f"Error linkando shader program: {error}")
        
        # Obtener ubicaciones de uniformes
        self.model_loc = glGetUniformLocation(self.shader_program, "model")
        self.view_loc = glGetUniformLocation(self.shader_program, "view")
        self.proj_loc = glGetUniformLocation(self.shader_program, "projection")

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
        """Sube malla a GPU"""
        if self.obj_model is None:
            return
        
        # Expandir caras
        verts = self._expand_faces()
        self.vertex_count = len(verts)
        
        # VAO/VBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        glBindVertexArray(0)
    
    def _expand_faces(self):
        """Convierte caras a triángulos"""
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
        return np.array(tri_verts, dtype=np.float32)
    
    def _apply_transform(self):
        """Crea matrices de transformación"""
        # Matriz modelo (escala + rotación + traslación)
        self.model_matrix = np.eye(4, dtype=np.float32)
        
        # Escala
        scale_mat = np.eye(4)
        scale_mat[:3, :3] *= self.model_scale
        
        # Rotación (3x3 a 4x4)
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = self.model_rotation
        
        # Traslación
        trans_mat = np.eye(4)
        trans_mat[:3, 3] = self.model_translation
        
        # Model = T * R * S
        self.model_matrix = trans_mat @ rot_mat @ scale_mat
        
        # Matriz vista (cámara simple)
        self.view_matrix = np.eye(4)
        
        # Matriz proyección (ortográfica)
        self.proj_matrix = np.eye(4)
        # Proyección ortográfica: left, right, bottom, top, near, far
        self.proj_matrix[0, 0] = 2.0 / 1.0  # right-left
        self.proj_matrix[1, 1] = 2.0 / 1.0  # top-bottom
        self.proj_matrix[2, 2] = -2.0 / 10.0  # far-near (negativo para Z)
        self.proj_matrix[3, 2] = -1.0  # -(far+near)/(far-near)
    
    def set_viewport(self, w, h):
        self.width = w
        self.height = h
        self._init_gl_objects()
    
    def _draw_model(self):
        """Dibuja el modelo con shaders - versión segura"""
        glUseProgram(self.shader_program)

        # Pasar matrices a shaders
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.model_matrix.T)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view_matrix.T)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.proj_matrix.T)

        # Pasar modo de renderizado
        render_mode_uniform = glGetUniformLocation(self.shader_program, "render_mode")
        mode_value = 0 if self.render_mode == "wireframe" else 1
        glUniform1i(render_mode_uniform, mode_value)

        glBindVertexArray(self.vao)

        # Configurar modo de renderizado SEGURO
        if self.render_mode == "wireframe":
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            # SOLUCIÓN: Usar 1.0 en lugar de 1.5
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

        # 3. Renderizar
        self._draw_model()

        # 4. Leer píxeles
        buffer = glReadPixels(0, 0, self.width, self.height,
                              GL_RGBA, GL_UNSIGNED_BYTE)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # 5. Convertir a numpy
        img = np.frombuffer(buffer, dtype=np.uint8)\
                .reshape(self.height, self.width, 4)

        img = cv2.flip(img, 0)
        return img