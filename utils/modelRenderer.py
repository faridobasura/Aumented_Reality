import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import os
import ctypes
import sys
from utils.app_args import args

from utils.textureRenderer import TextureRenderer

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
        if obj.has_texture_coordinates():
            print(f"✅ Modelo tiene coordenadas UV para texturas")
            self.render_mode = "texture"
        else:
            print(f"⚠️ Modelo NO tiene coordenadas UV - usando modo wireframe")
            self.render_mode = "wireframe"

        
        if obj:
            
            self.model_shoulder_dist = self._calculate_shoulder_distance()
            self.model_anchor_offset = self._calculate_anchor_offset()
            self.debug_draw_anchor_vertices()
            print(f"✅ Modelo cargado: {len(obj.vertices)} vértices")
        else:
            self.model_shoulder_dist = 0.0
            self.model_anchor_offset = np.zeros(3)

        
        self.texture_renderer = TextureRenderer()

        self.brightness = 0.2  # Valor por defecto (0.0 - 1.0)
        
        # Cargar texturas por defecto
        
        self._init_opengl()
        self._compile_shaders()
        self._init_gl_objects()
        self._upload_mesh()
        self._load_default_textures()

    
    def _load_shader_file(self, filepath):
        """Carga un shader desde archivo"""
        try:
            with open(filepath, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el shader {filepath}")
            # Fallback a shaders embebidos

    
    def _compile_shaders(self):
        """Compila shaders con soporte para texturas"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        shader_dir = os.path.join(parent_dir, "shaders")
        
        # 1. SHADERS PARA TEXTURAS (modelo principal)
        vertex_source = self._load_shader_file(os.path.join(shader_dir, "texture_vertex.glsl"))
        fragment_source = self._load_shader_file(os.path.join(shader_dir, "texture_fragment.glsl"))
        
        # Vertex shader
        self.vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vertex_shader, vertex_source)
        glCompileShader(self.vertex_shader)
        
        if not glGetShaderiv(self.vertex_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(self.vertex_shader)
            raise RuntimeError(f"Error compilando vertex shader:\n{error}")
        
        # Fragment shader
        self.fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fragment_shader, fragment_source)
        glCompileShader(self.fragment_shader)
        
        if not glGetShaderiv(self.fragment_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(self.fragment_shader)
            raise RuntimeError(f"Error compilando fragment shader:\n{error}")
        
        # Programa principal (para texturas)
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, self.vertex_shader)
        glAttachShader(self.shader_program, self.fragment_shader)
        glLinkProgram(self.shader_program)
        
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program)
            raise RuntimeError(f"Error linkando shader program:\n{error}")
        
        # Obtener ubicaciones de uniformes para el shader principal
        self.model_loc = glGetUniformLocation(self.shader_program, "model")
        self.view_loc = glGetUniformLocation(self.shader_program, "view")
        self.proj_loc = glGetUniformLocation(self.shader_program, "projection")
        self.use_texture_loc = glGetUniformLocation(self.shader_program, "useTexture")
        self.brightness_loc = glGetUniformLocation(self.shader_program, "brightness")
    
        
        print("✅ Shaders principales compilados con soporte para texturas")
        
        # 2. SHADERS PARA DEBUG/ANCLAS (si está en modo debug)
        if args.debug:
            self._compile_debug_shaders()
    
    def _compile_debug_shaders(self):
        """Compila shaders para debug/anclas"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        shader_dir = os.path.join(parent_dir, "shaders")
        
        # Cargar shaders para debug
        debug_vertex_source = self._load_shader_file(os.path.join(shader_dir, "vertex.glsl"))
        debug_fragment_source = self._load_shader_file(os.path.join(shader_dir, "fragment.glsl"))
        
        # Vertex shader para debug
        self.debug_vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.debug_vertex_shader, debug_vertex_source)
        glCompileShader(self.debug_vertex_shader)
        
        if not glGetShaderiv(self.debug_vertex_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(self.debug_vertex_shader)
            print(f"⚠️ Error compilando debug vertex shader:\n{error}")
            self.debug_shader_program = None
            return
        
        # Fragment shader para debug
        self.debug_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.debug_fragment_shader, debug_fragment_source)
        glCompileShader(self.debug_fragment_shader)
        
        if not glGetShaderiv(self.debug_fragment_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(self.debug_fragment_shader)
            print(f"⚠️ Error compilando debug fragment shader:\n{error}")
            self.debug_shader_program = None
            return
        
        # Programa para debug
        self.debug_shader_program = glCreateProgram()
        glAttachShader(self.debug_shader_program, self.debug_vertex_shader)
        glAttachShader(self.debug_shader_program, self.debug_fragment_shader)
        glLinkProgram(self.debug_shader_program)
        
        if not glGetProgramiv(self.debug_shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.debug_shader_program)
            print(f"⚠️ Error linkando debug shader program:\n{error}")
            self.debug_shader_program = None
        else:
            # Obtener ubicaciones de uniformes para debug
            self.debug_model_loc = glGetUniformLocation(self.debug_shader_program, "model")
            self.debug_view_loc = glGetUniformLocation(self.debug_shader_program, "view")
            self.debug_proj_loc = glGetUniformLocation(self.debug_shader_program, "projection")
            print("✅ Shaders de debug compilados")

    def _init_opengl(self):
        """Inicializa contexto OpenGL con GLFW"""
        if not glfw.init():
            raise RuntimeError("No se pudo inicializar GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        self.window = glfw.create_window(1, 1, "Hidden Render Context", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("No se pudo crear ventana GLFW oculta")

        glfw.make_context_current(self.window)
        glfw.hide_window(self.window)

        print(f"✅ Contexto OpenGL creado")

    def _load_default_textures(self):
        """Carga texturas iniciales"""

        # Cargar textura de camiseta si existe
        texture_path = os.path.expanduser('~/AR_python/Aumented_Reality/models/textures/white_solid.png')
        if os.path.exists(texture_path):
            if self.texture_renderer.load_texture("t_shirt", texture_path):
                print(f"✅ Textura de camiseta cargada: {texture_path}")
                self.texture_renderer.set_active_texture("t_shirt")
            else:
                print(f"⚠️ No se pudo cargar textura: {texture_path}")
        else:
            print(f"⚠️ No se encontró textura en: {texture_path}")
    
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
        SCALE_FACTOR = 1.5  # Aumentar este valor para hacer el modelo más grande
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

    def load_texture(self, name, filepath):
        """Carga una textura desde archivo"""
        success = self.texture_renderer.load_texture(name, filepath)
        if success:
            print(f"✅ Textura '{name}' cargada desde {filepath}")
        return success
    
    def set_texture(self, texture_name):
        """Establece la textura activa"""
        success = self.texture_renderer.set_active_texture(texture_name)
        if success and self.render_mode != "textured":
            self.set_render_mode("textured")
        return success
    
    def set_render_mode(self, mode):
        """Cambia el modo de renderizado"""
        valid_modes = ["wireframe", "textured", "solid"]
        if mode in valid_modes:
            self.render_mode = mode
            print(f"🎨 Modo de renderizado cambiado a: {mode}")
        else:
            print(f"❌ Modo inválido: {mode}. Usando 'wireframe'")
            self.render_mode = "wireframe"
    
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
        """Sube malla a GPU con posiciones y coordenadas UV"""
        if self.obj_model is None:
            return

        # Expandir caras para obtener posiciones y UVs
        if self.obj_model.has_texture_coordinates():
            positions, tex_coords = self._expand_faces_with_uvs()
        else:
            positions = self._expand_faces()
            tex_coords = np.zeros((len(positions), 2), dtype=np.float32)  # UVs por defecto
        
        self.vertex_count = len(positions)
        
        # Intercalar posiciones y coordenadas UV
        # Formato: [x, y, z, u, v]
        interleaved = np.zeros((self.vertex_count, 5), dtype=np.float32)
        interleaved[:, 0:3] = positions  # Posiciones (x, y, z)
        interleaved[:, 3:5] = tex_coords  # Coordenadas UV (u, v)
        
        # Aplanar el array
        vertex_data = interleaved.flatten()
        
        # VAO/VBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        
        # STRIDE = 5 floats * 4 bytes cada uno = 20 bytes
        stride = 5 * 4
        
        # Atributo 0: posición (3 floats)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
        
        # Atributo 1: coordenadas UV (2 floats) - offset 12 bytes
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        
        glBindVertexArray(0)
        
        print(f"✅ Malla subida a GPU: {self.vertex_count} vértices")
        if self.obj_model.has_texture_coordinates():
            print(f"   Incluye coordenadas UV para texturas")

    def _expand_faces_with_uvs(self):
        """Convierte caras a triángulos con coordenadas UV"""
        tri_verts = []
        tri_uvs = []
        
        for face, face_uv in zip(self.obj_model.faces, self.obj_model.face_uvs):
            if len(face) == 3:
                # Triángulo simple
                for i in range(3):
                    vertex_idx = face[i]
                    uv_idx = face_uv[i] if i < len(face_uv) and face_uv[i] >= 0 else 0
                    
                    tri_verts.append(self.obj_model.vertices[vertex_idx])
                    if uv_idx >= 0 and uv_idx < len(self.obj_model.uvs):
                        tri_uvs.append(self.obj_model.uvs[uv_idx])
                    else:
                        tri_uvs.append([0.0, 0.0])
            
            elif len(face) == 4:
                # Cuadrilátero - dividir en 2 triángulos
                # Triángulo 1: v0, v1, v2
                for i in [0, 1, 2]:
                    vertex_idx = face[i]
                    uv_idx = face_uv[i] if i < len(face_uv) and face_uv[i] >= 0 else 0
                    
                    tri_verts.append(self.obj_model.vertices[vertex_idx])
                    if uv_idx >= 0 and uv_idx < len(self.obj_model.uvs):
                        tri_uvs.append(self.obj_model.uvs[uv_idx])
                    else:
                        tri_uvs.append([0.0, 0.0])
                
                # Triángulo 2: v0, v2, v3
                for i in [0, 2, 3]:
                    vertex_idx = face[i]
                    uv_idx = face_uv[i] if i < len(face_uv) and face_uv[i] >= 0 else 0
                    
                    tri_verts.append(self.obj_model.vertices[vertex_idx])
                    if uv_idx >= 0 and uv_idx < len(self.obj_model.uvs):
                        tri_uvs.append(self.obj_model.uvs[uv_idx])
                    else:
                        tri_uvs.append([0.0, 0.0])
        
        return np.array(tri_verts, dtype=np.float32), np.array(tri_uvs, dtype=np.float32)
    
    def _expand_faces(self):
        """Convierte caras a triángulos (sin UVs)"""
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
        fov = 50.0  # Reducido de 60° a 45° para menos distorsión
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
        """Dibuja el modelo con shaders y texturas"""
        glUseProgram(self.shader_program)

        # Pasar matrices a shaders
        if hasattr(self, 'model_matrix'):
            glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.model_matrix.T)
            glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view_matrix.T)
            glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.proj_matrix.T)

        # Configurar si usar textura
        use_texture = (self.render_mode == "textured" and 
                      self.obj_model.has_texture_coordinates())
        glUniform1i(self.use_texture_loc, 1 if use_texture else 0)

        # NUEVO: Pasar valor de brightness al shader
        if hasattr(self, 'brightness_loc') and self.brightness_loc != -1:
            glUniform1f(self.brightness_loc, self.brightness)

        # Vincular textura si es necesario
        if use_texture:
            self.texture_renderer.bind_active_texture()
            glUniform1i(glGetUniformLocation(self.shader_program, "textureSampler"), 0)
        else:
            self.texture_renderer.unbind_texture()

        glBindVertexArray(self.vao)

        # Configurar modo de renderizado
        if self.render_mode == "wireframe":
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(1.0)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

        # Limpiar
        glBindVertexArray(0)
        if use_texture:
            self.texture_renderer.unbind_texture()
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
        if args.debug:
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

        # Si no hay shader de debug, usar el principal (pero sin texturas)
        if not hasattr(self, 'debug_shader_program') or self.debug_shader_program is None:
            print("⚠️ Usando shader principal para debug (sin texturas)")
            self._draw_anchor_with_main_shader()
            return

        # Asegurar que tenemos matrices válidas
        if not hasattr(self, 'model_matrix') or self.model_matrix is None:
            self._apply_transform()

        # Usar el shader de debug
        glUseProgram(self.debug_shader_program)

        # Configurar matrices
        glUniformMatrix4fv(self.debug_model_loc, 1, GL_FALSE, self.model_matrix.T)
        glUniformMatrix4fv(self.debug_view_loc, 1, GL_FALSE, self.view_matrix.T)
        glUniformMatrix4fv(self.debug_proj_loc, 1, GL_FALSE, self.proj_matrix.T)


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
            SCALE_ADJUSTMENT = 1.55  # ✅ Cambia este valor entre 1.0 y 3.0
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
    
    def set_brightness(self, value):
        """
        Establece el brillo adicional

        Args:
            value (float): Valor entre 0.0 y 1.0
                          0.0 = sin brillo extra
                          0.2 = brillo moderado (recomendado para texturas oscuras)
                          0.5 = brillo alto
                          1.0 = brillo máximo
        """
        self.brightness = max(0.0, min(1.0, value))  # Clamp entre 0 y 1
        print(f"💡 Brillo ajustado a: {self.brightness:.2f}")

    def increase_brightness(self, step=0.1):
        """Incrementa el brillo"""
        self.set_brightness(self.brightness + step)

    def decrease_brightness(self, step=0.1):
        """Decrementa el brillo"""
        self.set_brightness(self.brightness - step)
    

    def cleanup(self):
        """Limpia recursos OpenGL y GLFW"""
        print("🧹 Limpiando recursos OpenGL...")

    
        if hasattr(self, 'texture_renderer'):
            self.texture_renderer.cleanup()

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