# utils/modernGLRenderer.py - VERSIÓN CORREGIDA PARA TU OBJECTLOADER
import moderngl
import numpy as np
import cv2
import pyrr


class ModernGLRenderer:
    def __init__(self, obj_loader, width=640, height=480):
        self.ctx = moderngl.create_standalone_context(require=330)
        self.width = width
        self.height = height
        self.obj_loader = obj_loader

        print(f"\n🔄 Inicializando ModernGLRenderer: {width}x{height}")
        
        # Verificar que el loader tenga datos
        if not hasattr(obj_loader, 'vertices') or len(obj_loader.vertices) == 0:
            print("❌ ERROR: obj_loader no tiene vértices!")
            return

        print(f"   Vértices: {len(obj_loader.vertices)}")
        print(f"   Caras: {len(obj_loader.faces)}")
        print(f"   Coord. textura: {len(obj_loader.tex_coords)}")
        print(f"   Textura cargada: {obj_loader.texture_loaded}")
        
        # ---------------------------
        # SHADERS - VERSIÓN CON/SIN TEXTURA
        # ---------------------------
        if obj_loader.texture_loaded and obj_loader.texture_cv is not None:
            print("🎨 Usando shaders con textura")
            vert_src = '''
            #version 330
            in vec3 in_pos;
            in vec2 in_texcoord;
            uniform mat4 mvp;
            out vec2 v_texcoord;
            
            void main() {
                gl_Position = mvp * vec4(in_pos, 1.0);
                v_texcoord = in_texcoord;
            }
            '''
            
            frag_src = '''
                #version 330
                in vec2 v_texcoord;
                uniform sampler2D tex;
                out vec4 fragColor;

                void main() {
                    // DEBUG: verificar si estamos mirando la cara frontal o trasera
                    if (!gl_FrontFacing) {
                        // Si es la parte trasera, usar color diferente
                        fragColor = vec4(1.0, 0.0, 0.0, 1.0);  // ROJO para trasera
                    } else {
                        // Parte frontal normal
                        vec4 tex_color = texture(tex, v_texcoord);
                        if (tex_color.a < 0.1) discard;
                        fragColor = tex_color;
                    }
                }
            '''
        else:
            print("🎨 Usando shaders sin textura (color sólido)")
            vert_src = '''
            #version 330
            in vec3 in_pos;
            uniform mat4 mvp;
            
            void main() {
                gl_Position = mvp * vec4(in_pos, 1.0);
            }
            '''
            
            frag_src = '''
            #version 330
            out vec4 fragColor;
            
            void main() {
                // Color azul para debug cuando no hay textura
                fragColor = vec4(0.2, 0.5, 1.0, 0.8);
            }
            '''
        
        self.prog = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=frag_src,
        )
        print("✅ Shaders compilados")

        # ---------------------------
        # PREPARAR DATOS DEL MODELO
        # ---------------------------
        # 1. Vértices
        verts = np.array(obj_loader.vertices, dtype="f4")
        print(f"   Vértices shape: {verts.shape}")
        
        # 2. Coordenadas de textura (si existen)
        texcoords_array = None
        if obj_loader.tex_coords and len(obj_loader.tex_coords) > 0:
            texcoords_array = np.array(obj_loader.tex_coords, dtype="f4")
            print(f"   Texcoords shape: {texcoords_array.shape}")
        
        # 3. Crear arrays interleaved (posición + texcoord)
        indices_list = []
        vertices_interleaved = []
        
        print("🔄 Procesando caras...")
        vertices_interleaved = []
        face_count = 0

        for face in obj_loader.faces:
            if len(face) >= 3:
                # TRIANGULACIÓN SIMPLE - asume que las caras ya son triángulos
                for i in range(len(face)):
                    v_data = face[i]
                    v_idx = v_data[0]
                    vt_idx = v_data[1] if len(v_data) > 1 else -1

                    # Posición
                    vertices_interleaved.extend(verts[v_idx])

                    # Coordenadas de textura
                    if vt_idx != -1 and texcoords_array is not None and vt_idx < len(texcoords_array):
                        vertices_interleaved.extend(texcoords_array[vt_idx])
                    else:
                        vertices_interleaved.extend([0.0, 0.0])

                face_count += 1

        print(f"✅ Vértices procesados: {len(vertices_interleaved)//5}")  # 3 pos + 2 tex
        print(f"   Caras procesadas como triángulos individuales: {face_count}")
        
        # Convertir a numpy
        vertices_np = np.array(vertices_interleaved, dtype="f4")
        
        # Crear VBO
        self.vbo = self.ctx.buffer(vertices_np.tobytes())
        print(f"✅ VBO creado: {vertices_np.size} floats ({vertices_np.size * 4} bytes)")
        print(f"   Triángulos procesados: {face_count}")
        
        # Configurar VAO
        if obj_loader.texture_loaded and texcoords_array is not None:
            # Con textura: posición (3 floats) + texcoord (2 floats) = stride 5 floats
            self.vao = self.ctx.vertex_array(
                self.prog,
                [(self.vbo, "3f 2f", "in_pos", "in_texcoord")],
            )
        else:
            # Sin textura: solo posición (3 floats)
            self.vao = self.ctx.vertex_array(
                self.prog,
                [(self.vbo, "3f", "in_pos")],
            )
            
        
        print("✅ VAO creado")

        # Después de crear el VAO, verifica:
        print(f"🔍 VAO info:")
        print(f"   Program: {self.prog}")
        print(f"   VBO size: {self.vbo.size}")
        print(f"   Vertex count: {len(vertices_interleaved) // 5}")  # 3 pos + 2 tex
        
        # Prueba renderizando solo los primeros N vértices
        test_vertex_count = min(100, len(vertices_interleaved) // 5)
        print(f"   Probando con {test_vertex_count} vértices...")

        # ---------------------------
        # TEXTURA (si existe)
        # ---------------------------
        self.texture = None
        if obj_loader.texture_loaded and obj_loader.texture_cv is not None:
            try:
                print("🖼️ Cargando textura en ModernGL...")
                tex_data = obj_loader.texture_cv
                
                # Asegurar formato correcto
                if tex_data.shape[2] == 3:  # RGB
                    tex_data = cv2.cvtColor(tex_data, cv2.COLOR_RGB2RGBA)
                
                print(f"   Textura shape: {tex_data.shape}")
                print(f"   Textura dtype: {tex_data.dtype}")
                
                # Crear textura OpenGL
                self.texture = self.ctx.texture(
                    (tex_data.shape[1], tex_data.shape[0]),  # width, height
                    tex_data.shape[2],  # components
                    tex_data.tobytes(),
                )
                self.texture.build_mipmaps()
                
                # Configurar sampler
                self.texture.use(0)
                if "tex" in self.prog:
                    self.prog["tex"].value = 0
                
                print("✅ Textura cargada en OpenGL")
                
            except Exception as e:
                print(f"❌ Error cargando textura: {e}")
                import traceback
                traceback.print_exc()

        # ---------------------------
        # FRAMEBUFFER
        # ---------------------------
        self.color_tex = self.ctx.texture((width, height), 4, dtype="f1")
        self.depth_tex = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex],
            depth_attachment=self.depth_tex,
        )
        print("✅ Framebuffer creado")

        # ---------------------------
        # MATRICES DE CÁMARA
        # ---------------------------
        aspect = width / height
    
        # Proyección perspectiva - igual que import_obj.py
        self.projection = pyrr.matrix44.create_perspective_projection(
            fovy=45.0,  # Mismo que gluPerspective(45.0)
            aspect=aspect, 
            near=0.1,   
            far=100.0, 
            dtype="f4"
        )

        # Vista de cámara - igual que import_obj.py
        self.view = pyrr.matrix44.create_look_at(
            eye=[0.0, 0.0, 5.0],      # Cámara en Z=5 como gluLookAt(0,0,5,...)
            target=[0.0, 0.0, 0.0],   # Mirando al origen
            up=[0.0, 1.0, 0.0],       # Arriba en Y
            dtype="f4"
        )

        print("✅ Matrices de cámara creadas (igual que import_obj.py)")

    # ------------------------------
    # RENDER SIMPLE (para testing)
    # ------------------------------
    def render_simple(self, model_matrix=None):
        """Renderiza el modelo con una cámara fija (para testing)"""
        try:
            print(f"\n🎬 Renderizando modelo...")
            
            # Usar matriz de modelo por defecto si no se proporciona
            if model_matrix is None:
                model_matrix = self._create_default_model_matrix()
            
            # MVP = Projection * View * Model
            mvp = self.projection @ self.view @ model_matrix
            
            # Configurar render state
            self.fbo.use()
            self.ctx.viewport = (0, 0, self.width, self.height)
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)  # Transparente
            self.ctx.clear(depth=1.0)
            
            # Habilitar blending para transparencia
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            
            self.ctx.wireframe = True
            self.ctx.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )
            
            # Pasar matriz al shader
            self.prog["mvp"].write(mvp.astype("f4").tobytes())
            
            # Renderizar
            self.vao.render()
            self.ctx.finish()
            
            # Leer resultado
            img = self.fbo.read(components=4, dtype="u1")
            img = np.frombuffer(img, dtype=np.uint8).reshape(self.height, self.width, 4)
            img = cv2.flip(img, 0)  # Voltear verticalmente
            
            # Convertir RGBA a BGR para OpenCV
            img_bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
            
            # Verificar si se renderizó algo
            alpha_channel = img[:, :, 3]
            visible_pixels = np.sum(alpha_channel > 10)
            print(f"   Píxeles visibles: {visible_pixels}")
            
            if visible_pixels == 0:
                print("   ⚠️  ¡No se ve nada! El modelo puede estar fuera de vista")
                # Añadir texto de debug
                cv2.putText(img_bgr, "DEBUG: Modelo no visible", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return img_bgr
            
        except Exception as e:
            print(f"❌ Error en render_simple: {e}")
            import traceback
            traceback.print_exc()
            
            # Crear imagen de error
            error_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(error_img, f"ERROR: {str(e)[:50]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return error_img

    def _create_default_model_matrix(self):
        """Crea una matriz de modelo por defecto que posicione el modelo correctamente"""
        import pyrr
        
        # Obtener bounding box del modelo
        bbox_min, bbox_max = self.obj_loader.get_bounding_box()
        
        # Calcular centro y tamaño
        center = [
            (bbox_min[0] + bbox_max[0]) / 2,
            (bbox_min[1] + bbox_max[1]) / 2,
            (bbox_min[2] + bbox_max[2]) / 2
        ]
        
        size = [
            bbox_max[0] - bbox_min[0],
            bbox_max[1] - bbox_min[1],
            bbox_max[2] - bbox_min[2]
        ]
        
        print(f"📐 Bounding Box:")
        print(f"   Centro: {center}")
        print(f"   Tamaño: {size}")
        
        # 1. Centrar el modelo en el origen
        T_center = pyrr.matrix44.create_from_translation(
            [-center[0], -center[1], -center[2]], 
            dtype='f4'
        )
        
        # 2. Escalar para que sea visible (ajusta según necesites)
        max_size = max(size[0], size[1], size[2], 0.001)
        scale = 1.5 / max_size  # Para que quepa en ~1.5 unidades
        S = pyrr.matrix44.create_from_scale(
            [scale, scale, scale], 
            dtype='f4'
        )
        
        # 3. Rotar para verlo mejor
        Rx = pyrr.matrix44.create_from_x_rotation(np.radians(-20), dtype='f4')
        Ry = pyrr.matrix44.create_from_y_rotation(np.radians(30), dtype='f4')
        
        # 4. Posicionar delante de la cámara
        T_pos = pyrr.matrix44.create_from_translation([0, 0, -1], dtype='f4')
        
        # Combinar: T_pos * Ry * Rx * S * T_center
        model_matrix = T_pos @ Ry @ Rx @ S @ T_center
        
        return model_matrix

    # ------------------------------
    # RENDER PARA AR (con matriz específica)
    # ------------------------------
    def render(self, model_matrix):
        """Renderiza el modelo con la matriz dada"""
        try:
            self.fbo.use()
            self.ctx.viewport = (0, 0, self.width, self.height)
            self.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

            # Configurar blending y depth test
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )

            # Calcular MVP
            mvp = self.projection @ self.view @ model_matrix
            self.prog["mvp"].write(mvp.astype("f4").tobytes())

            # Renderizar
            self.vao.render()
            self.ctx.finish()

            # Leer resultado
            img = self.fbo.read(components=4, dtype="u1")
            img = np.frombuffer(img, dtype=np.uint8).reshape(self.height, self.width, 4)
            img = cv2.flip(img, 0)

            return img
            
        except Exception as e:
            print(f"❌ Error en render: {e}")
            return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    # ------------------------------
    # OVERLAY SOBRE VIDEO (para AR)
    # ------------------------------
    def render_on_frame(self, frame, model_matrix):
        """Renderiza el modelo sobre un frame de video"""
        try:
            if model_matrix is None:
                model_matrix = np.eye(4, dtype='f4')

            # Asegurar que la matriz sea 4x4
            model_matrix = np.array(model_matrix, dtype='f4')
            if model_matrix.shape != (4, 4):
                print("⚠️ model_matrix inválida → usando identidad")
                model_matrix = np.eye(4, dtype='f4')

            # Renderizar modelo
            rgba = self.render(model_matrix)
            
            # Hacer blend con el frame
            alpha = rgba[:, :, 3][:, :, None] / 255.0
            out = frame.copy()
            out[:, :, :3] = frame[:, :, :3] * (1 - alpha) + rgba[:, :, :3] * alpha
            
            return out
            
        except Exception as e:
            print(f"❌ Error en render_on_frame: {e}")
            return frame

    # ------------------------------
    # CALCULAR MATRIZ PARA LANDMARKS
    # ------------------------------
    def compute_model_matrix(self, landmarks, frame_shape):
        """Calcula matriz de modelo basada en landmarks de MediaPipe"""
        try:
            h, w = frame_shape[:2]

            # Índices de MediaPipe:
            # 11: hombro izquierdo
            # 12: hombro derecho
            if len(landmarks) < 13:
                return np.eye(4, dtype='f4')

            ls = landmarks[11]
            rs = landmarks[12]

            # Calcular distancia entre hombros en píxeles
            dx = (rs.x - ls.x) * w
            dy = (rs.y - ls.y) * h
            shoulder_dist = max(np.sqrt(dx*dx + dy*dy), 1e-6)

            # Obtener distancia de hombros del modelo
            anchor_pos = self.obj_loader.get_anchor_positions()
            if 'left_shoulder' in anchor_pos and 'right_shoulder' in anchor_pos:
                ls_model = anchor_pos['left_shoulder']
                rs_model = anchor_pos['right_shoulder']
                model_shoulder_dist = np.sqrt(
                    (rs_model[0] - ls_model[0])**2 + 
                    (rs_model[1] - ls_model[1])**2 + 
                    (rs_model[2] - ls_model[2])**2
                )
            else:
                # Fallback: usar bounding box
                bbox_min, bbox_max = self.obj_loader.get_bounding_box()
                model_shoulder_dist = max(abs(bbox_max[0] - bbox_min[0]), 1e-6)

            # Calcular escala
            scale = shoulder_dist / max(model_shoulder_dist, 1e-6)
            scale = np.clip(scale, 0.1, 10.0)

            # Calcular rotación en Z
            angle_z = np.arctan2(dy, dx)

            # Posición central entre hombros
            cx = int((ls.x + rs.x) * 0.5 * w)
            cy = int((ls.y + rs.y) * 0.5 * h)

            # Crear matriz de transformación
            import pyrr
            S = pyrr.matrix44.create_from_scale([scale, scale, scale], dtype='f4')
            Rz = pyrr.matrix44.create_from_z_rotation(angle_z, dtype='f4')
            T = pyrr.matrix44.create_from_translation([cx, cy, 0], dtype='f4')

            return T @ Rz @ S

        except Exception as e:
            print(f"❌ Error en compute_model_matrix: {e}")
            return np.eye(4, dtype='f4')
    
    def render_wireframe(self):
        """Renderiza en modo wireframe como import_obj.py"""
        try:
            print("\n🔧 Renderizando wireframe...")
            
            # Configurar para wireframe
            self.ctx.wireframe = True
            
            # Matriz de modelo simple
            import pyrr
            
            # Escala basada en bounding box
            bbox_min, bbox_max = self.obj_loader.get_bounding_box()
            size = [
                bbox_max[0] - bbox_min[0],
                bbox_max[1] - bbox_min[1],
                bbox_max[2] - bbox_min[2]
            ]
            
            max_size = max(size)
            scale = 1.0 / max_size if max_size > 0 else 1.0
            
            # Crear matriz de modelo: trasladar en Z como import_obj.py
            S = pyrr.matrix44.create_from_scale([scale, scale, scale], dtype='f4')
            T = pyrr.matrix44.create_from_translation([0.0, 0.0, -5.0], dtype='f4')  # Igual que glTranslatef(0,0,-5)
            
            model_matrix = T @ S
            
            # MVP matrix
            mvp = self.projection @ self.view @ model_matrix
            
            # Render
            self.fbo.use()
            self.ctx.viewport = (0, 0, self.width, self.height)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.clear(depth=1.0)
            
            self.ctx.enable(moderngl.DEPTH_TEST)
            
            # Usar shader simple para wireframe
            self.prog["mvp"].write(mvp.astype("f4").tobytes())
            
            # Renderizar como puntos/lineas para wireframe
            self.vao.render(moderngl.LINES)
            self.ctx.finish()
            
            # Leer resultado
            img = self.fbo.read(components=4, dtype="u1")
            img = np.frombuffer(img, dtype=np.uint8).reshape(self.height, self.width, 4)
            img = cv2.flip(img, 0)
            
            # Convertir a BGR
            img_bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
            
            # Desactivar wireframe para próximos renders
            self.ctx.wireframe = False
            
            return img_bgr
            
        except Exception as e:
            print(f"❌ Error en render_wireframe: {e}")
            import traceback
            traceback.print_exc()
            return None