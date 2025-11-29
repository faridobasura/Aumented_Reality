# utils/modernGLRenderer.py
import moderngl
import numpy as np
import cv2
import pyrr

class ModernGLRenderer:
    def __init__(self, obj_loader, width=640, height=480):
        # Contexto standalone
        self.ctx = moderngl.create_standalone_context(require=330)
        self.width = width
        self.height = height

        # Leer shaders
        with open("shaders/texture.vert") as f:
            vert_src = f.read()
        with open("shaders/texture.frag") as f:
            frag_src = f.read()

        self.prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # Construir buffers (unir posición+uv por índice compuesto)
        # faces: lista de caras; cada elemento: [[v_idx, vt_idx, vn_idx], ...]
        verts = obj_loader.vertices
        uvs = obj_loader.tex_coords    # tu loader usa tex_coords
        faces = obj_loader.faces

        # Mapa (v_idx, vt_idx) -> new index
        index_map = {}
        positions = []
        texcoords = []
        indices = []

        def add_vertex(v_idx, vt_idx):
            key = (v_idx, vt_idx)
            if key in index_map:
                return index_map[key]
            pos = verts[v_idx]
            # uv puede faltar; usa (0,0)
            uv = [0.0, 0.0]
            if vt_idx is not None and vt_idx >= 0 and vt_idx < len(uvs):
                uv = uvs[vt_idx]
            index = len(positions)
            positions.append(pos)
            texcoords.append(uv)
            index_map[key] = index
            return index

        # Triangular caras (fan triangulation)
        for face in faces:
            if len(face) < 3:
                continue
            # generar indices de la cara
            base_idx = add_vertex(face[0][0], face[0][1])
            for i in range(1, len(face)-1):
                i1 = add_vertex(face[i][0], face[i][1])
                i2 = add_vertex(face[i+1][0], face[i+1][1])
                indices.extend([base_idx, i1, i2])

        positions = np.array(positions, dtype='f4')
        texcoords = np.array(texcoords, dtype='f4')
        indices = np.array(indices, dtype='i4')

        # Crear buffers en GPU
        self.vbo = self.ctx.buffer(positions.tobytes())
        self.uvbo = self.ctx.buffer(texcoords.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '3f', 'in_pos'),
                (self.uvbo, '2f', 'in_uv'),
            ],
            self.ibo
        )

        # Cargar textura desde obj_loader.texture_cv (ya es RGB/RGBA según tu loader)
        if obj_loader.texture_cv is None:
            raise RuntimeError("Objeto no contiene texture_cv. Asegúrate de que ObjectLoader cargue la textura.")
        tex = obj_loader.texture_cv
        # Convertir a bytes RGB(A) en el orden esperado (row-major)
        # Si tiene 4 canales mantén 4, si 3 entonces 3.
        if tex.dtype != np.uint8:
            tex = (tex * 255).astype(np.uint8)
        h, w = tex.shape[:2]
        components = 3 if tex.shape[2] == 3 else 4
        # ModernGL expects RGB(A) order: tu texture_cv ya la guardaste como RGB/RGBA
        self.texture = self.ctx.texture((w, h), components, tex.tobytes())
        self.texture.build_mipmaps()
        self.texture.use(0)

        # Proyección ortográfica desde coordenadas de pantalla (0..w, 0..h)
        # Notar: invertimos Y en la proyección para que (0,0) sea esquina superior izquierda
        self.projection = pyrr.matrix44.create_orthogonal_projection(
            0.0, float(width),
            float(height), 0.0,
            -1000.0, 1000.0,
            dtype='f4'
        )

    def render(self, model_matrix):
        # Renderizar en framebuffer transparente
        fbo = self.ctx.simple_framebuffer((self.width, self.height), components=4)
        fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        # MVP = projection * model_matrix
        mvp = pyrr.matrix44.multiply(self.projection, model_matrix)  # projection * model
        # Asegurar shape f32
        self.prog['mvp'].write(mvp.astype('f4').tobytes())

        # Render
        self.vao.render()
        self.ctx.finish()

        # Leer framebuffer (RGBA u1)
        img = fbo.read(components=4, dtype='u1')
        img = np.frombuffer(img, dtype=np.uint8).reshape(self.height, self.width, 4)
        # Flip vertical
        img = cv2.flip(img, 0)
        return img
