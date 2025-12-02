import numpy as np
import cv2



# PyOpenGL
from OpenGL.GL import (
    glGenFramebuffers, glBindFramebuffer, GL_FRAMEBUFFER,
    glGenTextures, glBindTexture, GL_TEXTURE_2D,
    glTexImage2D, GL_RGB, GL_UNSIGNED_BYTE, glTexParameteri,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
    glFramebufferTexture2D, GL_COLOR_ATTACHMENT0,
    glGenRenderbuffers, glBindRenderbuffer, GL_RENDERBUFFER,
    glRenderbufferStorage, GL_DEPTH24_STENCIL8,
    glFramebufferRenderbuffer, glReadPixels,
    glClearColor, glClear, GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT, glEnable, GL_DEPTH_TEST,
    glViewport, glPolygonMode, GL_FRONT_AND_BACK,
    GL_LINE, GL_LINES,GL_POINT, GL_FILL,
    glPushMatrix, glPopMatrix, glRotatef,
    glGenVertexArrays, glBindVertexArray, glGenBuffers,
    glBindBuffer, GL_ARRAY_BUFFER, glBufferData,
    GL_STATIC_DRAW, glEnableVertexAttribArray,
    glVertexAttribPointer, GL_FLOAT, glDrawArrays,
    glBegin,glEnd, glColor3f, glVertex3fv
)
from OpenGL.GL import GL_TRIANGLES
from OpenGL.raw.GL.VERSION.GL_3_0 import GL_DEPTH_STENCIL_ATTACHMENT


class ModelRenderer:
    def __init__(self, width=512, height=512, obj=None):
        """
        Renderizador OpenGL para modelo 3D.
        - Crea FBO off-screen
        - Crea textura de salida
        - Sube la malla del .OBJ
        """
        self.width = width
        self.height = height
        self.obj_model = obj
        self.rotation = [0, 0, 0]
        self.render_mode = "wireframe"

        self._init_gl_objects()
        self._upload_mesh()

    # --------------------------------------------------------
    #   Inicialización del framebuffer y texturas
    # --------------------------------------------------------
    def _init_gl_objects(self):
        # Framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # Textura RGB
        self.color_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_tex)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            self.width, self.height, 0,
            GL_RGB, GL_UNSIGNED_BYTE, None
        )

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, self.color_tex, 0
        )

        # Depth buffer
        self.depth_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,
                              self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                                  GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL_RENDERBUFFER, self.depth_rbo)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # --------------------------------------------------------
    #   Subir malla del OBJ
    # --------------------------------------------------------
    def _upload_mesh(self):
        if self.obj_model is None:
            return

        verts = expand_faces(self.obj_model.vertices, self.obj_model.faces)
        self.vertex_count = len(verts)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)

        glBindVertexArray(0)

    # --------------------------------------------------------
    #   Configuraciones dinámicas
    # --------------------------------------------------------
    def set_viewport(self, w, h):
        self.width = w
        self.height = h
        self._init_gl_objects()

    def set_rotation(self, rx, ry, rz):
        self.rotation = [rx, ry, rz]

    def set_render_mode(self, mode):
        self.render_mode = mode

    # --------------------------------------------------------
    #   Render principal
    # --------------------------------------------------------
    def render_to_image(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)

        glEnable(GL_DEPTH_TEST)
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Render según modo
        if self.render_mode == "wireframe":
            self.render_wireframe()

        elif self.render_mode == "both":
            self.render()           # sólido
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            self.render_wireframe() # wireframe encima

        else:
            self.render()  # sólido

        # Obtener imagen del FBO
        buffer = glReadPixels(0, 0, self.width, self.height,
                              GL_RGB, GL_UNSIGNED_BYTE)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        img = np.frombuffer(buffer, dtype=np.uint8).reshape(self.height, self.width, 3)
        img = cv2.flip(img, 0)
        return img

    
    def render(self):
        """Render sólido del modelo usando GL_TRIANGLES."""
        glEnable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glPushMatrix()
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)

        glPopMatrix()


    def render_wireframe(self):
        """Renderiza SOLO el wireframe del modelo 3D."""

        if self.obj_model is None:
            return

        faces = self.obj_model.faces
        verts = self.obj_model.vertices

        glEnable(GL_DEPTH_TEST)

        # Modo wireframe
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glColor3f(0.0, 1.0, 0.0)  # Wireframe blanco

        glBegin(GL_LINES)
        for face in faces:
            for i in range(len(face)):
                v1 = verts[face[i]]
                v2 = verts[(i + 1) % len(face)]
                glVertex3fv(v1)
                glVertex3fv(v2)
        glEnd()

        # Volver al modo normal
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


def expand_faces(vertices, faces):
    tri_list = []
    for f in faces:
        if len(f) == 3:  
            tri_list.append(vertices[f[0]])
            tri_list.append(vertices[f[1]])
            tri_list.append(vertices[f[2]])
        elif len(f) == 4:  
            tri_list.append(vertices[f[0]])
            tri_list.append(vertices[f[1]])
            tri_list.append(vertices[f[2]])
            tri_list.append(vertices[f[0]])
            tri_list.append(vertices[f[2]])
            tri_list.append(vertices[f[3]])
    return np.array(tri_list, dtype=np.float32)

def set_render_mode(self, mode):
    """
    mode = "solid", "wireframe" o "both"
    """
    self.render_mode = mode
