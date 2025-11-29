import numpy as np
import cv2

class TextureRenderer:
    """Renderiza modelos 3D con texturas usando OpenCV"""
    
    def __init__(self, obj_loader):
        self.obj_loader = obj_loader
        self.vertices = np.array(obj_loader.vertices)
        self.faces = obj_loader.faces
        self.tex_coords = np.array(obj_loader.tex_coords) if obj_loader.tex_coords else None
        self.texture = obj_loader.texture_cv  # Textura cargada con OpenCV
        
        print(f"✅ TextureRenderer inicializado")
        print(f"   Textura: {self.texture.shape if self.texture is not None else 'None'}")
        print(f"   Coordenadas UV: {len(self.tex_coords) if self.tex_coords is not None else 0}")
    
    def render_textured_model(self, frame, projected_vertices):
        """Renderiza el modelo con texturas"""
        if self.texture is None or self.tex_coords is None:
            print("❌ No hay textura o coordenadas UV para renderizar")
            return
        
        # Renderizar cada cara con textura
        for i, face in enumerate(self.obj_loader.faces):
            if len(face) < 3:
                continue
                
            # Obtener puntos 2D y coordenadas de textura para esta cara
            screen_points = []
            texture_points = []
            
            valid_face = True
            for vertex_data in face:
                vertex_idx = vertex_data[0]
                tex_idx = vertex_data[1] if len(vertex_data) > 1 else -1
                
                if vertex_idx >= len(projected_vertices) or tex_idx == -1 or tex_idx >= len(self.tex_coords):
                    valid_face = False
                    break
                    
                # Punto en pantalla
                screen_points.append(projected_vertices[vertex_idx])
                
                # Coordenada de textura (convertir UV a coordenadas de píxel)
                u, v = self.tex_coords[tex_idx]
                tex_x = int(u * (self.texture.shape[1] - 1))
                tex_y = int((1 - v) * (self.texture.shape[0] - 1))  # Flip V
                texture_points.append((tex_x, tex_y))
            
            if valid_face and len(screen_points) >= 3:
                self._draw_textured_face(frame, screen_points, texture_points)
    
    def _draw_textured_face(self, frame, screen_points, texture_points):
        """Dibuja una cara texturizada usando affine transformation"""
        try:
            if len(screen_points) < 3:
                return
            
            # Crear máscara para la región
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            screen_array = np.array(screen_points, dtype=np.int32)
            cv2.fillConvexPoly(mask, screen_array, 255)
            
            # Para triángulos usar transformación afín
            if len(screen_points) == 3:
                src_tri = np.array(texture_points[:3], dtype=np.float32)
                dst_tri = np.array(screen_points[:3], dtype=np.float32)
                
                # Calcular transformación afín
                warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
                
                # Aplicar transformación a la textura
                warped_texture = cv2.warpAffine(
                    self.texture, warp_mat, 
                    (frame.shape[1], frame.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
                
                # Mezclar con el frame original usando la máscara
                for c in range(3):  # Para cada canal RGB
                    frame[:,:,c] = np.where(
                        mask == 255, 
                        warped_texture[:,:,c], 
                        frame[:,:,c]
                    )
            
            # Para polígonos con más de 3 vértices, usar relleno simple
            else:
                color = self._get_average_color(texture_points)
                cv2.fillPoly(frame, [screen_array], color)
                
        except Exception as e:
            # Fallback: dibujar polígono de color sólido
            color = (100, 100, 200)  # Azul
            cv2.fillPoly(frame, [np.array(screen_points, dtype=np.int32)], color)
    
    def _get_average_color(self, texture_points):
        """Obtiene el color promedio de una región de textura"""
        try:
            # Crear máscara para la región de textura
            tex_mask = np.zeros(self.texture.shape[:2], dtype=np.uint8)
            tex_array = np.array(texture_points, dtype=np.int32)
            cv2.fillConvexPoly(tex_mask, tex_array, 255)
            
            # Calcular color promedio
            mean_color = cv2.mean(self.texture, mask=tex_mask)
            return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
        except:
            return (150, 150, 150)  #Gris por defecto
    
    def render_solid_color(self, frame, projected_vertices, color=(100, 100, 200)):
        """Renderiza el modelo con un color sólido (más rápido)"""
        for face in self.obj_loader.faces:
            if len(face) >= 3:
                points = []
                for vertex_data in face:
                    vertex_idx = vertex_data[0]
                    if vertex_idx < len(projected_vertices):
                        points.append(projected_vertices[vertex_idx])
                
                if len(points) >= 3:
                    points_array = np.array(points, dtype=np.int32)
                    cv2.fillPoly(frame, [points_array], color)