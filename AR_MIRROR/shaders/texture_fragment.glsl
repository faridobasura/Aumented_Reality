#version 330 core
in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform sampler2D colorTexture;
uniform bool useTexture;
uniform float brightness;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform bool debugMode;

void main()
{
    if (debugMode) {
        // Modo debug - visualizar coordenadas UV
        // Rojo = eje U, Verde = eje V
        FragColor = vec4(TexCoord.x, TexCoord.y, 0.0, 1.0);
        return;
    }
    
    if (useTexture) {
        // Obtener color de la textura
        vec4 baseColor = texture(colorTexture, TexCoord);
        
        // Si la textura es completamente negra (0,0,0), mostrar coordenadas UV para debug
        if (baseColor.r < 0.01 && baseColor.g < 0.01 && baseColor.b < 0.01) {
            FragColor = vec4(TexCoord.x, TexCoord.y, 0.5, 1.0);
            return;
        }
        
        // Iluminación básica suave
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(normalize(Normal), lightDir), 0.0);
        
        // Suavizar cambios de iluminación
        diff = smoothstep(0.0, 1.0, diff);
        vec3 diffuse = vec3(0.4) * diff;
        
        // Luz ambiental fuerte
        vec3 ambient = vec3(0.7);
        
        // Combinar iluminación
        vec3 lighting = ambient + diffuse;
        
        // Aplicar iluminación a la textura
        vec3 litColor = baseColor.rgb * lighting;
        
        // Agregar brillo ajustable
        litColor += vec3(brightness);
        
        // Clamp para evitar valores fuera de rango
        litColor = clamp(litColor, 0.0, 1.0);
        
        FragColor = vec4(litColor, baseColor.a);
    } else {
        FragColor = vec4(0.7, 0.7, 0.7, 1.0);
    }
}
