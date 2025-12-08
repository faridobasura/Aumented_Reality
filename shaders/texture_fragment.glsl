#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D textureSampler;
uniform bool useTexture;
uniform float brightness;  // Control de brillo desde Python (0.0 - 1.0)

void main()
{
    if (useTexture) {
        // Obtener color de la textura
        vec4 texColor = texture(textureSampler, TexCoord);
        
        // ===== ILUMINACIÓN CONFIGURABLE =====
        
        // 1. Luz ambiental base (siempre presente)
        vec3 ambientLight = vec3(0.3, 0.3, 0.3);
        
        // 2. Luz direccional (simula luz desde el frente)
        vec3 directionalLight = vec3(0.5, 0.5, 0.5);
        
        // 3. Combinar luces
        vec3 totalLight = ambientLight + directionalLight;
        
        // 4. Aplicar iluminación a la textura
        vec3 litColor = texColor.rgb * totalLight;
        
        // 5. Agregar brillo adicional ajustable (desde Python)
        litColor += vec3(brightness);
        
        // 6. Clamp para evitar valores fuera de rango
        litColor = clamp(litColor, 0.0, 1.0);
        
        FragColor = vec4(litColor, texColor.a);
    } else {
        // Modo wireframe/sin textura
        FragColor = vec4(0.7, 0.7, 0.7, 1.0);
    }
}
