#version 330 core
in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in mat3 TBN;  // Tangent-Bitangent-Normal matrix

out vec4 FragColor;

uniform sampler2D colorTexture;
uniform sampler2D normalTexture;
uniform bool useTexture;
uniform bool useNormalMap;
uniform float brightness;
uniform vec3 lightPos;      // Posición de la luz
uniform vec3 viewPos;       // Posición de la cámara
uniform float normalStrength; // Intensidad del normal map (0.0 - 1.0)

void main()
{
    if (useTexture) {
        // ===== OBTENER COLORES =====
        vec4 baseColor = texture(colorTexture, TexCoord);
        
        // ===== NORMAL MAPPING =====
        vec3 normal = Normal;
        
        if (useNormalMap) {
            // Obtener el normal del mapa (rango 0-1, convertir a -1 a 1)
            vec3 sampledNormal = texture(normalTexture, TexCoord).rgb;
            sampledNormal = normalize(sampledNormal * 2.0 - 1.0);
            
            // Convertir a espacio mundial usando TBN
            sampledNormal = normalize(TBN * sampledNormal);
            
            // Interpolar entre normal original y normal mapeado
            normal = mix(Normal, sampledNormal, normalStrength);
            normal = normalize(normal);
        }
        
        // ===== CÁLCULOS DE ILUMINACIÓN =====
        
        // Vector de dirección de luz
        vec3 lightDir = normalize(lightPos - FragPos);
        
        // Difuso (Lambert)
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = vec3(0.7) * diff;
        
        // Especular (Blinn-Phong)
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 halfDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);
        vec3 specular = vec3(0.3) * spec;
        
        // Luz ambiental
        vec3 ambient = vec3(0.3);
        
        // Combinar todas las luces
        vec3 lighting = ambient + diffuse + specular;
        
        // Aplicar iluminación al color base
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

