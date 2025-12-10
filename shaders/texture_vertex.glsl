#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;
layout(location = 3) in vec3 aTangent;

out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;
out mat3 TBN;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Posición del fragmento en espacio mundial
    FragPos = vec3(model * vec4(aPos, 1.0));
    
    // Normal en espacio mundial
    Normal = normalize(mat3(transpose(inverse(model))) * aNormal);
    
    // Tangent en espacio mundial
    vec3 Tangent = normalize(mat3(model) * aTangent);
    
    // Bitangent se calcula desde Normal y Tangent
    vec3 Bitangent = cross(Normal, Tangent);
    
    // Crear matriz TBN (Tangent-Bitangent-Normal)
    TBN = mat3(Tangent, Bitangent, Normal);
    
    // Pasar coordenadas de textura
    TexCoord = aTexCoord;
    
    // Posición en espacio de pantalla
    gl_Position = projection * view * vec4(FragPos, 1.0);
}

