#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D textureSampler;
uniform bool useTexture;

void main()
{
    if (useTexture) {
        FragColor = texture(textureSampler, TexCoord);
    } else {
        // Color gris por defecto si no hay textura
        FragColor = vec4(0.7, 0.7, 0.7, 1.0);
    }
}
