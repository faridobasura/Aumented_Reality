#version 330

in vec3 in_pos;
in vec2 in_uv;

uniform mat4 mvp;

out vec2 v_uv;

void main() {
    // DEBUG: Posición de prueba (cuadrado completo)
    float x = float(gl_VertexID % 2) * 2.0 - 1.0;
    float y = float(gl_VertexID / 2) * 2.0 - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
    
    // UVs de prueba
    v_uv = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
}
