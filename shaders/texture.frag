#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D tex0;

void main() {
    // DEBUG: Mostrar patrón de ajedrez para verificar que se renderiza algo
    vec2 grid = floor(v_uv * 10.0);
    if (mod(grid.x + grid.y, 2.0) < 1.0) {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0); // Rojo
    } else {
        fragColor = vec4(0.0, 0.0, 1.0, 1.0); // Azul
    }
}
