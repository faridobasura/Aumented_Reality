#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D tex0;

void main() {
    vec4 color = texture(tex0, v_uv);
    if (color.a < 0.1)
        discard;
    fragColor = color;
}
