#version 330 core

in vec3 var_color;
out vec4 out_color;

void main(void) {
    // out_color = vec4(1,0,0,1);
    // out_color.xyz = var_color.xyz;
    // out_color.w = 1;
    out_color.rgb = var_color.xyz;
    out_color.a = 1;
}
