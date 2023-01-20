#version 330 core

in vec4 var_color;
out vec4 out_color;

void main(void) {
    // out_color = vec4(1,0,0,1);
    // out_color.xyzw = var_color.xyzw;
    out_color.rgba = var_color.xyza;
}
