#version 330 core

in vec3 vtx_position;
in vec3 vtx_color;
out vec3 var_color;

void main()
{
  gl_Position = vec4(vtx_position, 1.);
  var_color = vtx_color;
}
