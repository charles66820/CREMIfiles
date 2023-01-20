#version 330 core

in vec3 vtx_position;
in vec4 vtx_color;
out vec4 var_color;
uniform float zoom;
uniform vec2 translation;

void main()
{
  gl_Position = vec4(vtx_position.xy, -vtx_position.z, zoom);
  gl_Position.xy += translation;
  var_color = vtx_color;
}
