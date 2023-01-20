#version 330 core

in vec3 vtx_position;
uniform float zoom;
uniform vec2 translation;

void main()
{
  gl_Position = vec4(vtx_position.x + translation.x, vtx_position.y + translation.y, -vtx_position.z, zoom);
}