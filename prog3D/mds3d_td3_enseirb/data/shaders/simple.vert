#version 330 core

in vec3 vtx_position;
in vec4 vtx_color;
out vec4 var_color;
uniform float zoom;
uniform vec2 translation;

void main()
{
  gl_Position = vec4(vtx_position.x + translation.x, vtx_position.y + translation.y, -vtx_position.z, zoom);
  var_color = vtx_color;
}

// TP2
// in mat4 obj_mat;

// void main()
// {
//   gl_Position = vec4(vtx_position, 1.);
//   var_color = vtx_color;
// }
