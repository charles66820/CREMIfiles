#version 330 core

in vec3 vtx_position;
in vec4 vtx_color;
out vec4 var_color;
uniform mat4 obj_mat;

void main()
{
  gl_Position =  obj_mat * vec4(vtx_position, 1);

  // or

  // // Scale
  // gl_Position.xyz = vtx_position * vec3(obj_mat[0][0], obj_mat[1][1], obj_mat[2][2]);
  // gl_Position.w = 0;

  // // Rotations

  // // Translation
  // gl_Position.xyzw += vec4(obj_mat[3][0], obj_mat[3][1], obj_mat[3][2], obj_mat[3][3]);

  var_color = vtx_color;
}
