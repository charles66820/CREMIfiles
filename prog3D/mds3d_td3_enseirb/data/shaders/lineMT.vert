#version 330 core

in vec3 vtx_position;
uniform mat4 obj_mat;
void main()
{
  gl_Position =  obj_mat * vec4(vtx_position, 1);
}