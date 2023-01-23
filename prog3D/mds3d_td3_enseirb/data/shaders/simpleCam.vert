#version 330 core

in vec3 vtx_position;
in vec4 vtx_color;
out vec4 var_color;
uniform mat4 obj_mat; // matrice objet
uniform mat4 camera_view_mat; // matrice de vue
uniform mat4 percpective_mat; // matrice de perspective

void main()
{
  gl_Position = obj_mat * camera_view_mat * vec4(vtx_position, 1);

  var_color = vtx_color;
}
