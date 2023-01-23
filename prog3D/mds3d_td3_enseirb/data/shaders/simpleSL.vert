#version 330 core

in vec3 vtx_position;
in vec4 vtx_color;
out vec4 var_color;
uniform mat4 obj_mat; // matrice objet
uniform mat4 camera_view_mat; // matrice de vue
uniform mat4 percpective_mat; // matrice de perspective
uniform vec4 vtx_color2;

void main()
{
  gl_Position = percpective_mat * camera_view_mat * obj_mat * vec4(vtx_position, 1);

  if (vtx_color2 != 0) {
    var_color = vtx_color2;
  } else {
    var_color = vtx_color;
  }
}
