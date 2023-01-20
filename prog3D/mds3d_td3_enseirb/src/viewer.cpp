#include "viewer.h"
#include "camera.h"

using namespace Eigen;

Viewer::Viewer()
    : _winWidth(0),
      _winHeight(0),
      _zoom(0.5),
      _enableWires(false)
{
}

Viewer::~Viewer() {}

////////////////////////////////////////////////////////////////////////////////
// GL stuff

// #define FILENAME "chair"
#define FILENAME "lemming"

// initialize OpenGL context
void Viewer::init(int w, int h)
{
    loadShaders();

    if (!_mesh.load(DATA_DIR "/models/" FILENAME ".off"))
        exit(1);
    _mesh.initVBA();

    glEnable(GL_DEPTH_TEST);

    reshape(w, h);
    _trackball.setCamera(&_cam);
}

void Viewer::reshape(int w, int h)
{
    _winWidth = w;
    _winHeight = h;
    _cam.setViewport(w, h);
    glViewport(0, 0, _winWidth, _winHeight);
}

/*!
   callback to draw graphic primitives
 */
void Viewer::drawScene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glClearColor(0.5f, 0.5f, 0.5f, 1);

    _shader.activate();
    glUniform1f(_shader.getUniformLocation("zoom"), _zoom);
    glUniform2fv(_shader.getUniformLocation("translation"), 1, _translation.data());
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDepthFunc(GL_LESS);
    _mesh.draw(_shader);
    _shader.deactivate();

    if (_enableWires) {
        _shaderLine.activate();
        glUniform1f(_shaderLine.getUniformLocation("zoom"), _zoom);
        glUniform2fv(_shaderLine.getUniformLocation("translation"), 1, _translation.data());
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDepthFunc(GL_LESS);
        glEnable(GL_LINE_SMOOTH);
        _mesh.draw(_shader);
        glDisable(GL_LINE_SMOOTH);
        _shaderLine.deactivate();
    }
}

void Viewer::drawScene2D()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _shader.activate();
    Matrix4f M;
    // M << 1, 0, 0, 0, //
    //     0, 1, 0, 0,  //
    //     0, 0, 1, 0,  //
    //     0, 0, 0, 1;  //
    M.setIdentity();
    glUniformMatrix4fv(_shader.getUniformLocation("obj_mat"), 1, GL_FALSE, M.data());
    _mesh.draw(_shader);
    _shader.deactivate();
}

void Viewer::updateAndDrawScene()
{
    drawScene();
    // drawScene2D();
}

void Viewer::loadShaders()
{
    // Here we can load as many shaders as we want, currently we have only one:
    _shader.loadFromFiles(DATA_DIR "/shaders/simple.vert", DATA_DIR "/shaders/simple.frag");
    _shaderLine.loadFromFiles(DATA_DIR "/shaders/line.vert", DATA_DIR "/shaders/line.frag");
    checkError();
}

////////////////////////////////////////////////////////////////////////////////
// Events

/*!
   callback to manage keyboard interactions
   You can change in this function the way the user
   interact with the application.
 */
void Viewer::keyPressed(int key, int action, int /*mods*/)
{
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        loadShaders();
    }

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_UP) {
            _translation[1] += 0.1f;
        } else if (key == GLFW_KEY_DOWN) {
            _translation[1] -= 0.1f;
        } else if (key == GLFW_KEY_LEFT) {
            _translation[0] -= 0.1f;
        } else if (key == GLFW_KEY_RIGHT) {
            _translation[0] += 0.1f;
        } else if (key == GLFW_KEY_PAGE_UP || key == GLFW_KEY_KP_SUBTRACT) {
            _zoom += 0.1;
        } else if (key == GLFW_KEY_PAGE_DOWN || key == GLFW_KEY_KP_ADD) {
            _zoom -= 0.1;
        } else if (key == GLFW_KEY_L) {
            _enableWires = !_enableWires;
        }
    }
}

/*!
   callback to manage mouse : called when user press or release mouse button
   You can change in this function the way the user
   interact with the application.
 */
void Viewer::mousePressed(GLFWwindow* /*window*/, int /*button*/, int action)
{
    if (action == GLFW_PRESS) {
        _trackingMode = TM_ROTATE_AROUND;
        _trackball.start();
        _trackball.track(_lastMousePos);
    } else if (action == GLFW_RELEASE) {
        _trackingMode = TM_NO_TRACK;
    }
}

/*!
   callback to manage mouse : called when user move mouse with button pressed
   You can change in this function the way the user
   interact with the application.
 */
void Viewer::mouseMoved(int x, int y)
{
    if (_trackingMode == TM_ROTATE_AROUND) {
        _trackball.track(Vector2i(x, y));
    }

    _lastMousePos = Vector2i(x, y);
}

void Viewer::mouseScroll(double /*x*/, double y)
{
    _cam.zoom(-0.1 * y);
}

void Viewer::charPressed(int /*key*/) {}
