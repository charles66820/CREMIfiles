#include "viewer.h"
#include "camera.h"

using namespace Eigen;

#define toRadian(a) a * (M_PI / 180)

Viewer::Viewer()
    : _winWidth(0),
      _winHeight(0),
      _scale(1),
      _zoom(0),
      _rot(0.),
      _translation(Vector2f(0.f, 0.f)),
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

    Vector3f camPos = Vector3f(-.5f, .3f, 0.25f); //+ Vector3f(_translation.x(), _translation.y(), _zoom)
    Vector3f camTarget = Vector3f(0.f, .3f, 0.f);
    Vector3f camUpAnchor = Vector3f(0.f, 1.f, 0.f);
    _cam.lookAt(camPos, camTarget, camUpAnchor);

    _trackball.setCamera(&_cam);
}

void Viewer::reshape(int w, int h)
{
    _winWidth = w;
    _winHeight = h;
    _cam.setViewport(w, h);
}

/*!
   callback to draw graphic primitives
 */
void Viewer::drawScene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glClearColor(0.5f, 0.5f, 0.5f, 1);

    // Main view
    glViewport(0, 0, _winWidth / 2, _winHeight);
    _shaderFront.activate();
    glUniform1f(_shaderFront.getUniformLocation("zoom"), _zoom + 0.5f);
    glUniform2fv(_shaderFront.getUniformLocation("translation"), 1, _translation.data());
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDepthFunc(GL_LESS);
    _mesh.draw(_shaderFront);
    _shaderFront.deactivate();

    if (_enableWires) {
        _shaderLine.activate();
        glUniform1f(_shaderLine.getUniformLocation("zoom"), _zoom + 0.5f);
        glUniform2fv(_shaderLine.getUniformLocation("translation"), 1, _translation.data());
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDepthFunc(GL_LESS);
        glEnable(GL_LINE_SMOOTH);
        _mesh.draw(_shader);
        glDisable(GL_LINE_SMOOTH);
        _shaderLine.deactivate();
    }

    // Side view
    glViewport(_winWidth / 2, 0, _winWidth / 2, _winHeight);
    _shaderSide.activate();
    glUniform1f(_shaderSide.getUniformLocation("zoom"), _zoom + 0.5f);
    glUniform2fv(_shaderSide.getUniformLocation("translation"), 1, _translation.data());
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDepthFunc(GL_LESS);
    _mesh.draw(_shaderSide);
    _shaderSide.deactivate();
}

void Viewer::drawScene2D()
{
    glViewport(0, 0, _winWidth, _winHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.7f, 0.7f, 0.7f, 1);

    _shader.activate();
    Matrix4f M;
    //contrôle au clavier
    // M << _scale, 0, 0, _translation.x(), //
    //     0, _scale, 0, _translation.y(),  //
    //     0, 0, _scale, _zoom,             //
    //     0, 0, 0, 1;

    // chair left
    // M << 0.5, 0, 0, -0.5, //
    //     0, 0.5, 0, -1,    //
    //     0, 0, 0.5, 0,     //
    //     0, 0, 0, 1;
    Affine3f A = Scaling(0.5f) * Translation3f(Vector3f(-1, -2, 0));
    glUniformMatrix4fv(_shader.getUniformLocation("obj_mat"), 1, GL_FALSE, A.matrix().data());
    _mesh.draw(_shader);

    // chair right
    // M << -0.5, 0, 0, 0.5, //
    //     0, 0.5, 0, -1,    //
    //     0, 0, -0.5, 0,    //
    //     0, 0, 0, 1;
    // A = Translation3f(Vector3f(0.5, -1, 0)) * AngleAxisf(toRadian(180), Vector3f::UnitY()) * Scaling(.5f);
    A = Scaling(0.5f) * AngleAxisf(toRadian(180), Vector3f::UnitY()) * Translation3f(Vector3f(-1, -2, 0));
    glUniformMatrix4fv(_shader.getUniformLocation("obj_mat"), 1, GL_FALSE, A.matrix().data());
    _mesh.draw(_shader);

    // chair center
    // M << 0.707107, -0.707107, 0, .5, //
    //     0.707107, 0.707107, 0, 0,    //
    //     0, 0, 1, 0,                  //
    //     0, 0, 0, 1;
    A = Translation3f(0.f, 0.5f, 0.f) * AngleAxisf(toRadian(_rot), Vector3f::UnitZ()) * Translation3f(0.f, -0.5f, 0.f);
    glUniformMatrix4fv(_shader.getUniformLocation("obj_mat"), 1, GL_FALSE, A.matrix().data());
    _mesh.draw(_shader);
    _shader.deactivate();
}

void Viewer::drawSceneTP4()
{
    glViewport(0, 0, _winWidth, _winHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glClearColor(0.5f, 0.5f, 0.5f, 1);

    _shader.activate();

    Affine3f A = Translation3f(_translation.x(), _translation.y(), _zoom) * Scaling(_scale) *
                 AngleAxisf(toRadian(_rot), Vector3f(-0.5f, 1.5f, 0.f)) * Translation3f(0.f, 0.f, 0.f);
    // Affine3f A = Translation3f(0.f, 0.f, 0.f) * Scaling(_scale) * AngleAxisf(toRadian(_rot), Vector3f::UnitY()) *
    //              Translation3f(0.f, 0.f, 0.f);
    glUniformMatrix4fv(_shader.getUniformLocation("obj_mat"), 1, GL_FALSE, A.matrix().data());

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDepthFunc(GL_LESS);
    _mesh.draw(_shader);
    _shader.deactivate();

    if (_enableWires) {
        _shaderLineMT.activate();
        Affine3f A = Translation3f(_translation.x(), _translation.y(), _zoom) * Scaling(_scale) *
                     AngleAxisf(toRadian(_rot), Vector3f(-0.5f, 1.5f, 0.f)) * Translation3f(0.f, 0.f, 0.f);
        glUniformMatrix4fv(_shaderLineMT.getUniformLocation("obj_mat"), 1, GL_FALSE, A.matrix().data());

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDepthFunc(GL_LESS);
        glEnable(GL_LINE_SMOOTH);
        _mesh.draw(_shader);
        glDisable(GL_LINE_SMOOTH);
        _shaderLineMT.deactivate();
    }
}

void Viewer::drawSceneTP4Cam()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.5f, 0.5f, 0.5f, 1);
    glViewport(0, 0, _winWidth, _winHeight);

    _shaderCam.activate();

    Affine3f A = Scaling(_scale) * AngleAxisf(toRadian(_rot), Vector3f::UnitY()) * Translation3f(0.f, 0.f, 0.f);
    glUniformMatrix4fv(_shaderCam.getUniformLocation("obj_mat"), 1, GL_FALSE, A.matrix().data());

    Matrix4f vm = _cam.viewMatrix();
    glUniformMatrix4fv(_shaderCam.getUniformLocation("camera_view_mat"), 1, GL_FALSE, vm.data());

    auto proj = _cam.projectionMatrix();
    glUniformMatrix4fv(_shaderCam.getUniformLocation("percpective_mat"), 1, GL_FALSE, proj.data());

    _mesh.draw(_shaderCam);
    _shaderCam.deactivate();

}

void Viewer::updateAndDrawScene()
{
    // drawScene();
    // drawScene2D();
    // drawSceneTP4();
    drawSceneTP4Cam();
}

void Viewer::loadShaders()
{
    // Here we can load as many shaders as we want, currently we have only one:
    _shader.loadFromFiles(DATA_DIR "/shaders/simple.vert", DATA_DIR "/shaders/simple.frag");
    _shaderFront.loadFromFiles(DATA_DIR "/shaders/simpleFront.vert", DATA_DIR "/shaders/simple.frag");
    _shaderSide.loadFromFiles(DATA_DIR "/shaders/simpleSide.vert", DATA_DIR "/shaders/simple.frag");
    _shaderLine.loadFromFiles(DATA_DIR "/shaders/line.vert", DATA_DIR "/shaders/line.frag");
    _shaderLineMT.loadFromFiles(DATA_DIR "/shaders/lineMT.vert", DATA_DIR "/shaders/line.frag");
    _shaderCam.loadFromFiles(DATA_DIR "/shaders/simpleCam.vert", DATA_DIR "/shaders/simple.frag");
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
        } else if (key == GLFW_KEY_COMMA) {
            _scale += 0.1;
        } else if (key == GLFW_KEY_PERIOD) {
            _scale -= 0.1;
        } else if (key == GLFW_KEY_LEFT_BRACKET) {
            auto a = (_rot + 1);
            _rot = a > 360 ? a - 360 : a;
        } else if (key == GLFW_KEY_RIGHT_BRACKET) {
            auto a = (_rot - 1);
            _rot = a < 0 ? a + 360 : a;
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
