#ifndef VIEWER_H
#define VIEWER_H

#include "camera.h"
#include "mesh.h"
#include "opengl.h"
#include "shader.h"
#include "trackball.h"

#include <iostream>

class Viewer
{
public:
    //! Constructor
    Viewer();
    virtual ~Viewer();

    // gl stuff
    void init(int w, int h);
    void drawScene();
    void drawScene2D();
    void drawSceneTP4();
    void drawSceneTP4Cam();
    void drawSceneTP4SolarSystem();
    void updateAndDrawScene();
    void reshape(int w, int h);
    void loadShaders();

    // events
    void mousePressed(GLFWwindow* window, int button, int action);
    void mouseMoved(int x, int y);
    void mouseScroll(double x, double y);
    void keyPressed(int key, int action, int mods);
    void charPressed(int key);

private:
    int _winWidth, _winHeight;

    Camera _cam;
    Shader _shader;
    Shader _shaderFront;
    Shader _shaderSide;
    Shader _shaderLine;
    Shader _shaderLineMT;
    Shader _shaderCam;
    Shader _shaderSL;
    Mesh _mesh;
    float _scale;
    float _zoom;
    float _rot;
    Eigen::Vector2f _translation;
    bool _enableWires;

    // Mouse parameters for the trackball
    enum TrackMode { TM_NO_TRACK = 0, TM_ROTATE_AROUND, TM_ZOOM, TM_LOCAL_ROTATE, TM_FLY_Z, TM_FLY_PAN };
    TrackMode _trackingMode = TM_NO_TRACK;
    Trackball _trackball;
    Eigen::Vector2i _lastMousePos;
};

#endif
