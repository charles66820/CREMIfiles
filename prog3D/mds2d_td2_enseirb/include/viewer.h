#ifndef VIEWER_H
#define VIEWER_H

#include "scene.h"
#include "block.h"

#include <nanogui/screen.h>
#include <nanogui/glutil.h>

class Viewer : public nanogui::Screen
{
    // A scene contains a list of objects, a list of light sources, a camera and an integrator.
    Scene* m_scene = nullptr;

    ImageBlock* m_resultImage = nullptr;
    std::string m_curentFilename;
    bool m_renderingDone;

    // GUI
    nanogui::GLFramebuffer m_fbo;
    nanogui::GLShader m_tonemapProgram;
    nanogui::Slider *m_slider = nullptr;
    nanogui::CheckBox *m_checkbox = nullptr;
    nanogui::Button* m_button1 = nullptr;
    nanogui::Button* m_button2 = nullptr;
    nanogui::Widget *m_panel = nullptr;
    uint32_t m_texture = 0;
    float m_scale = 1.f;
    int m_srgb = 1;

  protected:
    void initializeGL();

    /** This method is automatically called everytime the opengl windows is resized. */
    virtual void framebufferSizeChanged();

    /** This method is automatically called everytime the opengl windows has to be refreshed. */
    virtual void drawContents();

    /** This method is automatically called everytime a key is pressed */
    virtual bool keyboardEvent(int key, int scancode, int action, int modifiers);

    /** This method is called when files are dropped on the window */
    virtual bool dropEvent(const std::vector<std::string> &filenames);

  public: 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    /** This method load a 3D scene from a file */
    void loadScene(const std::string &filename);

    /** This method load an OpenEXR image from a file */
    void loadImage(const std::string &filename);

    // default constructor
    Viewer();
    ~Viewer();
};

#endif // VIEWER_H

