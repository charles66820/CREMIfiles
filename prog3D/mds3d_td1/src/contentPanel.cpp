#include "contentPanel.h"

#include "parser.h"

#include <filesystem/resolver.h>
#include <nanogui/button.h>
#include <nanogui/checkbox.h>
#include <nanogui/layout.h>
#include <nanogui/slider.h>
#include <thread>

ContentPanel::ContentPanel(nanogui::Screen* screen)
    : nanogui::Widget(screen)
{
    using namespace nanogui;
    m_screen = screen;
}

ContentPanel::~ContentPanel()
{
}

int prevMouseX = 0;
int prevMouseY = 0;
int sensitivity = 100;

// bool ContentPanel::mouseButtonEvent(const Vector2i& p, int button, bool down, int modifiers)
// {
//     std::cout << "mouseButtonEvent" << std::endl;
//     std::cout << p << std::endl;
// }
bool ContentPanel::mouseMotionEvent(const Vector2i& p, const Vector2i& rel, int button, int modifiers)
{
    std::cout << "mouseMotionEvent" << std::endl;
    std::cout << p << std::endl;
}
// bool ContentPanel::mouseDragEvent(const Vector2i& p, const Vector2i& rel, int button, int modifiers)
// {
//     std::cout << "mouseDragEvent" << std::endl;
//     std::cout << p << std::endl;
// }
// bool ContentPanel::mouseEnterEvent(const Vector2i& p, bool enter)
// {
//     std::cout << "mouseEnterEvent" << std::endl;
//     std::cout << p << std::endl;
// }

// bool ContentPanel::mouseMotionEvent(const Vector2i& p, const Vector2i& rel, int button, int modifiers)
// {
    // std::cout << "toto" << std::endl;
    // std::cout << p << std::endl;
    // auto startRender = [&]() {
    //     if (m_scene && m_renderingDone) {
    //         m_renderingDone = false;
    //         /* Determine the filename of the output bitmap */
    //         std::string outputName = m_curentFilename;
    //         size_t lastdot = outputName.find_last_of(".");
    //         if (lastdot != std::string::npos)
    //             outputName.erase(lastdot, std::string::npos);
    //         outputName += ".exr";

    //         /* Allocate memory for the entire output image */
    //         if (m_resultImage)
    //             delete m_resultImage;
    //         m_resultImage = new ImageBlock(m_scene->camera()->outputSize());

    //         std::thread render_thread(render, m_scene, m_resultImage, outputName, &m_renderingDone);
    //         render_thread.detach();
    //         m_button1->setEnabled(true);
    //         m_button2->setEnabled(true);
    //     }
    // };

    // if (m_scene && m_renderingDone) {
    //     Eigen::Quaternionf currentOrientation = m_scene->camera()->orientation();

    //     auto mouseX = p.x();
    //     auto mouseY = p.y();

    //     auto right = m_scene->camera()->right();
    //     auto up = m_scene->camera()->up();

    //     int dx = mouseX - prevMouseX;
    //     int dy = mouseY - prevMouseY;

    //     float pitch = dy * sensitivity;
    //     float yaw = dx * sensitivity;

    //     Eigen::Quaternionf qPitch = Eigen::Quaternion<float>(Eigen::AngleAxis<float>(pitch, right));
    //     Eigen::Quaternionf qYaw = Eigen::Quaternion<float>(Eigen::AngleAxis<float>(yaw, up));
    //     Eigen::Quaternionf newOrientation = qYaw * qPitch * currentOrientation;

    //     float t = 0.5f; // Interpolation scalar (0-1)
    //     Eigen::Quaternionf interpolatedOrientation = currentOrientation.slerp(t, newOrientation);

    //     m_scene->camera()->setOrientation(interpolatedOrientation);

    //     prevMouseX = mouseX;
    //     prevMouseY = mouseY;
    //     startRender();
    // }
    // return mouseMotionEvent(p, rel, button, modifiers);
// }
