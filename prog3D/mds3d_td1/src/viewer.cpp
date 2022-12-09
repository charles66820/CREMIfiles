#include "viewer.h"

#include "parser.h"

#include <filesystem/resolver.h>
#include <nanogui/slider.h>
#include <nanogui/checkbox.h>
#include <nanogui/button.h>
#include <nanogui/layout.h>
#include <thread>

extern void render(Scene* scene, ImageBlock* result, std::string outputName, bool* done);

Viewer::Viewer() :
    nanogui::Screen(Vector2i(512,512+50), "Raytracer")
{
    m_renderingDone = true;

    /* Add some UI elements to adjust the exposure value */
    using namespace nanogui;
    m_panel = new Widget(this);
    m_panel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 10, 10));
    m_checkbox = new CheckBox(m_panel,"srgb");
    m_checkbox->setChecked(true);
    m_checkbox->setCallback(
        [&](bool value) {
            m_srgb = value ? 1 : 0;
        }
    );
    m_slider = new Slider(m_panel);
    m_slider->setValue(0.5f);
    m_slider->setFixedWidth(150);
    m_slider->setCallback(
        [&](float value) {
            m_scale = std::pow(2.f, (value - 0.5f) * 20);
        }
    );

    m_button1 = new Button(m_panel);
    m_button1->setCaption("PNG");
    m_button1->setEnabled(false);
    m_button1->setCallback(
        [&](void) {
            m_fbo.init(mFBSize,1);
            m_fbo.bind();
            drawContents();
            m_fbo.release();

            Bitmap img(Eigen::Vector2i(mFBSize.x(), mFBSize.y()-50));
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            m_fbo.bind();
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            glReadPixels(0, 50, mFBSize.x(), mFBSize.y() + 50, GL_RGB, GL_FLOAT, img.data());
            m_fbo.release();

            std::string outputName = m_curentFilename;
            size_t lastdot = outputName.find_last_of(".");
            if (lastdot != std::string::npos)
                outputName.erase(lastdot, std::string::npos);
            outputName += ".png";

            img.save(outputName, true);

        }
    );

    m_button2 = new Button(m_panel);
    m_button2->setCaption("EXR");
    m_button2->setEnabled(false);
    m_button2->setCallback(
        [&](void) {
            std::string outputName = m_curentFilename;
            size_t lastdot = outputName.find_last_of(".");
            if (lastdot != std::string::npos)
                outputName.erase(lastdot, std::string::npos);
            outputName += ".exr";

            Bitmap* img2 = m_resultImage->toBitmap();
            img2->save(outputName);
            delete img2;
        }
    );

    m_panel->setSize(Eigen::Vector2i(512,50));
    performLayout(mNVGContext);
    m_panel->setPosition(Eigen::Vector2i((512 - m_panel->size().x()) / 2, 512));

    initializeGL();

    drawAll();
    setVisible(true);
}

Viewer::~Viewer()
{
    m_tonemapProgram.free();
}

void Viewer::initializeGL()
{
    m_tonemapProgram.init(
            "Tonemapper",
            /* Vertex shader */
            "#version 330\n"
            "in vec2 position;\n"
            "out vec2 uv;\n"
            "void main() {\n"
            "    gl_Position = vec4(position.x*2-1, position.y*2-1, 0.0, 1.0);\n"
            "    uv = vec2(position.x, 1-position.y);\n"
            "}",
            /* Fragment shader */
            "#version 330\n"
            "uniform sampler2D source;\n"
            "uniform float scale;\n"
            "uniform int srgb;\n"
            "in vec2 uv;\n"
            "out vec4 out_color;\n"
            "float toSRGB(float value) {\n"
            "    if (value < 0.0031308)\n"
            "        return 12.92 * value;\n"
            "    return 1.055 * pow(value, 0.41666) - 0.055;\n"
            "}\n"
            "void main() {\n"
            "    vec4 color = texture(source, uv);\n"
            "    color *= scale / color.w;\n"
            "    if(srgb==1)\n"
            "         out_color = vec4(toSRGB(color.r), toSRGB(color.g), toSRGB(color.b), 1);\n"
            "    else\n"
            "         out_color = vec4(color.rgb, 1);\n"
            "}"
        );

    MatrixXu indices(3, 2); /* Draw 2 triangles */
    indices.col(0) << 0, 1, 2;
    indices.col(1) << 2, 3, 0;

    MatrixXf positions(2, 4);
    positions.col(0) << 0, 0;
    positions.col(1) << 1, 0;
    positions.col(2) << 1, 1;
    positions.col(3) << 0, 1;

    m_tonemapProgram.bind();
    m_tonemapProgram.uploadIndices(indices);
    m_tonemapProgram.uploadAttrib("position", positions);

    /* Allocate texture memory for the rendered image */
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

void Viewer::drawContents()
{
    if(m_resultImage) // raytracing in progress
    {
        /* Reload the partially rendered image onto the GPU */
        m_resultImage->lock();
        int borderSize = m_resultImage->getBorderSize();
        const Vector2i &size = m_resultImage->getSize();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_texture);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, m_resultImage->cols());
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x(), size.y(),
                     0, GL_RGBA, GL_FLOAT, (uint8_t *) m_resultImage->data() +
                     (borderSize * m_resultImage->cols() + borderSize) * sizeof(Color4f));
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        m_resultImage->unlock();

        glViewport(0, 50*mPixelRatio, mPixelRatio*size[0], mPixelRatio*size[1]);
        m_tonemapProgram.bind();
        m_tonemapProgram.setUniform("scale", m_scale);
        m_tonemapProgram.setUniform("srgb", m_srgb);
        m_tonemapProgram.setUniform("source", 0);
        m_tonemapProgram.drawIndexed(GL_TRIANGLES, 0, 2);
    } 
    glViewport(0, 0, mFBSize[0], mFBSize[1]);
}

void Viewer::framebufferSizeChanged()
{
    m_panel->setSize(Eigen::Vector2i(mFBSize[0],50));
    performLayout(mNVGContext);
    m_panel->setPosition(Eigen::Vector2i((mFBSize[0] - m_panel->size().x()) / 2, mFBSize[1]-50));
}

void Viewer::loadScene(const std::string& filename)
{
    if(filename.size()>0) {
        filesystem::path path(filename);

        if (path.extension() != "scn")
            return;
        m_renderingDone = true;
        if(m_resultImage) {
            delete m_resultImage;
            m_resultImage = nullptr;
        }

        getFileResolver()->prepend(path.parent_path());

        ::Object* root = loadFromXML(filename);
        if (root->getClassType() == ::Object::EScene){
            if (m_scene)
                delete m_scene;
            m_scene = static_cast<Scene*>(root);
            std::cout << m_scene->toString() << std::endl;
            m_curentFilename = filename;

            // Update GUI
            setSize(m_scene->camera()->outputSize() + Eigen::Vector2i(0,50));
            glfwSetWindowSize(glfwWindow(),m_scene->camera()->vpWidth(),m_scene->camera()->vpHeight()+50);
            m_panel->setSize(Eigen::Vector2i(m_scene->camera()->vpWidth(),50));
            performLayout(mNVGContext);
            m_panel->setPosition(Eigen::Vector2i((m_scene->camera()->vpWidth() - m_panel->size().x()) / 2,
                                                  m_scene->camera()->vpHeight()));
        }
        drawAll();
    }
}

void Viewer::loadImage(const std::string &filename)
{
    m_curentFilename = filename;
    Bitmap bitmap(filename);
    m_resultImage = new ImageBlock(Eigen::Vector2i(bitmap.cols(), bitmap.rows()));
    m_resultImage->fromBitmap(bitmap);
    m_renderingDone = false;
    // Update GUI
    setSize(Eigen::Vector2i(m_resultImage->cols(), m_resultImage->rows()+50));
    glfwSetWindowSize(glfwWindow(),m_resultImage->cols(), m_resultImage->rows()+50);
    m_panel->setSize(Eigen::Vector2i(mFBSize[0],50));
    m_button1->setEnabled(true);
    m_button2->setEnabled(true);
    performLayout(mNVGContext);
    m_panel->setPosition(Eigen::Vector2i((mFBSize[0] - m_panel->size().x()) / 2, m_resultImage->rows()));
}

bool Viewer::keyboardEvent(int key, int scancode, int action, int modifiers)
{
    if(action == GLFW_PRESS) {
        switch(key)
        {
        case GLFW_KEY_L:
        {
            std::string filename = nanogui::file_dialog( { {"scn", "Scene file"}, {"exr", "Image file"} }, false);
            filesystem::path path(filename);
            if (path.extension() == "scn")
                loadScene(filename);
            else if(path.extension() == "exr")
                loadImage(filename);
            return true;
        }
        case GLFW_KEY_R:
        {
            if(m_scene && m_renderingDone) {
                m_renderingDone = false;
                /* Determine the filename of the output bitmap */
                std::string outputName = m_curentFilename;
                size_t lastdot = outputName.find_last_of(".");
                if (lastdot != std::string::npos)
                    outputName.erase(lastdot, std::string::npos);
                outputName += ".exr";

                /* Allocate memory for the entire output image */
                if(m_resultImage)
                    delete m_resultImage;
                m_resultImage = new ImageBlock(m_scene->camera()->outputSize());

                std::thread render_thread(render,m_scene,m_resultImage,outputName,&m_renderingDone);
                render_thread.detach();
                m_button1->setEnabled(true);
                m_button2->setEnabled(true);
            }
            return true;
        }
        case GLFW_KEY_ESCAPE:
            exit(0);
        default:
            break;
        }
    }
    return Screen::keyboardEvent(key,scancode,action,modifiers);
}

bool Viewer::dropEvent(const std::vector<std::string> &filenames)
{
    // only tries to load the first file
    filesystem::path path(filenames.front());
    if (path.extension() == "scn")
        loadScene(filenames.front());
    else if(path.extension() == "exr")
        loadImage(filenames.front());

    drawAll();
    return true;
}
