#include "viewer.h"

#include <filesystem/resolver.h>

void render(Scene* scene, ImageBlock* result, std::string outputName, bool* done)
{
    if(!scene)
        return;

    clock_t t = clock();

    const Camera *camera = scene->camera();
    const Integrator* integrator = scene->integrator();
    integrator->preprocess(scene);

    float tanfovy2 = tan(camera->fovY()*0.5);
    Vector3f camX = camera->right() * tanfovy2 * camera->nearDist() * float(camera->vpWidth())/float(camera->vpHeight());
    Vector3f camY = -camera->up() * tanfovy2 * camera->nearDist();
    Vector3f camF = camera->direction() * camera->nearDist();

    // camF : distance focal ?
    // camX, camY : axes sur l'image

    /// FIXME: here
    ///  1. iterate over the image pixels
    uint width = camera->vpWidth();
    uint height = camera->vpHeight();
    Point3f origin = camera->position() + camF - camX - camY; // axe optique

    std::cout << "width :" << width << std::endl;
    std::cout << "height :" << height << std::endl;
    std::cout << "camera pos :" << camera->position().toString() << std::endl;
    std::cout << "camF :" << camF.toString() << std::endl;
    std::cout << "camX :" << camX.toString() << std::endl;
    std::cout << "camY :" << camY.toString() << std::endl;
    Vector3f C = camX + camY - camF;
    std::cout << "C (camX + camY - camF) :" << C.toString() << std::endl;
    std::cout << "origin :" << origin.toString() << std::endl;

    for (uint i = 0; i < width; i++)
        for (uint j = 0; j < height; j++) {
            Vector3f p;
            p = (camF * camF.norm()) + 2 * (i / (width - 0.5)) * camX +
                2 * (j / (height - 0.5)) * camY;

            Vector3f direction = p - C;
            direction.normalize();

            /// 2. generate a primary ray
            Ray ray = Ray(origin, direction);

            ///  3. call the integartor to compute the color along this ray
            Color3f color = integrator->Li(scene, ray);

            ///  4. write this color in the result image
            result->put(Vector2f(i, j), color);
        }

    t = clock() - t;
    std::cout << "Raytracing time : " << float(t)/CLOCKS_PER_SEC << "s"<<std::endl;

    *done = true;
}

int main(int argc, char *argv[])
{
    getFileResolver()->prepend(DATA_DIR);

    try {
        nanogui::init();
        Viewer *screen = new Viewer();;

        if (argc == 2) {
            /* load file from the command line */
            filesystem::path path(argv[1]);

            if(path.extension() == "scn") { // load scene file
                screen->loadScene(argv[1]);
            }else if(path.extension() == "exr") { // load OpenEXR image
                screen->loadImage(argv[1]);
            }
        }

        /* Enter the application main loop */
        nanogui::mainloop();

        delete screen;
        nanogui::shutdown();
    } catch (const std::exception &e) {
        cerr << "Fatal error: " << e.what() << endl;
        return -1;
    }
    return 0;
}
