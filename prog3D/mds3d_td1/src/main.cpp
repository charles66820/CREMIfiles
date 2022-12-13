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

    /// TODO: here
    ///  1. iterate over the image pixels
    uint width = camera->vpWidth();
    uint height = camera->vpHeight();
    Point3f origin = camera->position() + camF - camX - camY; // axe optique
    for (uint i = 0; i < width; i++)
        for (uint j = 0; j < height; j++) {
            Point3f position; // P_{i,j}
            position = 2 * (i / width - 0.5) * camX + 2 * (j / height - 0.5) * camY;
            // p = camera->position() + Vector3f(x, y, 0);
            Vector3f direction;
            // d = (p - origin);
            direction = camF + 2 * (i / width - 0.5) * camX + 2 * (j / height - 0.5) * camY;
            // printf("oui\n");
            // std::cout << d;
            direction.normalize(); // \frac{d}{||d||_2}
            // std::cout << d;

            // if (x == 0 && y == 0) {
            //     Vector3f vpX = camX;
            //     vpX.normalize();
            //     vpX *= camX.norm() / width * 2;
            //     Vector3f vpY = camY;
            //     vpY.normalize();
            //     vpY *= camY.norm() / height * 2;
            // }


            /// 2. generate a primary ray
            Ray ray = Ray(origin, direction);

            ///  3. call the integartor to compute the color along this ray
            Color3f color;// = getColor(ray);

            ///  4. write this color in the result image
            // result->put(position, color);
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
