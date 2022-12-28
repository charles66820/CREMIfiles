#include "integrator.h"
#include "scene.h"

class FlatIntegrator : public Integrator {
public:
    FlatIntegrator(const PropertyList &props) {
        /* No parameters this time */
    }

    Color3f Li(const Scene *scene, const Ray &ray) const {
        /** Find the surface that is visible in the requested direction
                   Return its ambiant color */
        Hit hit;
        scene->intersect(ray, hit);

        if (hit.foundIntersection()) {
            auto shape = hit.shape();
            // hit.normal();
            Color3f color = shape->material()->ambientColor();
            return color;
        }

        return scene->backgroundColor();
    }

    std::string toString() const {
        return "FlatIntegrator[]";
    }
};

REGISTER_CLASS(FlatIntegrator, "flat")
