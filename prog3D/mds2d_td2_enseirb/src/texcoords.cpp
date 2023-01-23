#include "integrator.h"
#include "scene.h"

class TexcoordsIntegrator : public Integrator {
public:
    TexcoordsIntegrator(const PropertyList &props) {
        /* No parameters this time */
    }

    Color3f Li(const Scene *scene, const Ray &ray) const {
        Hit hit;
        scene->intersect(ray, hit);

        if (hit.foundIntersection()) {
          auto shape = hit.shape();
          auto material = shape->material();
          return material->diffuseColor(hit.uv());
        }

        return scene->backgroundColor();
    }

    std::string toString() const {
        return "TexcoordsIntegrator[]";
    }
};

REGISTER_CLASS(TexcoordsIntegrator, "texcoords")
