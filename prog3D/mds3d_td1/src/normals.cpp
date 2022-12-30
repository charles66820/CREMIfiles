#include "integrator.h"
#include "scene.h"

class NormalsIntegrator : public Integrator
{
public:
    NormalsIntegrator(const PropertyList& props)
    {
        /* No parameters this time */
    }

    Color3f Li(const Scene* scene, const Ray& ray) const
    {
        Hit hit;
        scene->intersect(ray, hit);

        if (hit.foundIntersection()) {
            auto shape = hit.shape();
            auto normal = hit.normal();

            // [1, -1] => [1, 0]
            float r = (normal.x() + 1) / 2;
            float g = (normal.y() + 1) / 2;
            float b = (normal.z() + 1) / 2;

            return Color3f(r, g, b);
        }

        return scene->backgroundColor();
    }

    std::string toString() const
    {
        return "NormalsIntegrator[]";
    }
};

REGISTER_CLASS(NormalsIntegrator, "normals")
