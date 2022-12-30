#include "integrator.h"
#include "scene.h"

class DirectIntegrator : public Integrator
{
public:
    DirectIntegrator(const PropertyList& props)
    {
        /* No parameters this time */
    }

    Color3f Li(const Scene* scene, const Ray& ray) const
    {
        Hit hit;
        scene->intersect(ray, hit);

        if (hit.foundIntersection()) {
            auto shape = hit.shape();
            auto material = shape->material();
            auto normal = hit.normal();
            // auto viewDir = ;
            // auto lightDir = ;
            // auto uv = material->texture();

            // scene->lightList();

            Color3f ambient = material->ambientColor();
            // material->diffuseColor(uv);
            // material->brdf(viewDir, lightDir, normal, uv);
            Color3f reflection = material->reflectivity();

            float r = (abs(normal.x())) / 2;
            float g = (abs(normal.y())) / 2;
            float b = (abs(normal.z())) / 2;
            Color3f absolute = Color3f(r, g, b);

            Color3f color = ambient + reflection + absolute;
            return color;
        }

        return scene->backgroundColor();
    }

    std::string toString() const
    {
        return "DirectIntegrator[]";
    }
};

REGISTER_CLASS(DirectIntegrator, "direct")
