#include "integrator.h"
#include "scene.h"

static const float E_4 = 0.0001f;

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
            Point3f intersectPoint = ray.at(hit.t());

            auto n = hit.normal();
            auto v = ray.direction; // ray direction is the view direction
            Vector2f uv;            // material->texture();

            // The total Reflection (the final color to the view point)
            Color3f R = Color3f();
            // ∑_i(ρ⋅max(⟨l_i⋅n⟩,0)I_i)
            for (auto light : scene->lightList()) {
                float lightDistance = 0.f;
                Vector3f l = light->direction(intersectPoint, &lightDistance); // direction
                Color3f I = light->intensity(intersectPoint);                  // intensity

                Point3f xPr = intersectPoint + E_4 * l;
                Ray rayCast = Ray(xPr, l);
                Hit lightHit;
                scene->intersect(rayCast, lightHit);
                if (lightHit.foundIntersection() && lightHit.t() < lightDistance)
                    continue;

                Color3f rho = material->brdf(v, l, n, uv);
                R += rho * std::max(l.dot(n), 0.f) * I;
            }

            // Color3f reflectivity = material->reflectivity();

            return R;
        }

        return scene->backgroundColor();
    }

    std::string toString() const
    {
        return "DirectIntegrator[]";
    }
};

REGISTER_CLASS(DirectIntegrator, "direct")
