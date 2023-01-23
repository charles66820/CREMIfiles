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
            Point3f intersectPoint = ray.at(hit.t());

            auto n = hit.normal();
            auto v = ray.direction; // ray direction is the view direction
            Vector2f uv = hit.uv(); // material->texture();

            // The total Reflection (the final color to the view point)
            Color3f R = Color3f::Zero();
            // ∑_i(ρ⋅max(⟨l_i⋅n⟩,0)I_i)
            for (auto light : scene->lightList()) {
                float lightDistance = 0.f;
                Vector3f l = light->direction(intersectPoint, &lightDistance); // direction
                Color3f I = light->intensity(intersectPoint);                  // intensity

                // drop shadow
                Point3f xPr = intersectPoint + 1e-6f * l;
                Ray rayCast = Ray(xPr, l);
                Hit lightHit;
                scene->intersect(rayCast, lightHit);
                if (lightHit.foundIntersection() && lightHit.t() < lightDistance)
                    continue;

                Color3f rho = material->brdf(v, l, n, uv);
                R += rho * std::max(l.dot(n), 0.f) * I;
            }

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
