#include "integrator.h"
#include "scene.h"

class WhittedIntegrator : public Integrator
{
public:
    WhittedIntegrator(const PropertyList& props)
    {
        m_maxRecursion = props.getInteger("maxRecursion", 4);
    }

    Color3f Li(const Scene* scene, const Ray& ray) const
    {
        if (ray.recursionLevel >= m_maxRecursion) return Color3f();

        Hit hit;
        scene->intersect(ray, hit);

        if (hit.foundIntersection()) {
            auto shape = hit.shape();
            auto material = shape->material();
            Point3f intersectPoint = ray.at(hit.t());

            auto n = hit.normal();
            auto v = ray.direction; // ray direction is the view direction
            Vector2f uv = hit.uv();  // material->texture();

            Color3f reflectivity = material->reflectivity();

            // The total Reflection (the final color to the view point)
            Color3f R = Color3f();
            // ∑_i(ρ⋅max(⟨l_i⋅n⟩,0)I_i)
            for (auto light : scene->lightList()) {
                float lightDistance = 0.f;
                Vector3f l = light->direction(intersectPoint, &lightDistance); // direction
                Color3f I = light->intensity(intersectPoint);                  // intensity

                // drop shadow
                Point3f xPr = intersectPoint + 1e-4f * l;
                Ray rayCast = Ray(xPr, l);
                Hit lightHit;
                scene->intersect(rayCast, lightHit);
                if (lightHit.foundIntersection() && lightHit.t() < lightDistance)
                    continue;

                Color3f rho = material->brdf(v, l, n, uv);
                R += rho * std::max(l.dot(n), 0.f) * I;
            }

            // The vector v reflected about the normal n
            Vector3f mirrorD = v - 2 * (n.dot(v)) * n;

            Ray bouncyRay = Ray(intersectPoint, mirrorD);
            bouncyRay.recursionLevel = ray.recursionLevel + 1;
            R += Li(scene, bouncyRay) * reflectivity;// * R;

            return R;
        }

        return scene->backgroundColor();
    }

    std::string toString() const
    {
        return "WhittedIntegrator[]";
    }

protected:
    int m_maxRecursion;
};

REGISTER_CLASS(WhittedIntegrator, "whitted")
