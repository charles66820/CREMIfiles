#include "integrator.h"
#include "scene.h"
#include "material.h"

class Whitted : public Integrator
{
public:
    Whitted(const PropertyList &props) {
        m_maxRecursion = props.getInteger("maxRecursion",4);
    }

    Color3f Li(const Scene *scene, const Ray &ray) const {

        Color3f radiance = Color3f::Zero();

        // stopping criteria is recursion level > maxRecursion
        if (ray.recursionLevel >= m_maxRecursion) return Color3f();

        /* Find the surface that is visible in the requested direction */
             //TODO : FIX ME ENSEIRB STUDENTS

        Hit hit;
        scene->intersect(ray, hit);
        if (hit.foundIntersection())
        {
            auto shape = hit.shape();
            auto material = shape->material();
            Point3f intersectPoint = ray.at(hit.t());
            // TODO : FIX ME ENSEIRB STUDENTS
            //  DIRECT LIGHTING IMPLEMENTATION goes here

            auto n = hit.normal();
            auto v = ray.direction;  // ray direction is the view direction
            Vector2f uv = hit.uv();  // material->texture();

            // reflexions
            // TODO : FIX ME ENSEIRB STUDENTS
            Color3f reflectivity = material->reflectivity();

            // refraction
            // TODO : FIX ME ENSEIRB STUDENTS
            // The total Reflection (the final color to the view point)
            Color3f R = Color3f();
            // ∑_i(ρ⋅max(⟨l_i⋅n⟩,0)I_i)
            for (auto light : scene->lightList()) {
              float lightDistance = 0.f;
              Vector3f l = light->direction(intersectPoint,
                                            &lightDistance);  // direction
              Color3f I = light->intensity(intersectPoint);   // intensity

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
            R += Li(scene, bouncyRay) * reflectivity;

            return R;

        } else if(ray.recursionLevel == 0)
            return scene->backgroundColor();

        return radiance;
    }

    std::string toString() const {
        return tfm::format("Whitted[\n"
                           "  max recursion = %f\n"
                           " ]\n",
                           m_maxRecursion);
    }
private:
    int m_maxRecursion;
};

REGISTER_CLASS(Whitted, "whitted")
