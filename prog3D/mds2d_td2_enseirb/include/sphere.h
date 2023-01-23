
#ifndef SPHERE_H
#define SPHERE_H

#include "shape.h"

#include <Eigen/Core>

/** Represents a sphere
 */
class Sphere : public Shape
{
public:

    Sphere(float radius);
    Sphere(const PropertyList &propList);
    virtual ~Sphere();

    virtual bool intersect(const Ray& ray, Hit& hit) const;

    /// Return a human-readable summary
    std::string toString() const {
        return tfm::format(
            "Sphere[\n"
            "  center = %s,\n"
            "  radius = %f\n"
            "  material = %s,\n]"
            "]", ::toString(m_center),
                 m_radius,
                 m_material ? indent(m_material->toString()) : std::string("null"));
    }

protected:
    Point3f m_center;
    float   m_radius;
};

#endif
