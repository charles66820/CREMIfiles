#ifndef PLANE_H
#define PLANE_H

#include "shape.h"

/**
 * @brief Infinite plane whose normal is facing +Z
 */
class Plane : public Shape
{
public:
    Plane();
    Plane(const PropertyList &propList);
    virtual ~Plane();

    virtual bool intersect(const Ray& ray, Hit& hit) const;

    /// Return a human-readable summary
    std::string toString() const {
        return tfm::format("Plane[\n"
                           "  point = %s\n"
                           "  direction = %s\n"
                           "  material = %s,\n]",
                           m_position,
                           m_normal,
                           m_material ? indent(m_material->toString()) : std::string("null"));
    }

protected:
    Point3f  m_position;
    Vector3f m_normal;
};

#endif // PLANE_H
