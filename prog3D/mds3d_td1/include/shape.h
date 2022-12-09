#ifndef SHAPE_H
#define SHAPE_H

#include "ray.h"
#include "common.h"
#include "material.h"
#include "object.h"

/** represents a shape (geometry and material)
 */
class Shape : public Object
{
public:
    Shape() {}

    Shape(const PropertyList&) {}

    /** Search the nearest intersection between the ray and the shape.
      * It must be implemented in the derived class. */
    virtual bool intersect(const Ray& ray, Hit& hit) const {
        throw RTException("Shape::intersect must be implemented in the derived class"); }

    virtual const Material* material() const { return m_material; }
    virtual void setMaterial(const Material* mat) { m_material = mat; }

    /// Register a child object (e.g. a material) with the shape
    virtual void addChild(Object *child);

    /// \brief Return the type of object provided by this instance
    EClassType getClassType() const { return EShape; }

    virtual std::string toString() const {
        throw RTException("Shape::toString must be implemented in the derived class"); }

protected:
    const Material* m_material = nullptr;
};

#endif
