#include "plane.h"

Plane::Plane()
{
}

Plane::Plane(const PropertyList &propList)
{
    m_position = propList.getPoint("position",Point3f(0,0,0));
    m_normal = propList.getVector("normal",Point3f(0,0,1));
}

Plane::~Plane()
{
}

bool Plane::intersect(const Ray& ray, Hit& hit) const
{
    //TODO: here
    float dot = normal.dot(ray.direction);
    if (dot == 0) return false; // check si le ray est parallèle au plan

    // la distance parcouru par le ray
    float t = normal.dot(point - ray.origin) / dot;

    Point3f pointIntersectBetweenPlaneAndRay = ray.at(t);

    if (pointIntersectBetweenPlaneAndRay == NULL) return false;
    hit.shape(this);
    hit.setT(t);
    hit.setNormal(normal);
    return true;
}

REGISTER_CLASS(Plane, "plane")
