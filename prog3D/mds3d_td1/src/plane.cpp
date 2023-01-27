#include "plane.h"

Plane::Plane()
{
}

Plane::Plane(const PropertyList& propList)
{
    m_position = propList.getPoint("position", Point3f(0, 0, 0));
    m_normal = propList.getVector("normal", Point3f(0, 0, 1));
}

Plane::~Plane()
{
}

bool Plane::intersect(const Ray& ray, Hit& hit) const
{
    auto n = m_normal;
    auto a = m_position;
    auto o = ray.origin;
    auto d = ray.direction;

    // (o + t * d) * n - D = 0, ||n|| = 1
    // if (n.norm() != 1)
    //     return false;

    float para = d.dot(n);
    // t infini ⟹ le rayon est parallèle et distinct du plan
    if (para > -1e-6 && para < 1e-6)
        return false;

    // Distance du plan au centre (0, 0, 0)
    float D = a.dot(n);
    auto t = (D - o.dot(n)) / para;
    // float t = (a - o).dot(n) / para;

    if (n.dot(d) == 0)
        return false;
    // intersection derrière la caméra
    if (t < 0)
        return false;

    // t == 0 ⟹ le rayon est confondu avec le plan
    // t > 0 ⟹ intersection devant la caméra
    if (t >= 0) {
        Point3f intersectPoint = ray.at(t);
        float x = intersectPoint.x();
        float y = intersectPoint.y();
        hit.setShape(this);
        hit.setT(t);
        hit.setNormal(n);
        // FIXME:[1, -1] => [1, 0]
        hit.setUV(abs(x / 10 * 0.5 + 0.5), abs(y / 10 * 0.5 + 0.5));
        return true;
    }

    return false;
}

REGISTER_CLASS(Plane, "plane")
