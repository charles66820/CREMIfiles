#include "plane.h"

Plane::Plane()
{
}

Plane::Plane(const PropertyList &propList)
{
    m_position = propList.getPoint("position", Point3f(0, 0, 0));
    m_normal = propList.getVector("normal", Point3f(0, 0, 1));
}

Plane::~Plane()
{
}

bool Plane::intersect(const Ray &ray, Hit &hit) const
{
    auto n = m_normal;
    auto a = m_position;
    auto o = ray.origin;
    auto d = ray.direction;

    // Distance du plan au centre (0, 0, 0)
    auto D = a.dot(n);

    // (o + t * d) * n - D = 0, ||n|| = 1
    if (n.norm() != 1)
        return false;

    auto t = (D - o.dot(n)) / d.dot(n);

    // t infini ⟹ le rayon est parallèle et distinct du plan
    // if (t == infinity) ??
    if (n.dot(d) == 0)
        return false;
    // intersection derrière la caméra
    if (t < 0)
        return false;

    // t == 0 ⟹ le rayon est confondu avec le plan
    // t > 0 ⟹ intersection devant la caméra
    if (t == 0 || t > 0) {
        hit.setShape(this);
        hit.setT(t);
        // hit.setNormal(m_normal);
        return true;
    }

    // Point3f OpDxT = ray.at(t);
    // if (OpDxT.norm() == 0) return false;

    return false;
}

REGISTER_CLASS(Plane, "plane")
