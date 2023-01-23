#include "sphere.h"
#include <iostream>

Sphere::Sphere(float radius)
    : m_radius(radius)
{
}

Sphere::Sphere(const PropertyList &propList)
{
    m_radius = propList.getFloat("radius",1.f);
    m_center = propList.getPoint("center",Point3f(0,0,0));
}

Sphere::~Sphere()
{
}

bool Sphere::intersect(const Ray& ray, Hit& hit) const
{
    // compute ray-sphere intersection

    auto c = m_center;
    auto r = m_radius;
    auto o = ray.origin;
    auto d = ray.direction;

    // d*d * t^2 + (2 * d.dot(o - c)) * t + (||o - c||^2 - r^2) = 0
    auto a = 1;  // d.dot(d) == 1
    auto b = 2 * d.dot(o - c);
    auto c_ = (o - c).squaredNorm() - (r * r);

    auto delta = (b * b) - 4 * a * c_;

    if (delta >= 0) {
      float t;
      if (delta > 0) {
        float t1 = (-(b - sqrt(delta))) / (2 * a);
        float t2 = (-(b + sqrt(delta))) / (2 * a);
        t = t1 <= t2 ? t1 : t2;
      } else if (delta == 0) {
        t = -(b / (2 * a));
      }

      if (t > 0.f) {
        Point3f intersectPoint = ray.at(t);
        float x = intersectPoint.x();
        float y = intersectPoint.y();
        float z = intersectPoint.z();
        hit.setShape(this);
        hit.setT(t);
        hit.setNormal((intersectPoint - c).normalized());
        hit.setUV(atan2(z, sqrt(pow(x, 2) + pow(y, 2))), atan2(y, x));
        return true;
      }
    }

    // throw RTException("Sphere::intersect not implemented yet.");

    return false;
}

REGISTER_CLASS(Sphere, "sphere")