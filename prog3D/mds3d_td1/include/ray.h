#ifndef RAY
#define RAY

#include "common.h"
#include "vector.h"

class Shape;

class Ray
{
public:
    Ray(const Point3f& o, const Vector3f& d)
        : origin(o), direction(d), recursionLevel(0)
    {}
    Ray() : recursionLevel(0) {}

    Point3f origin;
    Vector3f direction;

    Point3f at(float t) const { return origin + t*direction; }

    int recursionLevel;   ///< recursion level (used as a stoping critera)
};

class Hit
{
public:
    Hit() : m_shape(nullptr), m_t(std::numeric_limits<float>::max()) {}

    bool foundIntersection() const { return m_t < std::numeric_limits<float>::max(); }

    void setT(float t) { m_t = t; }
    float t() const { return m_t; }

    void setShape(const Shape* shape) { m_shape = shape; }
    const Shape* shape() const { return m_shape; }

    void setNormal(const Normal3f& n) { m_normal = n; }
    const Normal3f& normal() const { return m_normal; }

private:
    Normal3f m_normal;
    const Shape* m_shape;
    float m_t;
};

/** Compute the intersection between a ray and an aligned box
  * \returns true if an intersection is found
  * The ranges are returned in tMin,tMax
  */
static inline bool intersect(const Ray& ray, const Eigen::AlignedBox3f& box, float& tMin, float& tMax, Normal3f& normal)
{
    Eigen::Array3f t1, t2;
    t1 = (box.min()-ray.origin).cwiseQuotient(ray.direction);
    t2 = (box.max()-ray.origin).cwiseQuotient(ray.direction);
    Eigen::Array3f::Index maxIdx, minIdx;
    tMin = t1.min(t2).maxCoeff(&maxIdx);
    tMax = t1.max(t2).minCoeff(&minIdx);
    normal = Normal3f::Zero();
    normal[maxIdx] = -1;
    return tMax>0 && tMin<=tMax;
}

#endif
