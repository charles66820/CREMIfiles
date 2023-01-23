#include "light.h"

class PointLight : public Light
{
public:
    PointLight(const PropertyList& propList)
        : Light(propList.getColor("intensity", Color3f(1.f)))
    {
        m_position = propList.getPoint("position", Point3f::UnitX());
    }

    Vector3f direction(const Point3f& x, float* dist = 0) const
    {
        auto xl = m_position; // x_l

        *dist = (xl - x).norm();

        // (x_l-x)/||x_l-x||  //^2
        return (xl-x)/(xl-x).norm();
    }

    Color3f intensity(const Point3f& x) const
    {
        auto xl = m_position; // x_l
        auto Il = m_intensity; // I_l

        // I_l/||x-x_l||^2
        return m_intensity / (x - xl).squaredNorm();
    }

    std::string toString() const
    {
        return tfm::format("PointLight[\n"
                           "  intensity = %s\n"
                           "  position = %s\n"
                           "]",
                           m_intensity.toString(),
                           ::toString(m_position));
    }

protected:
    Point3f m_position;
};

REGISTER_CLASS(PointLight, "pointLight")
