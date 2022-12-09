#include "light.h"

class PointLight : public Light
{
public:
    PointLight(const PropertyList &propList)
        : Light(propList.getColor("intensity", Color3f(1.f)))
    {
        m_position = propList.getPoint("position", Point3f::UnitX());
    }

    Vector3f direction(const Point3f& x, float* dist = 0) const
    {
        /// TODO
        throw RTException("PointLight::direction not implemented yet.");

        return Vector3f(0.f);
    }

    Color3f intensity(const Point3f& x) const
    {
        /// TODO
        throw RTException("PointLight::intensity not implemented yet.");

        return Color3f(0.f);
    }

    std::string toString() const {
        return tfm::format(
            "PointLight[\n"
            "  intensity = %s\n"
            "  position = %s\n"
            "]", m_intensity.toString(),
                 ::toString(m_position));
    }

protected:
    Point3f m_position;
};

REGISTER_CLASS(PointLight, "pointLight")
