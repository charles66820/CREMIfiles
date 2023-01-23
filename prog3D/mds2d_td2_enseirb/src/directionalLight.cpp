#include "light.h"

class DirectionalLight : public Light
{
public:
    DirectionalLight(const PropertyList &propList)
        : Light(propList.getColor("intensity", Color3f(1.f)))
    {
        m_direction = propList.getVector("direction", Vector3f(1.f,0.f,0.f)).normalized();
    }

    Vector3f direction(const Point3f& /*x*/, float* dist = 0) const
    {
        if(dist)
            *dist = std::numeric_limits<float>::max();
        return -m_direction;
    }

    Color3f intensity(const Point3f& x) const
    {
        return m_intensity;
    }

    std::string toString() const {
        return tfm::format(
            "DirectionalLight[\n"
            "  intensity = %s\n"
            "  direction = %s\n"
            "]", m_intensity.toString(),
                 ::toString(m_direction));
    }

protected:
    Vector3f m_direction;
};

REGISTER_CLASS(DirectionalLight, "directionalLight")
