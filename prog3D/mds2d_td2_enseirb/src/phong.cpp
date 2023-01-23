#include "material.h"

Phong::Phong(const PropertyList &propList)
    : Diffuse(propList.getColor("diffuse",Color3f(0.2)))
{
    m_specularColor = propList.getColor("specular",Color3f(0.9));
    m_reflectivity = propList.getColor("reflectivity",Color3f(0.0));
    m_exponent = propList.getFloat("exponent",0.2);

    std::string texturePath = propList.getString("texture","");
    if(texturePath.size()>0){
        filesystem::path filepath = getFileResolver()->resolve(texturePath);
        loadTextureFromFile(filepath.str());
        setTextureScale(propList.getFloat("scale",1));
        setTextureMode(TextureMode(propList.getInteger("mode",0)));
    }
}

Color3f Phong::brdf(const Vector3f& viewDir, const Vector3f& lightDir, const Normal3f& normal, const Vector2f& uv) const
{
    Vector3f v = viewDir;
    Vector3f l = lightDir;
    auto n = normal;

    Color3f md = this->diffuseColor(uv);  // (m_d) material diffuse color
    Color3f ms = m_specularColor;         // (m_s) material specular color

    Color3f rho_d = md;  // (ρ_d) diffuse part

    Vector3f r =
        l - 2 * (n.dot(l)) * n;  // The vector l reflected about the normal n
    auto s = m_exponent;  // The shine of the object (m_exponent because is the
                          // exponent i the equation)

    // ρ_s = m_s(cos(α))^s = m_s(max(r * v, 0))^s
    Color3f rho_s =
        ms * pow(std::max(r.dot(v), 0.f), s);  // (ρ_s) specular part

    Color3f rho = rho_d + rho_s;  // ρ

    return rho;
}

std::string Phong::toString() const {
    return tfm::format(
        "Phong[\n"
        "  diffuse color = %s\n"
        "  specular color = %s\n"
        "  exponent = %f\n"
        "]", m_diffuseColor.toString(),
             m_specularColor.toString(),
             m_exponent);
}

REGISTER_CLASS(Phong, "phong")
