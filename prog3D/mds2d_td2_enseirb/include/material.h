#ifndef MATERIAL_H
#define MATERIAL_H

#include "bitmap.h"
#include "object.h"

class Material : public Object
{
public:
    Material() : m_texture(nullptr) {}

    virtual Color3f ambientColor() const = 0;
    virtual Color3f diffuseColor(const Vector2f& uv) const = 0;
    virtual Color3f reflectivity() const = 0;

    /// evaluate the BRDF
    virtual Color3f brdf(const Vector3f& viewDir, const Vector3f& lightDir, const Normal3f& normal, const Vector2f& uv = Vector2f::Zero()) const = 0;

protected:
    /// texture
    enum TextureMode { MODULATE, REPLACE };

    float textureScaleU() const { return m_textureScaleU; }
    float textureScaleV() const { return m_textureScaleV; }
    TextureMode textureMode() const { return m_textureMode; }
    const Bitmap* texture() const { return m_texture; }

    void setTexture(Bitmap* texture) { m_texture = texture; }
    void loadTextureFromFile(const std::string& fileName);
    void setTextureScale(float textureScale) { setTextureScaleU(textureScale); setTextureScaleV(textureScale); }
    void setTextureScaleU(float textureScaleU) { if (fabs(textureScaleU) > 1e-3) m_textureScaleU = textureScaleU; }
    void setTextureScaleV(float textureScaleV) { if (fabs(textureScaleV) > 1e-3) m_textureScaleV = textureScaleV; }
    void setTextureMode(TextureMode textureMode) { m_textureMode = textureMode; }

    EClassType getClassType() const { return EMaterial; }

private:
    TextureMode m_textureMode;
    Bitmap* m_texture;
    float m_textureScaleU, m_textureScaleV;
};


class Diffuse : public Material
{
public:
   Diffuse(const Color3f& diffuseColor) : m_diffuseColor(diffuseColor) {}
   Diffuse(const PropertyList &propList);

   Color3f ambientColor() const { return m_diffuseColor; }
   Color3f diffuseColor(const Vector2f& uv) const;
   Color3f reflectivity() const { return Color3f(0); }

   Color3f brdf(const Vector3f& viewDir, const Vector3f& lightDir, const Normal3f& normal, const Vector2f& uv) const { return diffuseColor(uv) * INV_PI; }

   std::string toString() const {
       return tfm::format(
           "Diffuse[\n"
           "  color = %s\n"
           "]", m_diffuseColor.toString());
   }

protected:
    Color3f m_diffuseColor;
};


class Phong : public Diffuse
{
public:
    Phong(const PropertyList &propList);

    Color3f reflectivity() const { return m_reflectivity; }

    Color3f brdf(const Vector3f& viewDir, const Vector3f& lightDir, const Normal3f& normal, const Vector2f& uv) const;

    std::string toString() const;

protected:
    Color3f m_specularColor;
    Color3f m_reflectivity;
    float m_exponent;
};

#endif // MATERIAL_H
