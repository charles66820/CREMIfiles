#ifndef SCENE_H
#define SCENE_H

#include "camera.h"
#include "shape.h"
#include "light.h"
#include "integrator.h"

#include <nanogui/glutil.h>

typedef std::vector<Shape*> ShapeList;
typedef std::vector<Light*> LightList;

class Scene : public Object
{
public :
    Scene(const PropertyList& props);

    /// \return a pointer to the scene's camera
    Camera* camera() { return m_camera; }
    const Camera* camera() const { return m_camera; }
    void setCamera(Camera* camera) { m_camera = camera; }

    /// \return a pointer to the scene's integrator
    Integrator* integrator() { return m_integrator; }
    const Integrator* integrator() const { return m_integrator; }

    /// \return a reference to an array containing all the shapes
    const ShapeList& shapeList() const { return m_shapeList; }

    /// \return a reference to an array containing all the lights
    const LightList& lightList() const { return m_lightList; }

    /// \return the background color
    Color3f backgroundColor() const { return m_backgroundColor; }

    /// Search the nearest intersection between the ray and the shape list
    void intersect(const Ray& ray, Hit& hit) const;

    /// Register a child object (e.g. a material) with the shape
    virtual void addChild(Object *child);

    /// \brief Return the type of object provided by this instance
    EClassType getClassType() const { return EScene; }

    /// Clear the scene and release the memory
    void clear();

    /// Return a string summary of the scene (for debugging purposes)
    std::string toString() const;

private:

    Integrator* m_integrator = nullptr;

    Camera* m_camera = nullptr;

    ShapeList m_shapeList;

    LightList m_lightList;

    Color3f m_backgroundColor;
};

#endif // SCENE_H
