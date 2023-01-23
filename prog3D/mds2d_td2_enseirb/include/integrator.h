/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#if !defined(__NORI_INTEGRATOR_H)
#define __NORI_INTEGRATOR_H

#include "object.h"

class Scene;
class Ray;

/**
 * \brief Abstract integrator (i.e. a rendering technique)
 *
 * The different rendering techniques are collectively referred to as
 * integrators, since they perform integration over a high-dimensional
 * space. Each integrator represents a specific approach for solving
 * the light transport equation---usually favored in certain scenarios, but
 * at the same time affected by its own set of intrinsic limitations.
 */

class Integrator : public Object {
public:
    /// Release all memory
    virtual ~Integrator() { }

    /// Perform an (optional) preprocess step
    virtual void preprocess(const Scene *scene) const { }

    /**
     * \brief Sample the incident radiance along a ray
     *
     * \param scene
     *    A pointer to the underlying scene
     * \param ray
     *    The ray in question
     * \return
     *    An estimate of the radiance in this direction
     */
    virtual Color3f Li(const Scene *scene, const Ray &ray) const = 0;

    /**
     * \brief Return the type of object provided by this instance
     * */
    EClassType getClassType() const { return EIntegrator; }
};

#endif /* __NORI_INTEGRATOR_H */
