// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_CAMERA_H
#define EIGEN_CAMERA_H

#include "common.h"
#include "object.h"

#include <nanogui/glutil.h>
#include <Eigen/Geometry>

/// Represents a 3D frame, i.e. an orthogonal basis with the position of the origin.
class Frame
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Frame(const Point3f& pos = Point3f::Zero(),
          const Eigen::Quaternionf& o = Eigen::Quaternionf())
      : orientation(o), position(pos) {}

    Frame lerp(float alpha, const Frame& other) const;

    Eigen::Matrix4f getMatrix() const;

    Eigen::Quaternionf orientation;
    Point3f position;
};

/// Represents a virtual camera, which is essentially a Frame with a given view frustum and viewport
class Camera : public Object
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Camera();
    Camera(const PropertyList& props);
    
    Camera(const Camera& other);
    
    virtual ~Camera();
    
    Camera& operator=(const Camera& other);
    
    //------------------------------------------------------------
    // Viewport setters and getters
    void setViewport(uint offsetx, uint offsety, uint width, uint height);
    void setViewport(uint width, uint height);
    inline uint vpX() const { return mVpX; }
    inline uint vpY() const { return mVpY; }
    inline uint vpWidth() const { return mVpWidth; }
    inline uint vpHeight() const { return mVpHeight; }
    inline Vector2i outputSize() const { return Vector2i(mVpWidth,mVpHeight); }
    //------------------------------------------------------------

    
    //------------------------------------------------------------
    // View frustum setters and getters
    /// \returns the vertical field of view angle (in radian)
    inline float fovY() const { return mFovY; }
    /// sets the vertical field of view angle (in radian)
    void setFovY(float value);
    /// \returns the distance of the image (near) plane
    float nearDist() const { return mNearDist; }
    /** Setup the perspective projection matrix */
    void setPerspective(float fovY, float near, float far);
    //------------------------------------------------------------
    
    //------------------------------------------------------------
    // Frame setters and getters
    /// sets the position of the camera
    void setPosition(const Point3f& pos);
    /// \returns the position of the camera
    inline const Point3f& position() const { return mFrame.position; }
    /// sets the orientation of the camera
    void setOrientation(const Eigen::Quaternionf& q);
    /// \returns the orientation of the camera
    inline const Eigen::Quaternionf& orientation() const { return mFrame.orientation; }
    void setFrame(const Frame& f);
    const Frame& frame(void) const { return mFrame; }
    /// \returns the view direction, i.e., the -z axis of the frame
    Vector3f direction() const;
    /// \returns the up vertical direction, i.e., the y axis of the frame
    Vector3f up() const;
    /// \returns the right horizontal direction , i.e., the x axis of the frame
    Vector3f right() const;
    //------------------------------------------------------------
    
    
    //------------------------------------------------------------
    // Advanced Frame setters
    /** Setup the camera position and orientation based on its \a position, \a a target point, \a the up vector */
    void lookAt(const Point3f& position, const Point3f& target, const Point3f& up);
    /// \returns the priviligied view target point
    inline const Point3f& target(void) { return mTarget; }
    //------------------------------------------------------------

    /// \returns the affine transformation from camera to global space
    const Eigen::Affine3f& viewMatrix() const;
    /// \returns the projective transformation matrix from camera space to the normalized image space
    const Eigen::Matrix4f& projectionMatrix() const;
    
    /// rotates the camera around the target point using the rotation \a q
    void rotateAroundTarget(const Eigen::Quaternionf& q);
    /// rotates the camera around the own camera position using the rotation \a q
    void localRotate(const Eigen::Quaternionf& q);
    /// moves the camera toward the target
    void zoom(float d);
    /// moves the camera by \a t defined in the local camera space
    void localTranslate(const Point3f& t);
    
    Point3f unProject(const Vector2f& uv, float depth, const Eigen::Matrix4f& invModelview) const;
    /// project a given point from the image space to the global space
    Point3f unProject(const Vector2f& uv, float depth) const;

    void convertClickToLine(const Point2i &p, Point3f& orig, Vector3f& dir) const;

    void draw(nanogui::GLShader* prg);

    uint sampleCount() const { return mSampleCount;}
    void setSampleCount(uint s) { mSampleCount = s; }

    /// \brief Return the type of object provided by this instance
    EClassType getClassType() const { return ECamera; }

    std::string toString() const;
    
protected:
    void updateViewMatrix() const;
    void updateProjectionMatrix() const;

protected:

    uint mVpX, mVpY;
    uint mVpWidth, mVpHeight;
    uint mSampleCount;

    Frame mFrame;

    mutable Eigen::Affine3f mViewMatrix;
    mutable Eigen::Matrix4f mProjectionMatrix;

    mutable bool mViewIsUptodate;
    mutable bool mProjIsUptodate;

    Point3f mTarget;
    
    float mFovY;
    float mNearDist;
    float mFarDist;

    mutable unsigned int mVertexBufferId; ///< the id of the BufferObject storing the vertex attributes
    mutable unsigned int mIndexBufferId;  ///< the id of the BufferObject storing the faces indices
    mutable unsigned int mVertexArrayId;  ///< the id of the VertexArray object
    mutable bool mIsInitialized;

    std::vector<Point3f> mPoints;
};

#endif // EIGEN_CAMERA_H
