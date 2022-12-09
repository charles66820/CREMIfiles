
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

#include "camera.h"

#include <iostream>
#include <Eigen/LU>

Frame Frame::lerp(float alpha, const Frame& other) const
{
    return Frame((1.f-alpha)*position + alpha * other.position,
                 orientation.slerp(alpha,other.orientation));
}


Eigen::Matrix4f Frame::getMatrix() const
{
    Eigen::Matrix3f rotation = orientation.toRotationMatrix();
    Eigen::Translation3f translation(position);
    Eigen::Affine3f transformation = translation * rotation;
    return transformation.matrix();
}

Camera::Camera()
    : mViewIsUptodate(false), mProjIsUptodate(false)
{
    mViewMatrix.setIdentity();
    
    mFovY = M_PI/3.;
    mNearDist = 0.1;
    mFarDist = 50000.;
    
    mVpX = 0;
    mVpY = 0;

    lookAt(Vector3f::Constant(100.), Vector3f::Zero(), Vector3f::UnitZ());

    mSampleCount = 1;
    mIsInitialized = false;
}

Camera::Camera(const PropertyList &propList)
    : mViewIsUptodate(false), mProjIsUptodate(false)
{
    mViewMatrix.setIdentity();
    mVpX = 0;
    mVpY = 0;

    /* Width and height in pixels.*/
    mVpWidth = propList.getInteger("width", 512);
    mVpHeight = propList.getInteger("height", 512);

    /* Specifies a camera-to-world transformation. Default: none */
    Transform mCameraToWorld = propList.getTransform("toWorld", Transform());

    mFrame.position = mCameraToWorld.getMatrix().topRightCorner<3,1>();
    mFrame.orientation = Eigen::Quaternionf(mCameraToWorld.getMatrix().topLeftCorner<3,3>());

    /* Horizontal field of view in degrees */
    mFovY = degToRad(propList.getFloat("fieldOfView", 30.0f));

    /* Near and far clipping planes in world-space units */
    mNearDist = propList.getFloat("nearClip", 1e-4f);
    mFarDist = propList.getFloat("farClip", 1e4f);

    /* Number of samples per pixel */
    mSampleCount = propList.getInteger("samples", 1);

    updateViewMatrix();
    updateProjectionMatrix();

    mIsInitialized = false;
}

Camera& Camera::operator=(const Camera& other)
{
    mViewIsUptodate = false;
    mProjIsUptodate = false;
    
    mVpX = other.mVpX;
    mVpY = other.mVpY;
    mVpWidth = other.mVpWidth;
    mVpHeight = other.mVpHeight;

    mTarget = other.mTarget;
    mFovY = other.mFovY;
    mNearDist = other.mNearDist;
    mFarDist = other.mFarDist;
    
    mViewMatrix = other.mViewMatrix;
    mProjectionMatrix = other.mProjectionMatrix;

    mFrame = other.mFrame;
    mSampleCount = other.mSampleCount;

    mIsInitialized = false;

    return *this;
}

Camera::Camera(const Camera& other)
{
    *this = other;
}

Camera::~Camera()
{
}

void Camera::setPerspective(float fovY, float near, float far)
{
    mFovY = fovY;
    mNearDist = near;
    mFarDist = far;
    mProjIsUptodate = false;
}

void Camera::setViewport(uint offsetx, uint offsety, uint width, uint height)
{
    mVpX = offsetx;
    mVpY = offsety;
    mVpWidth = width;
    mVpHeight = height;

    mProjIsUptodate = false;
}

void Camera::setViewport(uint width, uint height)
{
    mVpWidth = width;
    mVpHeight = height;

    mProjIsUptodate = false;
}

void Camera::setFovY(float value)
{
    mFovY = value;
    mProjIsUptodate = false;
}

Vector3f Camera::direction() const
{
    updateViewMatrix();
    return -mViewMatrix.linear().row(2);
}

Vector3f Camera::up() const
{
    updateViewMatrix();
    return mViewMatrix.linear().row(1);
}

Vector3f Camera::right() const
{
    updateViewMatrix();
    return mViewMatrix.linear().row(0);
}

void Camera::lookAt(const Point3f& position, const Point3f& target, const Point3f& up)
{
    mTarget = target;
    mFrame.position = position;
    Eigen::Matrix3f R;
    R.col(2) = (position-target).normalized();
    R.col(0) = up.cross(R.col(2)).normalized();
    R.col(1) = R.col(2).cross(R.col(0));
    setOrientation(Eigen::Quaternionf(R));
    mViewIsUptodate = false;
}

void Camera::setPosition(const Point3f &p)
{
    mFrame.position = p;
    mViewIsUptodate = false;
}

void Camera::setOrientation(const Eigen::Quaternionf& q)
{
    mFrame.orientation = q;
    mViewIsUptodate = false;
}

void Camera::setFrame(const Frame& f)
{
    mFrame = f;
    mViewIsUptodate = false;
}

void Camera::rotateAroundTarget(const Eigen::Quaternionf& q)
{
    Eigen::Matrix4f mrot, mt, mtm;

    // update the transform matrix
    updateViewMatrix();
    Vector3f t = mViewMatrix * mTarget;

    mViewMatrix = Eigen::Translation3f(t)
            * q
            * Eigen::Translation3f(-t)
            * mViewMatrix;

    Eigen::Quaternionf qa(mViewMatrix.linear());
    qa = qa.conjugate();
    setOrientation(qa);
    setPosition(- (qa * mViewMatrix.translation()) );

    mViewIsUptodate = true;
}

void Camera::localRotate(const Eigen::Quaternionf& q)
{
    float dist = (position() - mTarget).norm();
    setOrientation(orientation() * q);
    mTarget = position() + dist * direction();
    mViewIsUptodate = false;
}

void Camera::zoom(float d)
{
    float dist = (position() - mTarget).norm();
    if(dist > d)
    {
        setPosition(position() + direction() * d);
        mViewIsUptodate = false;
    }
}

void Camera::localTranslate(const Point3f &t)
{
    Vector3f trans = orientation() * t;
    setPosition( position() + trans );
    mTarget += trans;

    mViewIsUptodate = false;
}

void Camera::updateViewMatrix() const
{
    if(!mViewIsUptodate)
    {
        Eigen::Quaternionf q = orientation().conjugate();
        mViewMatrix.linear() = q.toRotationMatrix();
        mViewMatrix.translation() = - (mViewMatrix.linear() * position());

        mViewIsUptodate = true;
    }
}

const Eigen::Affine3f& Camera::viewMatrix() const
{
    updateViewMatrix();
    return mViewMatrix;
}

void Camera::updateProjectionMatrix() const
{
    if(!mProjIsUptodate)
    {
        mProjectionMatrix.setIdentity();
        float aspect = float(mVpWidth)/float(mVpHeight);
        float theta = mFovY*0.5;
        float range = mFarDist - mNearDist;
        float invtan = 1./tan(theta);

        mProjectionMatrix(0,0) = invtan / aspect;
        mProjectionMatrix(1,1) = invtan;
        mProjectionMatrix(2,2) = -(mNearDist + mFarDist) / range;
        mProjectionMatrix(3,2) = -1;
        mProjectionMatrix(2,3) = -2 * mNearDist * mFarDist / range;
        mProjectionMatrix(3,3) = 0;

        mProjIsUptodate = true;
    }
}

const Eigen::Matrix4f& Camera::projectionMatrix() const
{
    updateProjectionMatrix();
    return mProjectionMatrix;
}


Point3f Camera::unProject(const Vector2f& uv, float depth) const
{
    Eigen::Matrix4f inv = mViewMatrix.inverse().matrix();
    return unProject(uv, depth, inv);
}

Point3f Camera::unProject(const Vector2f& uv, float depth, const Eigen::Matrix4f& invModelview) const
{
    updateViewMatrix();
    updateProjectionMatrix();

    Vector3f a(2.*uv.x()/float(mVpWidth)-1., 2.*uv.y()/float(mVpHeight)-1., 1.);
    a.x() *= depth/mProjectionMatrix(0,0);
    a.y() *= depth/mProjectionMatrix(1,1);
    a.z() = -depth;

    Vector3f b = a.x() * right() + a.y() * up() - a.z() * direction() + position();

    return Point3f(b.x(), b.y(), b.z());
}

void Camera::draw(nanogui::GLShader* prg)
{
    if(!mIsInitialized)
    {
        mIsInitialized = true;
        mPoints.clear();

        // grille
        float ym = tan(mFovY*0.5);
        float xm = ((float)mVpWidth)*(ym*1.0/mVpHeight);
        float zm = 0.75f;
        for(uint x=1; x<mVpWidth; ++x){
            mPoints.push_back(Point3f(xm*(x*2.0/mVpWidth-1.0),ym,-zm));
            mPoints.push_back(Point3f(xm*(x*2.0/mVpWidth-1.0),-ym,-zm));
        }
        for(uint y=1; y<mVpHeight; ++y){
            mPoints.push_back(Point3f(xm,ym*(y*2.0/mVpHeight-1.0),-zm));
            mPoints.push_back(Point3f(-xm,ym*(y*2.0/mVpHeight-1.0),-zm));
        }

        //pyramide
        mPoints.push_back(Point3f(0.0,0.0,0.0));
        mPoints.push_back(Point3f(xm,ym,-zm));
        mPoints.push_back(Point3f(xm,ym,-zm));
        mPoints.push_back(Point3f(xm,-ym,-zm));
        mPoints.push_back(Point3f(xm,-ym,-zm));
        mPoints.push_back(Point3f(0.0,0.0,0.0));
        mPoints.push_back(Point3f(0.0,0.0,0.0));
        mPoints.push_back(Point3f(-xm,ym,-zm));
        mPoints.push_back(Point3f(-xm,ym,-zm));
        mPoints.push_back(Point3f(-xm,-ym,-zm));
        mPoints.push_back(Point3f(-xm,-ym,-zm));
        mPoints.push_back(Point3f(0.0,0.0,0.0));
        mPoints.push_back(Point3f(0.0,0.0,0.0));
        mPoints.push_back(Point3f(xm,ym,-zm));
        mPoints.push_back(Point3f(xm,ym,-zm));
        mPoints.push_back(Point3f(-xm,ym,-zm));
        mPoints.push_back(Point3f(-xm,ym,-zm));
        mPoints.push_back(Point3f(0.0,0.0,0.0));
        mPoints.push_back(Point3f(0.0,0.0,0.0));
        mPoints.push_back(Point3f(xm,-ym,-zm));
        mPoints.push_back(Point3f(xm,-ym,-zm));
        mPoints.push_back(Point3f(-xm,-ym,-zm));
        mPoints.push_back(Point3f(-xm,-ym,-zm));
        mPoints.push_back(Point3f(0.0,0.0,0.0));

        glGenBuffers(1,&mVertexBufferId);
        glBindBuffer(GL_ARRAY_BUFFER, mVertexBufferId);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Point3f)*mPoints.size(), mPoints[0].data(), GL_STATIC_DRAW);

        glGenVertexArrays(1,&mVertexArrayId);
    }

    // bind the vertex array
    glBindVertexArray(mVertexArrayId);

    glBindBuffer(GL_ARRAY_BUFFER, mVertexBufferId);

    int vertex_loc = prg->attrib("vtx_position");
    glVertexAttribPointer(vertex_loc, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(vertex_loc);

    glUniformMatrix4fv(prg->uniform("mat_obj"), 1, GL_FALSE, mFrame.getMatrix().data());

    glDrawArrays(GL_LINES,0,mPoints.size());

    glDisableVertexAttribArray(vertex_loc);

    // release the vertex array
    glBindVertexArray(0);
}

void Camera::convertClickToLine(const Point2i &p, Point3f &orig, Vector3f& dir) const
{
    orig = position();
    dir = Vector3f( ((2.0 * p[0] / vpWidth()) - 1.0) * tan(fovY()/2.0) * vpWidth() / vpHeight(),
                    ((2.0 * (vpHeight() - p[1]) / vpHeight()) - 1.0) * tan(fovY()/2.0),
                    -1.0 );
    Eigen::Matrix3f rotation = mFrame.orientation.toRotationMatrix();
    dir = rotation * dir + mFrame.position;
    dir = dir - orig;
    dir.normalize();
}

/// Return a human-readable summary
std::string Camera::toString() const {
    std::ostringstream oss;
    oss << mFrame.getMatrix().format(Eigen::IOFormat(4, 0, ", ", ";\n", "", "", "[", "]"));
    return tfm::format(
        "Camera[\n"
        "  frame = %s,\n"
        "  outputSize = %f x %f,\n"
        "  samples = %f,\n"
        "  fov = %f,\n"
        "  clip = [%f, %f],\n"
        "]",
        indent(oss.str(), 10),
        mVpWidth, mVpHeight,
        mSampleCount,
        mFovY,
        mNearDist,
        mFarDist
    );
}

REGISTER_CLASS(Camera, "perspective");
