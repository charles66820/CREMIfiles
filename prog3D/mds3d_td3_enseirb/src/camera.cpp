
#include "camera.h"

using namespace Eigen;

Camera::Camera()
{
    mViewMatrix.setIdentity();
    setPerspective(M_PI / 2, 0.1, 10000);
    setViewport(0, 0);
}

void Camera::lookAt(const Vector3f& position, const Vector3f& target, const Vector3f& up)
{
    mTarget = target;
    Vector3f z = (target - position).normalized(); // vecteur opposé du vecteur de visée
    Vector3f y = z.cross(up).normalized(); // Crée un vecteur orthogonal à z et up
    Vector3f x = y.cross(z);

    mViewMatrix << y(0), y(1), y(2), -y.dot(position), //
        x(0), x(1), x(2), -x.dot(position),            //
        -z(0), -z(1), -z(2), z.dot(position),          //
        0, 0, 0, 1;
}

void Camera::setPerspective(float fovY, float near, float far)
{
    m_fovY = fovY;
    m_near = near;
    m_far = far;
}

void Camera::setViewport(int width, int height)
{
    mVpWidth = width;
    mVpHeight = height;
}

void Camera::zoom(float x)
{
    Vector3f t = Affine3f(mViewMatrix) * mTarget;
    mViewMatrix = Affine3f(Translation3f(Vector3f(0, 0, x * t.norm())).inverse()) * mViewMatrix;
}

void Camera::rotateAroundTarget(float angle, Vector3f axis)
{
    Affine3f A = Translation3f(0.f, 0.f, 0.f) * AngleAxisf(angle, axis);
    mViewMatrix = mViewMatrix * A.matrix();
}

Camera::~Camera() {}

const Matrix4f& Camera::viewMatrix() const
{
    return mViewMatrix;
}

Matrix4f Camera::projectionMatrix() const
{
    float aspect = float(mVpWidth) / float(mVpHeight);
    float theta = m_fovY * 0.5;
    float range = m_far - m_near;
    float invtan = 1. / std::tan(theta);

    Matrix4f projMat;
    projMat.setZero();
    projMat(0, 0) = invtan / aspect;
    projMat(1, 1) = invtan;
    projMat(2, 2) = -(m_near + m_far) / range;
    projMat(2, 3) = -2 * m_near * m_far / range;
    projMat(3, 2) = -1;

    return projMat;
}
