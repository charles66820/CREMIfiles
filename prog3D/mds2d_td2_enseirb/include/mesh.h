// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>

#ifndef _Mesh_h_
#define _Mesh_h_

#include "common.h"
#include "shape.h"
#include "bvh.h"

#include <vector>
#include <string>
#include <nanogui/glutil.h>

/** \class Mesh
  * A class to represent a 3D triangular mesh
  */
class Mesh : public Shape
{
public:
    static long int ms_itersection_count;

    /** Represents a vertex of the mesh */
    struct Vertex
    {
      Vertex()
        : position(Point3f::Zero()), normal(Normal3f::Zero()), texcoord(Vector2f::Zero()) {}

      Vertex(const Point3f& pos)
        : position(pos), normal(Vector3f::Zero()), texcoord(Vector2f::Zero()) {}

      Vertex(const Point3f& pos, const Normal3f& n, const Vector2f& uv)
        : position(pos), normal(n), texcoord(uv) {}

      Point3f position;
      Normal3f normal;
      Vector2f texcoord;
    };
  
    Mesh() {}

    Mesh(const PropertyList &propList);

    /** Destructor */
    virtual ~Mesh();

    void loadFromFile(const std::string& filename);

    /** Loads a triangular mesh in the OFF format */
    void loadOFF(const std::string& filename);

    /** Loads a triangular mesh in the OBJ format */
    void loadOBJ(const std::string& filename);
    
    void loadRawData(float* positions, int nbVertices, int* indices, int nbTriangles); 
    
    virtual bool intersect(const Ray& ray, Hit& hit) const;

    /** compute the intersection between a ray and a given triangular face */
    bool intersectFace(const Ray& ray, Hit& hit, int faceId) const;

    void makeUnitary();
    void computeAABB();
    void buildBVH();

    /// \returns  the number of faces
    int nbFaces() const { return m_faces.size(); }

    /// \returns a const references to the \a vertexId -th vertex of the \a faceId -th face. vertexId must be between 0 and 2 !!
    const Vertex& vertexOfFace(int faceId, int vertexId) const { return m_vertices[m_faces[faceId](vertexId)]; }

    virtual const Eigen::AlignedBox3f& AABB() const { return m_AABB; }

    /// Return a human-readable summary of this instance
    std::string toString() const;

    BVH* bvh() { return m_BVH; }
    
protected:

    /** Represent a triangular face via its 3 vertex indices. */
    typedef Eigen::Vector3i FaceIndex;

    /** Represents a sequential list of vertices */
    typedef std::vector<Vertex> VertexArray;

    /** Represents a sequential list of triangles */
    typedef std::vector<FaceIndex> FaceIndexArray;

    /** The list of vertices */
    VertexArray m_vertices;
    /** The list of face indices */
    FaceIndexArray m_faces;

    /** The bounding box of the mesh */
    Eigen::AlignedBox3f m_AABB;

    BVH* m_BVH;
};

#endif
