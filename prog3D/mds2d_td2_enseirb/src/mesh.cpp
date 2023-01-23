// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>

#include "mesh.h"
#include "bvh.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <filesystem/resolver.h>
#include <tiny_obj_loader.h>

Mesh::Mesh(const PropertyList &propList)
    : m_BVH(nullptr)
{
    std::string filename = propList.getString("filename");
    loadFromFile(filename);
    // buildBVH();
}

void Mesh::loadFromFile(const std::string& filename)
{
    filesystem::path filepath = getFileResolver()->resolve(filename);
    std::ifstream is(filepath.str());
    if (is.fail())
        throw RTException("Unable to open mesh file \"%s\"!", filepath.str());

    const std::string ext = filepath.extension();
    if(ext=="off" || ext=="OFF")
        loadOFF(filepath.str());
    else if(ext=="obj" || ext=="OBJ")
        loadOBJ(filepath.str());
    else
        std::cerr << "Mesh: extension \'" << ext << "\' not supported." << std::endl;
}

void Mesh::loadOFF(const std::string& filename)
{
    std::ifstream in(filename.c_str(),std::ios::in);
    if(!in)
    {
        std::cerr << "File not found " << filename << std::endl;
        return;
    }

    std::string header;
    in >> header;

    // check the header file
    if(header != "OFF")
    {
        std::cerr << "Wrong header = " << header << std::endl;
        return;
    }

    int nofVertices, nofFaces, inull;
    int nb, id0, id1, id2;
    Vector3f v;

    in >> nofVertices >> nofFaces >> inull;

    for(int i=0 ; i<nofVertices ; ++i)
    {
        in >> v.x() >> v.y() >> v.z();
        m_vertices.push_back(Vertex(v));
    }

    for(int i=0 ; i<nofFaces ; ++i)
    {
        in >> nb >> id0 >> id1 >> id2;
        assert(nb==3);
        m_faces.push_back(FaceIndex(id0, id1, id2));
    }

    in.close();

    computeAABB();
}

void Mesh::loadOBJ(const std::string& filename)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err, warn;
    bool success = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());

    if (!success) {
        throw RTException("Mesh::loadObj: error loading file %s: %s", filename, err);
    }

    bool needNormals = false;

    int face_idx = 0;

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            assert(fv == 3);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                Point3f pos;
                pos[0] = attrib.vertices[3*idx.vertex_index+0];
                pos[1] = attrib.vertices[3*idx.vertex_index+1];
                pos[2] = attrib.vertices[3*idx.vertex_index+2];

                Vector3f n;
                if (attrib.normals.size() > 0) {
                    n[0] = attrib.normals[3*idx.normal_index+0];
                    n[1] = attrib.normals[3*idx.normal_index+1];
                    n[2] = attrib.normals[3*idx.normal_index+2];
                } else {
                    needNormals = true;
                }

                Vector2f tc = Vector2f::Zero();
                if (attrib.texcoords.size() > 0) {
                    tc[0] = attrib.texcoords[2*idx.texcoord_index+0];
                    tc[1] = attrib.texcoords[2*idx.texcoord_index+1];
                }
                m_vertices.push_back(Vertex(pos, n, tc));
            }

            m_faces.push_back(FaceIndex(face_idx, face_idx+1, face_idx+2));
            face_idx += 3;
            index_offset += fv;
        }
    }

    computeAABB();
}

void Mesh::loadRawData(float* positions, int nbVertices, int* indices, int nbTriangles)
{
    m_vertices.resize(nbVertices);
    for(int i=0; i<nbVertices; ++i)
        m_vertices[i].position = Point3f::Map(positions+3*i);
    m_faces.resize(nbTriangles);
    for(int i=0; i<nbTriangles; ++i)
        m_faces[i] = Eigen::Vector3i::Map(indices+3*i);

    computeAABB();
}

Mesh::~Mesh()
{
    if(m_BVH)
        delete m_BVH;
}

void Mesh::makeUnitary()
{
    // computes the lowest and highest coordinates of the axis aligned bounding box,
    // which are equal to the lowest and highest coordinates of the vertex positions.
    Eigen::Vector3f lowest, highest;
    lowest.fill(std::numeric_limits<float>::max());   // "fill" sets all the coefficients of the vector to the same value
    highest.fill(-std::numeric_limits<float>::max());

    for(VertexArray::iterator v_iter = m_vertices.begin() ; v_iter!=m_vertices.end() ; ++v_iter)
    {
        // - v_iter is an iterator over the elements of mVertices,
        //   an iterator behaves likes a pointer, it has to be dereferenced (*v_iter, or v_iter->) to access the referenced element.
        // - Here the .aray().min(_) and .array().max(_) operators work per component.
        //
        lowest  = lowest.array().min(v_iter->position.array());
        highest = highest.array().max(v_iter->position.array());
    }

    Point3f center = (lowest+highest)/2.0;
    float m = (highest-lowest).maxCoeff();
    for(VertexArray::iterator v_iter = m_vertices.begin() ; v_iter!=m_vertices.end() ; ++v_iter)
        v_iter->position = (v_iter->position - center) / m;

    computeAABB();
}

void Mesh::computeAABB()
{
    m_AABB.setNull();
    for(VertexArray::iterator v_iter = m_vertices.begin() ; v_iter!=m_vertices.end() ; ++v_iter)
        m_AABB.extend(v_iter->position);
}

void Mesh::buildBVH()
{
    if(m_BVH)
        delete m_BVH;
    m_BVH = new BVH;
    m_BVH->build(this, 10, 100);
}

long int Mesh::ms_itersection_count = 0;

bool Mesh::intersectFace(const Ray& ray, Hit& hit, int faceId) const
{
    ms_itersection_count++;
    return true;
}

bool Mesh::intersect(const Ray& ray, Hit& hit) const
{
    float tMin, tMax;
    Normal3f normal;
    if( (!::intersect(ray, m_AABB, tMin, tMax, normal)) || tMin>hit.t())
        return false;

    hit.setT(tMin);
    hit.setNormal(normal);
    hit.setShape(this);

    return true;
}

std::string Mesh::toString() const {
    return tfm::format(
        "Mesh[\n"
        "  vertexCount = %i,\n"
        "  triangleCount = %i,\n"
        "  material = %s\n"
        "]",
        m_vertices.size(),
        m_faces.size(),
        m_material ? indent(m_material->toString()) : std::string("null")
    );
}

REGISTER_CLASS(Mesh, "mesh")
