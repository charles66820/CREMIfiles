
#ifndef BVH_H
#define BVH_H

#include <Eigen/Geometry>
#include <vector>
#include <nanogui/glutil.h>
#include "ray.h"
class Mesh;

class BVH
{

    struct Node {
        Eigen::AlignedBox3f box;
        union {
            int first_child_id; // for inner nodes
            int first_face_id;  // for leaves
        };
        unsigned short nb_faces;
        short is_leaf = false;
    };

    typedef std::vector<Node> NodeList;

public:

    void build(const Mesh* pMesh, int targetCellSize, int maxDepth);
    bool intersect(const Ray& ray, Hit& hit) const;

    void draw(nanogui::GLShader* prg, int maxDepth) const;

protected:

    bool intersectNode(int nodeId, const Ray& ray, Hit& hit) const;

    int split(int start, int end, int dim, float split_value);

    void buildNode(int nodeId, int start, int end, int level, int targetCellSize, int maxDepth);

    void drawNode(nanogui::GLShader* prg, int id, int maxDepth, int currentDepth) const;

    const Mesh* m_pMesh;
    NodeList m_nodes;
    std::vector<int> m_faces;
    std::vector<Point3f> m_centroids;
};

#endif
