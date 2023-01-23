#include "shape.h"

void Shape::addChild(Object *obj) {
    switch (obj->getClassType()) {
        case EMaterial:
            if (m_material)
                throw RTException(
                    "Shape: tried to register multiple material instances!");
            m_material = static_cast<Material *>(obj);
            break;


        default:
            throw RTException("Shape::addChild(<%s>) is not supported!",
                                classTypeName(obj->getClassType()));
    }
}

REGISTER_CLASS(Shape, "shape")
