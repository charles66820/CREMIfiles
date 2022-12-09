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

#include "object.h"


void Object::addChild(Object *) {
    throw RTException(
        "Object::addChild() is not implemented for objects of type '%s'!",
        classTypeName(getClassType()));
}

void Object::activate() { /* Do nothing */ }
void Object::setParent(Object *) { /* Do nothing */ }

std::map<std::string, ObjectFactory::Constructor> *ObjectFactory::m_constructors = nullptr;

void ObjectFactory::registerClass(const std::string &name, const Constructor &constr) {
    if (!m_constructors)
        m_constructors = new std::map<std::string, ObjectFactory::Constructor>();
    (*m_constructors)[name] = constr;
}
