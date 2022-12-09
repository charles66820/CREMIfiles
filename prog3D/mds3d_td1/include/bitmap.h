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

#if !defined(__NORI_BITMAP_H)
#define __NORI_BITMAP_H

#include <color.h>

/**
 * \brief Stores a RGB high dynamic-range bitmap
 *
 * The bitmap class provides I/O support using the OpenEXR file format
 */
class Bitmap : public Eigen::Array<Color3f, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> {
public:
    typedef Eigen::Array<Color3f, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Base;

    /**
     * \brief Allocate a new bitmap of the specified size
     *
     * The contents will initially be undefined, so make sure
     * to call \ref clear() if necessary
     */
    Bitmap(const Eigen::Vector2i &size = Eigen::Vector2i(0, 0))
        : Base(size.y(), size.x()) { }

    /// Load an EXR or PNG file with the specified filename
    Bitmap(const filesystem::path &filename);

    /// Save the bitmap with the specified filename according to its extension
    void save(const filesystem::path &filename, bool flip=false);

protected:

    /// Load the bitmap as an EXR file with the specified filename
    void loadEXR(const std::string &filename);

    /// Load the bitmap as an PNG file with the specified filename
    void loadPNG(const std::string &filename);

    /// Save the bitmap as an EXR file with the specified filename
    void saveEXR(const std::string &filename);

    /// Save the bitmap as an PNG file with the specified filename
    void savePNG(const std::string &filename, bool flip);
};

#endif /* __NORI_BITMAP_H */
