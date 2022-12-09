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

#include <bitmap.h>

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#include <lodepng.h>

// Simple tile -> scanline converter. Assumes FLOAT pixel type for all channels.
static void TiledImageToScanlineImage(EXRImage* src, const EXRHeader* header)
{
    size_t data_width  = header->data_window[2] - header->data_window[0] + 1;
    size_t data_height = header->data_window[3] - header->data_window[1] + 1;

    src->images = static_cast<unsigned char**>(malloc(sizeof(float*) * header->num_channels));
    for (size_t c = 0; c < static_cast<size_t>(header->num_channels); c++) {
        assert(header->pixel_types[c] == TINYEXR_PIXELTYPE_FLOAT);
        src->images[c] = static_cast<unsigned char*>(malloc(sizeof(float) * data_width * data_height));
        memset(src->images[c], 0, sizeof(float) * data_width * data_height);
    }

    for (size_t tile_idx = 0; tile_idx < static_cast<size_t>(src->num_tiles); tile_idx++) {

        int sx = src->tiles[tile_idx].offset_x * header->tile_size_x;
        int sy = src->tiles[tile_idx].offset_y * header->tile_size_y;
        int ex = src->tiles[tile_idx].offset_x * header->tile_size_x + src->tiles[tile_idx].width;
        int ey = src->tiles[tile_idx].offset_y * header->tile_size_y + src->tiles[tile_idx].height;

        for (size_t c = 0; c < static_cast<size_t>(header->num_channels); c++) {
            float *dst_image = reinterpret_cast<float*>(src->images[c]);
            const float *src_image = reinterpret_cast<const float*>(src->tiles[tile_idx].images[c]);
            for (size_t y = 0; y < static_cast<size_t>(ey - sy); y++) {
                for (size_t x = 0; x < static_cast<size_t>(ex - sx); x++) {
                    dst_image[(y + sy) * data_width + (x + sx)] = src_image[y * header->tile_size_x + x];
                }
            }
        }
    }
}

Bitmap::Bitmap(const filesystem::path &filename) {
    if(filename.extension() == "exr")
        loadEXR(filename.str());
    else if(filename.extension() == "png")
        loadPNG(filename.str());
    else
        cout << "Unknown file type" << endl;
    return;
}

void Bitmap::loadEXR(const std::string &filename) {
    EXRVersion exr_version;

    int ret = ParseEXRVersionFromFile(&exr_version, filename.c_str());
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Invalid EXR file: %s\n", filename.c_str());
        return;
    }

    if (exr_version.multipart) {
        // must be multipart flag is false.
        fprintf(stderr, "Cannot load multi-part EXR\n");
        return;
    }

    EXRHeader exr_header;
    InitEXRHeader(&exr_header);

    const char *err;
    ret = ParseEXRHeaderFromFile(&exr_header, &exr_version, filename.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Parse EXR err: %s\n", err);
        return;
    }

    // Read HALF channel as FLOAT.
    for (int i = 0; i < exr_header.num_channels; i++) {
        if (exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
            exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        }
    }

    EXRImage exr_image;
    InitEXRImage(&exr_image);

    ret = LoadEXRImageFromFile(&exr_image, &exr_header, filename.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Load EXR err: %s\n", err);
        return;
    }

    resize(exr_image.height, exr_image.width);
    cout << "Reading a " << cols() << "x" << rows() << " OpenEXR file from \"" << filename << "\"" << endl;

    if (exr_header.tiled) {
        TiledImageToScanlineImage(&exr_image, &exr_header);
    }

    int idxR = -1, idxG = -1, idxB = -1;
    for (int c = 0; c < exr_header.num_channels; c++) {
        if (strcmp(exr_header.channels[c].name, "R") == 0) {
            idxR = c;
        } else if (strcmp(exr_header.channels[c].name, "G") == 0) {
            idxG = c;
        } else if (strcmp(exr_header.channels[c].name, "B") == 0) {
            idxB = c;
        }
    }

    float *out_rgb = reinterpret_cast<float *>(data());

    for (int i = 0; i < exr_image.width * exr_image.height; i++) {
        out_rgb[3 * i + 0] = reinterpret_cast<float **>(exr_image.images)[idxR][i];
        out_rgb[3 * i + 1] = reinterpret_cast<float **>(exr_image.images)[idxG][i];
        out_rgb[3 * i + 2] = reinterpret_cast<float **>(exr_image.images)[idxB][i];
    }

    FreeEXRImage(&exr_image);
}

void Bitmap::loadPNG(const std::string &filename) {
    std::vector<unsigned char> image;
    unsigned width, height;

    unsigned error = lodepng::decode(image, width, height, filename);

    resize(height, width);

    if(error) {
        fprintf(stderr, "Load PNG err: %s: %s\n", error, lodepng_error_text(error));
        return;
    }

    cout << "Reading a " << cols() << "x" << rows() << " PNG file from \"" << filename << "\"" << endl;

    float *out_rgb = reinterpret_cast<float *>(data());

    for (unsigned i = 0; i < width * height; i++) {
        float alpha = static_cast<float>(image[4 * i + 3]) > 0 ? 1 : 0;
        out_rgb[3 * i + 0] = alpha * static_cast<float>(image[4 * i + 0]) / 255.f;
        out_rgb[3 * i + 1] = alpha * static_cast<float>(image[4 * i + 1]) / 255.f;
        out_rgb[3 * i + 2] = alpha * static_cast<float>(image[4 * i + 2]) / 255.f;
    }
}

void Bitmap::save(const filesystem::path &filename, bool flip) {
    if(filename.extension() == "exr")
        saveEXR(filename.str());
    else if(filename.extension() == "png")
        savePNG(filename.str(), flip);
    else
        cout << "Unknown file type" << endl;
    return;
}

void Bitmap::saveEXR(const std::string &filename) {
    cout << "Writing a " << cols() << "x" << rows()
         << " OpenEXR file to \"" << filename << "\"" << endl;

    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(cols() * rows());
    images[1].resize(cols() * rows());
    images[2].resize(cols() * rows());

    float *rgb = reinterpret_cast<float *>(data());

    for (unsigned int i = 0; i < cols() * rows(); i++) {
        images[0][i] = rgb[3*i+0];
        images[1][i] = rgb[3*i+1];
        images[2][i] = rgb[3*i+2];
    }

    float* image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = reinterpret_cast<unsigned char**>(image_ptr);
    image.width = cols();
    image.height = rows();

    header.num_channels = 3;
    header.channels = (EXRChannelInfo *) malloc(sizeof(EXRChannelInfo) * header.num_channels);
    strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types = (int *) malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int *) malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    header.compression_type = (cols() < 64 || rows() < 64) ? TINYEXR_COMPRESSIONTYPE_NONE :
                                                             TINYEXR_COMPRESSIONTYPE_ZIP;
    const char* err;
    int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Save EXR err: %s\n", err);
        return;
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

void Bitmap::savePNG(const std::string &filename, bool flip) {
    cout << "Writing a " << cols() << "x" << rows() << " PNG file to \"" << filename << "\"" << endl;

    std::vector<unsigned char> image;
    image.resize(cols() * rows() * 4);
    for (unsigned i = 0; i < cols(); ++i)
        for (unsigned j = 0; j < rows(); ++j) {
            unsigned y = flip ? rows() - 1 - j : j;
            image[4 * (i + j * cols()) + 0] = static_cast<unsigned char>((*this)(y,i).r() * 255);
            image[4 * (i + j * cols()) + 1] = static_cast<unsigned char>((*this)(y,i).g() * 255);
            image[4 * (i + j * cols()) + 2] = static_cast<unsigned char>((*this)(y,i).b() * 255);
            image[4 * (i + j * cols()) + 3] = 255;
        }

    unsigned error = lodepng::encode(filename, image, cols(), rows());

    if(error)
        fprintf(stderr, "Save PNG err: %d: %s\n", error, lodepng_error_text(error));
}
