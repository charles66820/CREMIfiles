#include <stdio.h>
#include <stdlib.h>

typedef unsigned int uint;

typedef struct colorRGB {
     unsigned char red;
     unsigned char green;
     unsigned char blue;
} s_colorRGB;

typedef struct colorYCbCr {
     float  y;
     float cb;
     float cr;
} s_colorYCbCr;

typedef union color {
     s_colorRGB rgb;
     s_colorYCbCr ycbcr;
     unsigned char gray;
} u_color;

typedef enum {
    RGB,
    GRAY,
    YCBCR
} e_color;

typedef struct _pixMat {
	uint width;
	uint height;
	u_color ** matrix;
}pixMat;

typedef struct image {
    uint width;
    uint height;
    e_color color_type;
    pixMat * pixels;
} s_image;


pixMat* createPixMat(uint width, uint height);
void setPixel(pixMat* m, uint x, uint y, u_color color);
u_color getPixel(pixMat* m, uint x, uint y);
void freePixMatrix(pixMat* m);


int main(void){
    s_image* image = createImage(200, 100, RGB);
    printf(
        "image is a %dx%d image of type %s\n", 
        image->height, image->width, e_colorAsChar(image->color_type)
    );
    freeImage(image);

    s_image* image2 = createImage(250, 150, YCBCR);
    printf(
        "image2 is a %dx%d image of type %s\n", 
        image2->height, image2->width, e_colorAsChar(image2->color_type)
    );
    freeImage(image2);

    return EXIT_SUCCESS;
}