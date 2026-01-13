#ifndef BMP_IMAGE_H
#define BMP_IMAGE_H

/* Data structures for representing 24-bit BMP images in memory */

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} RGB;

typedef struct {
    int width;
    int height;
    RGB *data;
} Image;

#endif