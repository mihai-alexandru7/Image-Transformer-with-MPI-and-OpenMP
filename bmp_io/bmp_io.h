#ifndef BMP_IO_H
#define BMP_IO_H

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

/* Reads a 24-bit BMP file, builds an Image struct and returns it */
Image *read_image_from_BMP_file(const char *file_name);

/* Saves an Image struct in the given file in the 24-bit BMP format */
int save_image_to_BMP_file(const Image *image, const char *file_name);

#endif