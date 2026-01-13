#ifndef BMP_IO_H
#define BMP_IO_H

#include "../bmp_image.h"

/* Reads a 24-bit BMP file, builds an Image struct and returns it */
Image *read_image_from_BMP_file(const char *file_name);

/* Saves an Image struct in the given file in the 24-bit BMP format */
int save_image_to_BMP_file(const Image *image, const char *file_name);

#endif