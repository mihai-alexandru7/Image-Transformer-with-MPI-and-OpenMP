#include <stdio.h>
#include <stdlib.h>
#include "bmp_io.h"

/* Reads a 24-bit BMP file, builds an Image struct and returns it */
Image *read_image_from_BMP_file(const char *file_name) {
    FILE *file = fopen(file_name, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", file_name);
        return NULL;
    }

    int header_size = 54;
    unsigned char header[header_size];
    if (fread(header, sizeof(unsigned char), header_size, file) != header_size) {
        fprintf(stderr, "Error: Invalid BMP header\n");
        fclose(file);
        return NULL;
    }

    if (header[0] != 'B' || header[1] != 'M') {
        fprintf(stderr, "Error: Not a valid BMP file\n");
        fclose(file);
        return NULL;
    }

    int width = *(int *)&header[18];
    int height = *(int *)&header[22];
    int bits_per_pixel = *(short *)&header[28];

    if (bits_per_pixel != 24) {
        fprintf(stderr, "Error: Only 24-bit BMPs are supported\n");
        fclose(file);
        return NULL;
    }

    int row_with_padding_size = (width * 3 + 3) & (~3);
    unsigned char *row_with_padding = (unsigned char *)malloc(row_with_padding_size * sizeof(unsigned char));
    if (!row_with_padding) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    RGB *data = (RGB *)malloc(height * width * sizeof(RGB));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        free(row_with_padding);
        return NULL;
    }

    for (int y = 0; y < height; y++) {
        fread(row_with_padding, sizeof(unsigned char), row_with_padding_size, file);
        for (int x = 0; x < width; x++) {
            data[(height - 1 - y) * width + x].b = row_with_padding[x * 3];
            data[(height - 1 - y) * width + x].g = row_with_padding[x * 3 + 1];
            data[(height - 1 - y) * width + x].r = row_with_padding[x * 3 + 2];
        }
    }
    
    free(row_with_padding);

    fclose(file);

    Image *image = (Image *)malloc(sizeof(Image));
    if (!image) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(data);
        return NULL;
    }

    image->width = width;
    image->height = height;
    image->data = data;

    return image;
}

/* Saves an Image struct in the given file in the 24-bit BMP format */
int save_image_to_BMP_file(const Image *image, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Could not create file %s\n", filename);
        return 1;
    }

    int width = image->width;
    int height = image->height;
    int header_size = 54;
    int row_with_padding_size = (width * 3 + 3) & (~3);
    int file_size = header_size + height * row_with_padding_size;

    unsigned char header[54] = {
        'B', 'M',       // Signature
        0, 0, 0, 0,     // File Size
        0, 0, 0, 0,     // Reserved
        54, 0, 0, 0,    // File Offset to Image Data
        40, 0, 0, 0,    // DIB Header Size
        0, 0, 0, 0,     // Image Width
        0, 0, 0, 0,     // Image Height
        1, 0,           // Color Planes
        24, 0,          // Bits per Pixel
        0, 0, 0, 0,     // Compression (none)
        0, 0, 0, 0,     // Image Size
        0, 0, 0, 0,     // X Pixels per Meter
        0, 0, 0, 0,     // Y Pixels per Meter
        0, 0, 0, 0,     // Colors in Color Palette
        0, 0, 0, 0      // Important Colors Count
    };

    *(int *)&header[2] = file_size;
    *(int *)&header[18] = width;
    *(int *)&header[22] = height;

    fwrite(header, sizeof(unsigned char), header_size, file);

    unsigned char *row_with_padding = (unsigned char *)calloc(row_with_padding_size, sizeof(unsigned char));
    if (!row_with_padding) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return 1;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            RGB pixel = image->data[(height - 1 - y) * width + x];
            row_with_padding[x * 3] = pixel.b;
            row_with_padding[x * 3 + 1] = pixel.g;
            row_with_padding[x * 3 + 2] = pixel.r;
        }
        fwrite(row_with_padding, sizeof(unsigned char), row_with_padding_size, file);
    }

    free(row_with_padding);

    fclose(file);

    return 0;
}