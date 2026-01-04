#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmp_io/bmp_io.h"
#include "kernels.h"

RGB *add_padding(const RGB *data, int height, int width, int padding, int height_with_padding, int width_with_padding) {
    RGB *data_with_padding = (RGB *)calloc(height_with_padding * width_with_padding, sizeof(RGB));
    if (!data_with_padding) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            data_with_padding[(y + padding) * width_with_padding + (x + padding)] = data[y * width + x];
        }
    }

    return data_with_padding;
}

RGB *serial_convolution(const RGB *data, int height, int width, const double *kernel, int kernel_size) {
    int padding = kernel_size / 2;
    int height_with_padding = height + (2 * padding);
    int width_with_padding = width + (2 * padding);

    RGB *data_with_padding = add_padding(data, height, width, padding, height_with_padding, width_with_padding);
    if (!data_with_padding) {
        return NULL;
    }

    RGB *new_data = (RGB *)malloc(height * width * sizeof(RGB));
    if (!new_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(data_with_padding);
        return NULL;
    }

    double accumulator_b;
    double accumulator_g;
    double accumulator_r;

    int offset = kernel_size / 2;

    for (int y = padding; y < height_with_padding - padding; y++) {
        for (int x = padding; x < width_with_padding - padding; x++) {

            accumulator_b = 0.0;
            accumulator_g = 0.0;
            accumulator_r = 0.0;

            for (int i = -offset; i <= offset; i++) {
                for (int j = -offset; j <= offset; j++) {
                    RGB pixel = data_with_padding[(y + i) * width_with_padding + (x + j)];
                    double kernel_value = kernel[(i + offset) * kernel_size + (j + offset)];
                    accumulator_b += (double)pixel.b * kernel_value;
                    accumulator_g += (double)pixel.g * kernel_value;
                    accumulator_r += (double)pixel.r * kernel_value;
                }
            }

            if (accumulator_b < 0.0) accumulator_b = 0.0;
            if (accumulator_b > 255.0) accumulator_b = 255.0;
            if (accumulator_g < 0.0) accumulator_g = 0.0;
            if (accumulator_g > 255.0) accumulator_g = 255.0;
            if (accumulator_r < 0.0) accumulator_r = 0.0;
            if (accumulator_r > 255.0) accumulator_r = 255.0;

            RGB *new_pixel = &new_data[(y - padding) * width + (x - padding)];
            new_pixel->b = (unsigned char)accumulator_b;
            new_pixel->g = (unsigned char)accumulator_g;
            new_pixel->r = (unsigned char)accumulator_r;
        }
    }

    free(data_with_padding);

    return new_data;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stdout, "Usage: %s <operation> <input file> <output file>\n", argv[0]);
        return 1;
    }

    const char *operation = argv[1];
    const char *in_file_name = argv[2];
    const char *out_file_name = argv[3];

    const double *kernel = NULL;
    int kernel_size;

    if (strcmp(operation, "RIDGE") == 0) {
        kernel = RIDGE_KERNEL;
        kernel_size = 3;
    } else if (strcmp(operation, "EDGE") == 0) {
        kernel = EDGE_KERNEL;
        kernel_size = 3;
    } else if (strcmp(operation, "SHARPEN") == 0) {
        kernel = SHARPEN_KERNEL;
        kernel_size = 3;
    } else if (strcmp(operation, "BOXBLUR") == 0) {
        kernel = BOX_BLUR_KERNEL;
        kernel_size = 3;
    } else if (strcmp(operation, "GAUSSIANBLUR3") == 0) {
        kernel = GAUSSIAN_BLUR_3x3_KERNEL;
        kernel_size = 3;
    } else if (strcmp(operation, "GAUSSIANBLUR5") == 0) {
        kernel = GAUSSIAN_BLUR_5x5_KERNEL;
        kernel_size = 5;
    } else if (strcmp(operation, "UNSHARP5") == 0) {
        kernel = UNSHARP_MASKING_5x5_KERNEL;
        kernel_size = 5;
    } else {
        fprintf(stdout, "Unknown operation!");
        return 1;
    }

    Image *image = read_image_from_BMP_file(in_file_name);
    if (!image) {
        return 1;
    }

    RGB *new_data = serial_convolution(image->data, image->height, image->width, kernel, kernel_size);
    if (!new_data) {
        free(image->data);
        free(image);
        return 1;
    }

    Image *new_image = (Image *)malloc(sizeof(Image));
    if (!new_image) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(image->data);
        free(image);
        free(new_data);
        return 1;
    }

    new_image->height = image->height;
    new_image->width = image->width;
    new_image->data = new_data;

    free(image->data);
    free(image);

    save_image_to_BMP_file(new_image, out_file_name);

    free(new_image->data);
    free(new_image);

    return 0;
}