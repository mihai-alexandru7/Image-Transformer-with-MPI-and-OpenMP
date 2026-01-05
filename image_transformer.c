#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "bmp_io/bmp_io.h"
#include "kernels.h"

void allocate_local_data(int process_rank, int number_of_processes, RGB **initial_local_data, RGB **new_local_data, int local_height, int rest, int width, int padding) {
    int extended_local_height = local_height + ((process_rank < rest) ? 1 : 0) + (2 * padding);

    *initial_local_data = (RGB *)calloc(extended_local_height * width, sizeof(RGB));
    if (!(*initial_local_data)) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    *new_local_data = (RGB *)calloc(local_height * width, sizeof(RGB));
    if (!(*new_local_data)) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fflush(stderr);
        free(*initial_local_data);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void scatter_whole_data_into_local_data(int process_rank, int number_of_processes, RGB *whole_initial_data, RGB *initial_local_data, int local_height, int rest, int width, int padding) {
    int sendcount[number_of_processes];
    int displs[number_of_processes];

    int offset = 0;
    for (int i = 0; i < number_of_processes; i++) {
        sendcount[i] = (local_height + ((i < rest) ? 1 : 0)) * width * 3;
        displs[i] = offset;
        offset += sendcount[i];
    }

    MPI_Scatterv(whole_initial_data, sendcount, displs, MPI_UNSIGNED_CHAR, initial_local_data + (padding * width), sendcount[process_rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}

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
    MPI_Init(&argc, &argv);

    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    int number_of_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

    if (process_rank == 0) {
        if (argc != 4) {
            fprintf(stdout, "Usage: %s <operation> <input file> <output file>\n", argv[0]);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
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
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (process_rank == 0) {
        Image *serial_version_image = read_image_from_BMP_file(in_file_name);
        if (!serial_version_image) {
            fprintf(stderr, "Error reading %s\n", in_file_name);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fprintf(stdout, "Start doing serial work ...\n");
        fflush(stdout);

        double serial_version_start_time = MPI_Wtime();

        RGB *serial_version_new_data = serial_convolution(serial_version_image->data, serial_version_image->height, serial_version_image->width, kernel, kernel_size);
        
        double serial_version_end_time = MPI_Wtime();

        if (!serial_version_new_data) {
            free(serial_version_image->data);
            free(serial_version_image);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        double serial_version_elapsed_time = serial_version_end_time - serial_version_start_time;

        fprintf(stdout, "Serial version elapsed time: %f seconds\n", serial_version_elapsed_time);
        fflush(stdout);

        Image *serial_version_new_image = (Image *)malloc(sizeof(Image));
        if (!serial_version_new_image) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            fflush(stderr);
            free(serial_version_image->data);
            free(serial_version_image);
            free(serial_version_new_data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        serial_version_new_image->height = serial_version_image->height;
        serial_version_new_image->width = serial_version_image->width;
        serial_version_new_image->data = serial_version_new_data;

        free(serial_version_image->data);
        free(serial_version_image);

        save_image_to_BMP_file(serial_version_new_image, out_file_name);

        free(serial_version_new_image->data);
        free(serial_version_new_image);
    }

    Image *image = NULL;
    int image_dimensions[2];

    if (process_rank == 0) {
        fprintf(stdout, "Loading image from file %s\n", in_file_name);
        fflush(stdout);

        image = read_image_from_BMP_file(in_file_name);
        if (!image) {
            fprintf(stderr, "Error reading %s\n", in_file_name);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        image_dimensions[0] = image->height;
        image_dimensions[1] = image->width;
    }

    MPI_Bcast(image_dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);

    int height = image_dimensions[0];
    int width = image_dimensions[1];
    RGB *whole_initial_data = NULL;

    if (process_rank == 0) {
        whole_initial_data = image->data;
    }

    if (process_rank == 0) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                printf("%3d ", whole_initial_data[y * width + x].r);
            }
            printf("\n");
        }
        printf("\n");
        fflush(stdout);
    }

    int local_height = height / number_of_processes;
    int rest = height % number_of_processes;

    int padding = kernel_size / 2;

    RGB *initial_local_data;
    RGB *new_local_data;

    allocate_local_data(process_rank, number_of_processes, &initial_local_data, &new_local_data, local_height, rest, width, padding);

    scatter_whole_data_into_local_data(process_rank, number_of_processes, whole_initial_data, initial_local_data, local_height, rest, width, padding);

    printf("Process: %d\n", process_rank);
    for (int y = 0; y < local_height + ((process_rank < rest) ? 1 : 0) + (2 * padding); y++) {
        for (int x = 0; x < width; x++) {
            printf("%3d ", initial_local_data[y * width + x].r);
        }
        printf("\n");
    }
    printf("\n");
    fflush(stdout);

    MPI_Finalize();
}