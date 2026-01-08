#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "bmp_io/bmp_io.h"
#include "operations.h"

void allocate_local_data(
    int process_rank,
    int number_of_processes,
    RGB **initial_local_data,
    RGB **new_local_data,
    int local_height,
    int width,
    int padding
) {

    int extended_local_height = local_height + (2 * padding);

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

void scatter_whole_data_into_local_data(
    int process_rank,
    int number_of_processes,
    RGB *whole_initial_data,
    RGB *initial_local_data,
    int height_per_process,
    int rest,
    int local_height,
    int width,
    int padding
) {
    int sendcount[number_of_processes];
    int displs[number_of_processes];

    int offset = 0;
    for (int i = 0; i < number_of_processes; i++) {
        sendcount[i] = (height_per_process + ((i < rest) ? 1 : 0)) * width * 3;
        displs[i] = offset;
        offset += sendcount[i];
    }

    MPI_Scatterv(
        whole_initial_data,
        sendcount,
        displs,
        MPI_UNSIGNED_CHAR,
        initial_local_data + (padding * width),
        local_height * width * 3,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );
}

void exchange_frontiers(
    int process_rank,
    int number_of_processes,
    RGB *initial_local_data,
    int local_height,
    int width,
    int padding
) {
    int top_halo = 0 * width;
    int bottom_halo = (local_height + padding) * width;

    int top_real = padding * width;
    int bottom_real = local_height * width;

    MPI_Status status;

    if (process_rank > 0) {
        MPI_Sendrecv(
            initial_local_data + top_real,
            padding * width * 3,
            MPI_UNSIGNED_CHAR,
            process_rank - 1,
            0,
            initial_local_data + top_halo,
            padding * width * 3,
            MPI_UNSIGNED_CHAR,
            process_rank - 1,
            0,
            MPI_COMM_WORLD,
            &status
        );
    }

    if (process_rank < number_of_processes - 1) {
        MPI_Sendrecv(
            initial_local_data + bottom_real,
            padding * width * 3,
            MPI_UNSIGNED_CHAR,
            process_rank + 1,
            0,
            initial_local_data + bottom_halo,
            padding * width * 3,
            MPI_UNSIGNED_CHAR,
            process_rank + 1,
            0,
            MPI_COMM_WORLD,
            &status
        );
    }
}

void compute_local_data(
    int process_rank,
    int number_of_processes,
    RGB *initial_local_data,
    RGB *new_local_data,
    int local_height,
    int width,
    const double *kernel,
    int kernel_size,
    int padding
) {
    double accumulator_b;
    double accumulator_g;
    double accumulator_r;

    int offset = kernel_size / 2;

    for (int y = padding; y < local_height + padding; y++) {
        for (int x = 0; x < width; x++) {

            accumulator_b = 0.0;
            accumulator_g = 0.0;
            accumulator_r = 0.0;

            for (int i = -offset; i <= offset; i++) {
                for (int j = -offset; j <= offset; j++) {
                    if ((x + j) >= 0 && (x + j) < width) {
                        RGB pixel = initial_local_data[(y + i) * width + (x + j)];
                        double kernel_value = kernel[(i + offset) * kernel_size + (j + offset)];
                        accumulator_b += (double)pixel.b * kernel_value;
                        accumulator_g += (double)pixel.g * kernel_value;
                        accumulator_r += (double)pixel.r * kernel_value;
                    }
                }
            }

            if (accumulator_b < 0.0) accumulator_b = 0.0;
            if (accumulator_b > 255.0) accumulator_b = 255.0;
            if (accumulator_g < 0.0) accumulator_g = 0.0;
            if (accumulator_g > 255.0) accumulator_g = 255.0;
            if (accumulator_r < 0.0) accumulator_r = 0.0;
            if (accumulator_r > 255.0) accumulator_r = 255.0;

            RGB *new_pixel = &new_local_data[(y - padding) * width + x];
            new_pixel->b = (unsigned char)accumulator_b;
            new_pixel->g = (unsigned char)accumulator_g;
            new_pixel->r = (unsigned char)accumulator_r;
        }
    }
}

void gather_local_data_into_whole_data(
    int process_rank,
    int number_of_processes,
    RGB *whole_new_data,
    RGB *new_local_data,
    int height_per_process,
    int rest,
    int local_height,
    int width,
    int padding
) {
    int recvcounts[number_of_processes];
    int displs[number_of_processes];

    int offset = 0;
    for (int i = 0; i < number_of_processes; i++) {
        recvcounts[i] = (height_per_process + ((i < rest) ? 1 : 0)) * width * 3;
        displs[i] = offset;
        offset += recvcounts[i];
    }

    MPI_Gatherv(
        new_local_data,
        local_height * width * 3,
        MPI_UNSIGNED_CHAR,
        whole_new_data,
        recvcounts,
        displs,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );
}

RGB *add_padding(
    const RGB *data,
    int height,
    int width,
    int padding,
    int height_with_padding,
    int width_with_padding
) {
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

RGB *serial_convolution(
    const RGB *data,
    int height,
    int width,
    const double *kernel,
    int kernel_size
) {
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

int equal_results(
    RGB *serial_new_data,
    RGB *parallel_new_data,
    int height,
    int width
) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int different_r = serial_new_data[y * width + x].r != parallel_new_data[y * width + x].r;
            int different_g = serial_new_data[y * width + x].g != parallel_new_data[y * width + x].g;
            int different_b = serial_new_data[y * width + x].b != parallel_new_data[y * width + x].b;
            if (different_r || different_g || different_b) {
                return 0;
            }
        }
    }
    return 1;
}