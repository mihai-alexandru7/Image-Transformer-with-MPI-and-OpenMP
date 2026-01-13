#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "operations.h"

void allocate_local_data(
    int process_rank,           /* in */
    int number_of_processes,    /* in */
    RGB **initial_local_data,   /* out */
    RGB **new_local_data,       /* out */
    int local_height,           /* in */
    int width                   /* in */
) {
    *initial_local_data = (RGB *)calloc(local_height * width, sizeof(RGB));
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
    int process_rank,           /* in */
    int number_of_processes,    /* in */
    RGB *whole_initial_data,    /* in */
    RGB *initial_local_data,    /* in / out */
    int height_per_process,     /* in */
    int rest,                   /* in */
    int local_height,           /* in */
    int width                   /* in */
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
        initial_local_data,
        local_height * width * 3,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );
}

void add_padding_to_data(
    const RGB *data,            /* in */
    int height,                 /* in */
    int width,                  /* in */
    int padding,                /* in */
    RGB **data_with_padding,    /* out */
    int *height_with_padding,   /* out */
    int *width_with_padding     /* out */
) {
    *height_with_padding = height + (2 * padding);
    *width_with_padding = width + (2 * padding);

    *data_with_padding = (RGB *)calloc((*height_with_padding) * (*width_with_padding), sizeof(RGB));
    if (!(*data_with_padding)) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            (*data_with_padding)[(y + padding) * (*width_with_padding) + (x + padding)] = data[y * width + x];
        }
    }
}

void exchange_frontiers(
    int process_rank,                       /* in */
    int number_of_processes,                /* in */
    RGB *initial_local_data_with_padding,   /* in / out */
    int local_height_with_padding,          /* in */
    int width_with_padding,                 /* in */
    int padding                             /* in */
) {
    int top_halo = 0 * width_with_padding;
    int bottom_halo = (local_height_with_padding - padding) * width_with_padding;

    int top_real = padding * width_with_padding;
    int bottom_real = (local_height_with_padding - 2 * padding) * width_with_padding;

    MPI_Status status;

    if (process_rank > 0) {
        MPI_Sendrecv(
            initial_local_data_with_padding + top_real,
            padding * width_with_padding * 3,
            MPI_UNSIGNED_CHAR,
            process_rank - 1,
            0,
            initial_local_data_with_padding + top_halo,
            padding * width_with_padding * 3,
            MPI_UNSIGNED_CHAR,
            process_rank - 1,
            0,
            MPI_COMM_WORLD,
            &status
        );
    }

    if (process_rank < number_of_processes - 1) {
        MPI_Sendrecv(
            initial_local_data_with_padding + bottom_real,
            padding * width_with_padding * 3,
            MPI_UNSIGNED_CHAR,
            process_rank + 1,
            0,
            initial_local_data_with_padding + bottom_halo,
            padding * width_with_padding * 3,
            MPI_UNSIGNED_CHAR,
            process_rank + 1,
            0,
            MPI_COMM_WORLD,
            &status
        );
    }
}

void convolution(
    int number_of_threads,          /* in */
    const RGB *data_with_padding,   /* in */
    int height_with_padding,        /* in */
    int width_with_padding,         /* in */
    RGB *new_data,                  /* in / out */
    int height,                     /* in */
    int width,                      /* in */
    const double *kernel,           /* in */
    int kernel_size,                /* in */
    int padding                     /* in */
) {
    double accumulator_b;
    double accumulator_g;
    double accumulator_r;

    int offset = kernel_size / 2;

    #pragma omp parallel for num_threads(number_of_threads) \
        private(accumulator_b, accumulator_g, accumulator_r) \
        schedule(static)
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
}

void gather_local_data_into_whole_data(
    int process_rank,
    int number_of_processes,
    RGB *whole_new_data,
    RGB *new_local_data,
    int height_per_process,
    int rest,
    int local_height,
    int width
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