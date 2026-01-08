#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "bmp_io/bmp_io.h"
#include "kernels.h"
#include "operations.h"

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
        fprintf(stdout, "Unknown operation!\n");
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    double parallel_version_start_time = 0.0;
    double parallel_version_end_time = 0.0;
    double parallel_version_elapsed_time = 0.0;

    int image_dimensions[2];
    RGB *whole_initial_data = NULL;

    if (process_rank == 0) {
        fprintf(stdout, "\nLoading image from file %s\n", in_file_name);
        fflush(stdout);

        Image *image = read_image_from_BMP_file(in_file_name);
        if (!image) {
            fprintf(stderr, "Error reading %s\n", in_file_name);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        image_dimensions[0] = image->height;
        image_dimensions[1] = image->width;
        whole_initial_data = image->data;

        free(image);
    }

    MPI_Bcast(image_dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);

    int height = image_dimensions[0];
    int width = image_dimensions[1];

    int height_per_process = height / number_of_processes;
    int rest = height % number_of_processes;

    int local_height = height_per_process + ((process_rank < rest) ? 1 : 0);

    int padding = kernel_size / 2;

    RGB *initial_local_data;
    RGB *new_local_data;

    allocate_local_data(
        process_rank,
        number_of_processes,
        &initial_local_data,
        &new_local_data,
        local_height,
        width,
        padding
    );

    if (process_rank == 0) {
        fprintf(stdout, "\nStart doing parallel work ...\n");
        fflush(stdout);
        parallel_version_start_time = MPI_Wtime();
    }

    scatter_whole_data_into_local_data(
        process_rank,
        number_of_processes,
        whole_initial_data,
        initial_local_data,
        height_per_process,
        rest,
        local_height,
        width,
        padding
    );

    exchange_frontiers(
        process_rank,
        number_of_processes,
        initial_local_data,
        local_height,
        width,
        padding
    );

    compute_local_data(
        process_rank,
        number_of_processes,
        initial_local_data,
        new_local_data,
        local_height,
        width,
        kernel,
        kernel_size,
        padding
    );

    RGB *whole_new_data = NULL;

    if (process_rank == 0) {
        whole_new_data = (RGB *)malloc(height * width * sizeof(RGB));
        if (!whole_new_data) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            fflush(stderr);
            free(whole_initial_data);
            free(initial_local_data);
            free(new_local_data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    gather_local_data_into_whole_data(
        process_rank,
        number_of_processes,
        whole_new_data,
        new_local_data,
        height_per_process,
        rest,
        local_height,
        width,
        padding
    );

    if (process_rank == 0) {
        parallel_version_end_time = MPI_Wtime();
        parallel_version_elapsed_time = parallel_version_end_time - parallel_version_start_time;
        fprintf(stdout, "\nParallel version elapsed time: %f seconds\n", parallel_version_elapsed_time);
        fflush(stdout);
    }

    free(initial_local_data);
    free(new_local_data);

    if (process_rank == 0) {
        Image *image = read_image_from_BMP_file(in_file_name);
        if (!image) {
            fprintf(stderr, "Error reading %s\n", in_file_name);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        double serial_version_start_time = 0.0;
        double serial_version_end_time = 0.0;
        double serial_version_elapsed_time = 0.0;

        fprintf(stdout, "\nStart doing serial work ...\n");
        fflush(stdout);

        serial_version_start_time = MPI_Wtime();

        RGB *serial_version_new_data = serial_convolution(image->data, image->height, image->width, kernel, kernel_size);
        
        serial_version_end_time = MPI_Wtime();

        if (!serial_version_new_data) {
            free(image->data);
            free(image);
            free(serial_version_new_data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        serial_version_elapsed_time = serial_version_end_time - serial_version_start_time;

        fprintf(stdout, "\nSerial version elapsed time: %f seconds\n", serial_version_elapsed_time);
        fflush(stdout);

        fprintf(stdout, "\nSpeedup = %f\n", serial_version_elapsed_time / parallel_version_elapsed_time);
        fflush(stdout);

        if (!equal_results(serial_version_new_data, whole_new_data, height, width)) {
            fprintf(stdout, "\nSerial and parallel results are different!\n");
            fflush(stdout);
        } else {
            fprintf(stdout, "\nSerial and parallel results are the same!\n");
            fflush(stdout);
        }

        free(image->data);
        free(image);

        free(serial_version_new_data);

        Image *new_image = (Image *)malloc(sizeof(Image));
        new_image->height = height;
        new_image->width = width;
        new_image->data = whole_new_data;

        save_image_to_BMP_file(new_image, out_file_name);
        printf("\nModified image saved in file %s\n", out_file_name);

        free(whole_new_data);
        free(new_image);
    }

    MPI_Finalize();
}