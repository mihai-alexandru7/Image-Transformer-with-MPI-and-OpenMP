#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "bmp_io/bmp_io.h"
#include "shared_file_system_bmp_io/shared_file_system_bmp_io.h"
#include "kernels.h"
#include "operations/operations.h"

#define SHARED_FILE_SYSTEM

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    int number_of_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

    if (argc != 5) {
        if (process_rank == 0) {
            fprintf(stdout, "Usage: %s <number of threads> <operation> <input file> <output file>\n", argv[0]);
            fflush(stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int number_of_threads = strtol(argv[1], NULL, 10);

    if (number_of_threads < 1) {
        if (process_rank == 0) {
            fprintf(stdout, "Error: The number of threads must be at least 1\n");
            fflush(stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const char *operation = argv[2];
    const char *in_file_name = argv[3];
    const char *out_file_name = argv[4];

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
        if (process_rank == 0) {
            fprintf(stdout, "Unknown operation!\n");
            fflush(stdout);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    double parallel_version_start_time = 0.0;
    double parallel_version_end_time = 0.0;
    double parallel_version_elapsed_time = 0.0;

#ifdef SHARED_FILE_SYSTEM

    if (process_rank == 0) {
        fprintf(stdout, "\nLoading image from file %s\n", in_file_name);
        fflush(stdout);
    }

    MPI_File in_file_handle;

    MPI_File_open(
        MPI_COMM_WORLD,     /* the communicator */
        in_file_name,       /* the name of the file to open */
        MPI_MODE_RDONLY,    /* the file access mode */
        MPI_INFO_NULL,      /* the info object */
        &in_file_handle     /* the file handle */
    );

    int height;
    int width;

    read_image_height_and_width_from_BMP_file(
        process_rank,
        number_of_processes,
        &in_file_handle,
        &height,
        &width
    );

    int height_per_process = height / number_of_processes;
    int rest = height % number_of_processes;

    int local_height = height_per_process + ((process_rank < rest) ? 1 : 0);

    RGB *initial_local_data;
    RGB *new_local_data;

    allocate_local_data(
        process_rank,
        number_of_processes,
        &initial_local_data,
        &new_local_data,
        local_height,
        width
    );

    if (process_rank == 0) {
        fprintf(stdout, "\nStarted parallel work ...\n");
        fflush(stdout);
        parallel_version_start_time = MPI_Wtime();
    }

    read_local_data_from_BMP_file(
        process_rank,
        number_of_processes,
        &in_file_handle,
        height,
        width,
        initial_local_data,
        local_height,
        height_per_process,
        rest
    );
    
    MPI_File_close(&in_file_handle);

#else

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

    RGB *initial_local_data;
    RGB *new_local_data;

    allocate_local_data(
        process_rank,
        number_of_processes,
        &initial_local_data,
        &new_local_data,
        local_height,
        width
    );

    if (process_rank == 0) {
        fprintf(stdout, "\nStarted parallel work ...\n");
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
        width
    );

#endif

    int padding = kernel_size / 2;

    RGB *initial_local_data_with_padding;
    int local_height_with_padding;
    int width_with_padding;
    
    add_padding_to_data(
        initial_local_data,
        local_height,
        width,
        padding,
        &initial_local_data_with_padding,
        &local_height_with_padding,
        &width_with_padding
    );

    exchange_frontiers(
        process_rank,
        number_of_processes,
        initial_local_data_with_padding,
        local_height_with_padding,
        width_with_padding,
        padding
    );
    
    convolution(
        number_of_threads,
        initial_local_data_with_padding,
        local_height_with_padding,
        width_with_padding,
        new_local_data,
        local_height,
        width,
        kernel,
        kernel_size,
        padding
    );

#ifdef SHARED_FILE_SYSTEM

    if (process_rank == 0) {
        parallel_version_end_time = MPI_Wtime();
        fprintf(stdout, "\nEnded parallel work ...\n");
        fflush(stdout);
    }

    MPI_File out_file_handle;

    MPI_File_open(
        MPI_COMM_WORLD,                         /* the communicator */
        out_file_name,                          /* the name of the file to open */
        MPI_MODE_WRONLY | MPI_MODE_CREATE,      /* the file access mode */
        MPI_INFO_NULL,                          /* the info object */
        &out_file_handle                        /* the file handle */
    );

    write_local_data_to_BMP_file(
        process_rank,
        number_of_processes,
        &out_file_handle,
        height,
        width,
        new_local_data,
        local_height,
        height_per_process,
        rest
    );

    MPI_File_close(&out_file_handle);

    if (process_rank == 0) {
        printf("\nModified image saved in file %s\n", out_file_name);
    }

    if (process_rank == 0) {
        parallel_version_elapsed_time = parallel_version_end_time - parallel_version_start_time;
        fprintf(stdout, "\nParallel version elapsed time: %f seconds\n", parallel_version_elapsed_time);
        fflush(stdout);
    }

#else

    RGB *whole_new_data = NULL;

    if (process_rank == 0) {
        whole_new_data = (RGB *)malloc(height * width * sizeof(RGB));
        if (!whole_new_data) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            fflush(stderr);
            free(whole_initial_data);
            free(initial_local_data);
            free(initial_local_data_with_padding);
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
        width
    );

    if (process_rank == 0) {
        parallel_version_end_time = MPI_Wtime();
        fprintf(stdout, "\nEnded parallel work ...\n");
        fflush(stdout);
    }

    if (process_rank == 0) {
        Image *new_image = (Image *)malloc(sizeof(Image));
        new_image->height = height;
        new_image->width = width;
        new_image->data = whole_new_data;

        save_image_to_BMP_file(new_image, out_file_name);
        fprintf(stdout, "\nModified image saved in file %s\n", out_file_name);
        fflush(stdout);

        parallel_version_elapsed_time = parallel_version_end_time - parallel_version_start_time;
        fprintf(stdout, "\nParallel version elapsed time: %f seconds\n", parallel_version_elapsed_time);
        fflush(stdout);

        free(new_image->data);
        free(new_image);
    }

#endif

    free(initial_local_data);
    free(initial_local_data_with_padding);
    free(new_local_data);

    if (process_rank == 0) {
        double serial_version_start_time = 0.0;
        double serial_version_end_time = 0.0;
        double serial_version_elapsed_time = 0.0;

        fprintf(stdout, "\nLoading image from file %s\n", in_file_name);
        fflush(stdout);
        
        Image *image = read_image_from_BMP_file(in_file_name);
        if (!image) {
            fprintf(stderr, "Error reading %s\n", in_file_name);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        RGB *new_data = (RGB *)malloc(image->height * image->width * sizeof(RGB));
        if (!new_data) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fprintf(stdout, "\nStart serial work ...\n");
        fflush(stdout);

        serial_version_start_time = MPI_Wtime();

        int padding = kernel_size / 2;

        RGB *data_with_padding;
        int height_with_padding;
        int width_with_padding;

        add_padding_to_data(
            image->data,
            image->height,
            image->width,
            padding,
            &data_with_padding,
            &height_with_padding,
            &width_with_padding
        );

        free(image->data);

        convolution(
            1,
            data_with_padding,
            height_with_padding,
            width_with_padding,
            new_data,
            height,
            width,
            kernel,
            kernel_size,
            padding
        );

        serial_version_end_time = MPI_Wtime();

        fprintf(stdout, "\nEnded serial work ...\n");
        fflush(stdout);

        free(data_with_padding);

        image->data = new_data;

        save_image_to_BMP_file(image, "serial_version.bmp");

        fprintf(stdout, "\nModified image saved in file serial_version.bmp\n");
        fflush(stdout);

        serial_version_elapsed_time = serial_version_end_time - serial_version_start_time;

        fprintf(stdout, "\nSerial version elapsed time: %f seconds\n", serial_version_elapsed_time);
        fflush(stdout);

        fprintf(stdout, "\nSpeedup = %f\n", serial_version_elapsed_time / parallel_version_elapsed_time);
        fflush(stdout);

        Image *image_from_parallel_version = read_image_from_BMP_file(out_file_name);
        if (!image_from_parallel_version) {
            fprintf(stderr, "Error reading %s\n", in_file_name);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (!equal_results(new_data, image_from_parallel_version->data, image->height, image->width)) {
            fprintf(stdout, "\nSerial and parallel results are different!\n");
            fflush(stdout);
        } else {
            fprintf(stdout, "\nSerial and parallel results are the same!\n");
            fflush(stdout);
        }

        free(image->data);
        free(image);

        free(image_from_parallel_version->data);
        free(image_from_parallel_version);
    }

    MPI_Finalize();
}