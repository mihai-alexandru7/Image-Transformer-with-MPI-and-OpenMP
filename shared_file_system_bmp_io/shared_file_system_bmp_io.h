#ifndef SHARED_FILE_SYSTEM_BMP_IO_H
#define SHARED_FILE_SYSTEM_BMP_IO_H

#include "mpi.h"
#include "../bmp_image.h"

void read_image_height_and_width_from_BMP_file(
    int process_rank,           /* in */
    int number_of_processes,    /* in */
    MPI_File *file_handle,      /* in */
    int *image_height,          /* out */
    int *image_width            /* out */
);

void read_local_data_from_BMP_file(
    int process_rank,           /* in */
    int number_of_processes,    /* in */
    MPI_File *file_handle,      /* in */
    int height,                 /* in */
    int width,                  /* in */
    RGB *initial_local_data,    /* out */
    int local_height,           /* in */
    int height_per_process,     /* in */
    int rest                    /* in */
);

void write_local_data_to_BMP_file(
    int process_rank,           /* in */
    int number_of_processes,    /* in */
    MPI_File *file_handle,      /* in */
    int height,                 /* in */
    int width,                  /* in */
    RGB *new_local_data,        /* in */
    int local_height,           /* in */
    int height_per_process,     /* in */
    int rest                    /* in */
);

#endif