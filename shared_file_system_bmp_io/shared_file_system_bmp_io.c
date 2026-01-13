#include <stdio.h>
#include <stdlib.h>
#include "shared_file_system_bmp_io.h"

void read_image_height_and_width_from_BMP_file(
    int process_rank,           /* in */
    int number_of_processes,    /* in */
    MPI_File *file_handle,      /* in */
    int *image_height,          /* out */
    int *image_width            /* out */
) {
    int header_size = 54;
    unsigned char header[header_size];

    MPI_Status status;

    MPI_File_read_at_all(
        *file_handle,       /* the file handle */
        0,                  /* the file offset */
        header,             /* the initial address of the buffer */
        header_size,        /* the number of elements in the buffer */
        MPI_UNSIGNED_CHAR,  /* the datatype of each buffer element */
        &status             /* the status object */
    );

    if (header[0] != 'B' || header[1] != 'M') {
        if (process_rank == 0) {
            fprintf(stderr, "Error: Not a valid BMP file\n");
            fflush(stderr);
        }
        MPI_File_close(file_handle);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    *image_width = *(int *)&header[18];
    *image_height = *(int *)&header[22];
    int bits_per_pixel = *(short *)&header[28];

    if (bits_per_pixel != 24) {
        if (process_rank == 0) {
            fprintf(stderr, "Error: Only 24-bit BMPs are supported\n");
            fflush(stderr);
        }
        MPI_File_close(file_handle);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

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
) {
    int row_with_padding_size = (width * 3 + 3) & (~3);

    unsigned char *rows_with_padding = (unsigned char *)malloc(local_height * row_with_padding_size * sizeof(unsigned char));
    if (!rows_with_padding) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fflush(stderr);
        MPI_File_close(file_handle);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int start_row = 0;
    if (process_rank < rest) {
        start_row = height - (process_rank + 1) * (height_per_process + 1);
    } else {
        start_row = height - (process_rank + 1) * height_per_process - rest;
    }

    MPI_Status status;

    MPI_Offset file_offset = 54 + start_row * row_with_padding_size;

    MPI_File_read_at_all(
        *file_handle,                           /* the file handle */
        file_offset,                            /* the file offset */
        rows_with_padding,                      /* the initial address of the buffer */
        local_height * row_with_padding_size,   /* the number of elements in the buffer */
        MPI_UNSIGNED_CHAR,                      /* the datatype of each buffer element */
        &status                                 /* the status object */
    );

    for (int y = 0; y < local_height; y++) {
        for (int x = 0; x < width; x++) {
            initial_local_data[(local_height - 1 - y) * width + x].b = rows_with_padding[y * row_with_padding_size + x * 3];
            initial_local_data[(local_height - 1 - y) * width + x].g = rows_with_padding[y * row_with_padding_size + x * 3 + 1];
            initial_local_data[(local_height - 1 - y) * width + x].r = rows_with_padding[y * row_with_padding_size + x * 3 + 2];
        }
    }

    free(rows_with_padding);
}

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
) {
    int header_size = 54;
    int row_with_padding_size = (width * 3 + 3) & (~3);
    int file_size = header_size + height * row_with_padding_size;

    if (process_rank == 0) {
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

        MPI_Status status;

        MPI_File_write_at(
            *file_handle,       /* the file handle */
            0,                  /* the file offset */
            header,             /* the initial address of the buffer */
            header_size,        /* the number of elements in the buffer */
            MPI_UNSIGNED_CHAR,  /* the datatype of each buffer element */
            &status             /* the status object */
        );
    }

    unsigned char *rows_with_padding = (unsigned char *)calloc(local_height * row_with_padding_size, sizeof(unsigned char));
    if (!rows_with_padding) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fflush(stderr);
        MPI_File_close(file_handle);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int y = 0; y < local_height; y++) {
        for (int x = 0; x < width; x++) {
            RGB pixel = new_local_data[(local_height - 1 - y) * width + x];
            rows_with_padding[y * row_with_padding_size + x * 3] = pixel.b;
            rows_with_padding[y * row_with_padding_size + x * 3 + 1] = pixel.g;
            rows_with_padding[y * row_with_padding_size + x * 3 + 2] = pixel.r;
        }
    }

    int start_row = 0;
    if (process_rank < rest) {
        start_row = height - (process_rank + 1) * (height_per_process + 1);
    } else {
        start_row = height - (process_rank + 1) * height_per_process - rest;
    }

    MPI_Status status;

    MPI_Offset file_offset = header_size + start_row * row_with_padding_size;

    MPI_File_write_at_all(
        *file_handle,                           /* the file handle */
        file_offset,                            /* the file offset */
        rows_with_padding,                      /* the initial address of the buffer */
        local_height * row_with_padding_size,   /* the number of elements in the buffer */
        MPI_UNSIGNED_CHAR,                      /* the datatype of each buffer element */
        &status                                 /* the status object */
    );

    free(rows_with_padding);
}