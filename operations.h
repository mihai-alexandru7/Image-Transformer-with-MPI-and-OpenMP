#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "bmp_io/bmp_io.h"

void allocate_local_data(
    int process_rank,
    int number_of_processes,
    RGB **initial_local_data,
    RGB **new_local_data,
    int local_height,
    int width,
    int padding
);

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
);

void exchange_frontiers(
    int process_rank,
    int number_of_processes,
    RGB *initial_local_data,
    int local_height,
    int width,
    int padding
);

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
);

void gather_local_data_into_whole_data(
    int process_rank,
    int number_of_processes,
    RGB *whole_new_data, RGB *new_local_data,
    int height_per_process,
    int rest,
    int local_height,
    int width,
    int padding
);

RGB *add_padding(
    const RGB *data,
    int height,
    int width,
    int padding,
    int height_with_padding,
    int width_with_padding
);

RGB *serial_convolution(
    const RGB *data,
    int height,
    int width,
    const double *kernel,
    int kernel_size
);

int equal_results(
    RGB *serial_new_data,
    RGB *parallel_new_data,
    int height, 
    int width
);

#endif