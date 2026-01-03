#include <stdio.h>
#include <stdlib.h>
#include "bmp_io.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stdout, "Usage: %s <input file> <output file>\n", argv[0]);
        return 1;
    }

    const char *in_file_name = argv[1];
    const char *out_file_name = argv[2];

    Image *image = read_image_from_BMP_file(in_file_name);
    if (!image) {
        return 1;
    }

    save_image_to_BMP_file(image, out_file_name);
    
    free(image->data);
    free(image);

    return 0;
}