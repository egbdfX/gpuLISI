#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"
#include <vector>
#include <iostream>
#include <chrono>

int splitlis(float* input_data_1, float* input_data_2, float* result_array, size_t size, size_t unit_size, size_t ima);

using namespace std;

float* read_fits_image(const char* filename, long* naxes) {
    fitsfile *fptr; // FITS file pointer
    int status = 0; // Status variable for FITSIO functions

    // Open the FITS file
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fits_report_error(stderr, status);
        return NULL;
    }

    // Get image dimensions
    int naxis;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }
    fits_get_img_size(fptr, 2, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Allocate memory for image data
    float *image_data = (float *)malloc(naxes[0] * naxes[1] * sizeof(float));
    if (image_data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Read image data
    fits_read_img(fptr, TFLOAT, 1, naxes[0] * naxes[1], NULL, image_data, NULL, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Close the FITS file
    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        return NULL;
    }

    return image_data;
}

// Function to write FITS image
int write_fits_image(const char* filename, float *image_data, long* naxes) {
    fitsfile *fptr; // FITS file pointer
    int status = 0; // Status variable for FITSIO functions

    // Create new FITS file
    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(stderr, status);
        return status;
    }

    // Create image extension
    long naxis = 2; // 2-dimensional image
    fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    // Write image data
    fits_write_img(fptr, TFLOAT, 1, naxes[0] * naxes[1], image_data, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    // Close the FITS file
    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        return status;
    }

    return 0; // Success
}

int main(int argc, char* argv[]) {
    char input_data_file_1[1000];
    char input_data_file_2[1000];
    size_t unit_size = 64;//400;
    size_t ima = 2048;//60000;
    size_t imapow = ima*ima;
    size_t unit_num = ima/unit_size;
    long imasize[2];
    
	// Process the first file
    sprintf(input_data_file_1, "%s", argv[1]);
    // Read FITS image
    float *input_data_1 = read_fits_image(input_data_file_1, imasize);
    if (input_data_1 == NULL) {
        return 1; // Error handling
    }
    
    // Process the second file
    sprintf(input_data_file_2, "%s", argv[2]);
	// Read FITS image
    float *input_data_2 = read_fits_image(input_data_file_2, imasize);
    if (input_data_2 == NULL) {
        return 1; // Error handling
    }
    
    float* data_1_array = input_data_1;
    float* data_2_array = input_data_2;
    
    float* result_array = (float*)malloc(unit_num*unit_num*sizeof(float));
    
	splitlis(data_1_array, data_2_array, result_array, imapow, unit_size, ima);
	
	// Write FITS image
	long naxes[2] = {long(unit_num), long(unit_num)};
    int status = write_fits_image("output_LISI.fits", result_array, naxes);
    if (status) {
        fprintf(stderr, "Error writing FITS image\n");
        return 1; // Error handling
    }
}
