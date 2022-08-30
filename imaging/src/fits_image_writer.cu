// Copyright 2021 Adam Campbell, Andrew Ensor, Anthony Griffin, Seth Hall
// Copyright 2021 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "../fits_image_writer.h" // Andrew: in SEP change this to "../fits_image_writer.h" 

/**********************************************************************
 * Appends a string keyword header to the FITS header unit with comment appended, all padded to a total of 80 characters
 * Note keyword is presumed to be all uppercase with no special characters, will be trimmed if over 8 chars
 * Note value string is presumed to be at most 20 characters and will appear right aligned
 * Note comment will be trimmed or padded with spaces to make a total of 80 characters for header
 * Returns the number of characters in the header appended, should always be 80 characters for FITS
 **********************************************************************/
int set_header_string_value(char *fits_header_unit, const char *keyword, const char *value, const char *comment)
{
    char fits_header[81];
    memset(fits_header, '\0', sizeof(fits_header));
    sprintf(fits_header, "%-8.8s= %20.20s / %-47.47s",
        keyword, // left align keyword and trim or pad with spaces to 8 characters
        value, // right align value and trim or pad with spaces to 20 characters
        comment // left align comment and trim or pad to remaining characters to give 80 char total
        );
    strcat(fits_header_unit, fits_header);
    return strlen(fits_header);
}

/**********************************************************************
 * Appends a quoted string keyword header to the FITS header unit with comment appended, all padded to a total of 80 characters
 * Note value string is presumed to be at most 18 characters and will appear left aligned within single quotes
 **********************************************************************/
int set_header_quoted_value(char *fits_header_unit, const char *keyword, const char *value, const char *comment)
{
    char fits_header[81];
    memset(fits_header, '\0', sizeof(fits_header));
    char value_quoted[strlen(value)+3];
    memset(value_quoted, '\0', sizeof(value_quoted));
    sprintf(value_quoted, "'%-.18s'", value);
    sprintf(fits_header, "%-8.8s= %-20.20s / %-47.47s",
        keyword, // left align keyword and trim or pad with spaces to 8 characters
        value_quoted, // left align value and trim or pad with spaces to 18 characters enclosed in single quotes
        comment // left align comment and trim or pad to remaining characters to give 80 char total
        );
    strcat(fits_header_unit, fits_header);
    return strlen(fits_header);
}

/**********************************************************************
 * Appends a double value header to the FITS header unit with comment appended, all padded to a total of 80 characters
 * Note value will appear using printf %.14g format specifier at most 20 characters and will appear right aligned
 **********************************************************************/
int set_header_double_value(const char *fits_header_unit, const char *keyword, double value, const char *comment)
{
    char value_string[21];
    memset(value_string, '\0', sizeof(value_string));
    sprintf(value_string, "%.14g", value);
    return set_header_string_value((char*)fits_header_unit, keyword, value_string, comment);
}

/**********************************************************************
 * Appends an int value header to the FITS header unit with comment appended, all padded to a total of 80 characters
 * Note value will appear using printf %d format specifier at most 20 characters and will appear right aligned
 **********************************************************************/
int set_header_int_value(const char *fits_header_unit, const char *keyword, int value, const char *comment)
{
    char value_string[21];
    memset(value_string, '\0', sizeof(value_string));
    sprintf(value_string, "%d", value);
    return set_header_string_value((char*)fits_header_unit, (char*)keyword, value_string, comment);
}

/**********************************************************************
 * Appends a text header to the FITS header unit, padded to a total of 80 characters
 * Note text string is presumed to be at most 71 characters and will appear left aligned
 * Note this is typically used for COMMENT or HISTORY headers
 **********************************************************************/
int set_header_text(const char *fits_header_unit, const char *keyword, const char *text)
{
    char fits_header[81];
    memset(fits_header, '\0', sizeof(fits_header));
    sprintf(fits_header, "%-8.8s %-71.71s",
        keyword, // left align keyword and trim or pad with spaces to 8 characters
        text // left align text and trim or pad to remaining characters to give 80 char total
        );
    strcat((char*)fits_header_unit, fits_header);
    return strlen(fits_header);
}

/**********************************************************************
 * Create a FITS header unit and adds the required SIMPLE and BITPIX headers
 * Note the created FITS header unit must be destroyed in output_fits_header_unit
 * Returns the char* array that will hold the FITS headers
 **********************************************************************/
char *create_fits_header_unit()
{
    const int MAX_HEADER_BLOCKS = 5; // each header block allows 36 headers
    const size_t FITS_HEADER_CAPACITY = sizeof(char)*2880*MAX_HEADER_BLOCKS+1; // FITS has 2880 chars per block, plus NUL string terminator
    char *fits_header_unit = (char *)malloc(FITS_HEADER_CAPACITY);
    memset(fits_header_unit, '\0', FITS_HEADER_CAPACITY);
    set_header_string_value(fits_header_unit, "SIMPLE", "T", "file conforms to FITS standard");
    set_header_int_value(fits_header_unit, "BITPIX", -(int)(sizeof(PRECISION)*8), "fp32 or fp64 precision data pixels"); // -32 for SINGLE pixels, -64 for DOUBLE pixels    
    return fits_header_unit;
}


/**********************************************************************
 * Appends the FITS headers for NAXIS, each NAXISi, WCSAXES, and each CRPIXi set to centre of its axis
 **********************************************************************/
void set_fits_axes_headers(char *fits_header_unit, int image_width, int image_height, int num_stokes, int num_freq)
{
    set_header_int_value(fits_header_unit, "NAXIS", 4, "number of data axes"); // two-dimensional image with one polarisation product and one frequency
    set_header_int_value(fits_header_unit, "NAXIS1", image_width, "length of data axis 1"); // width of image
    set_header_int_value(fits_header_unit, "NAXIS2", image_height, "length of data axis 2"); // height of image
    set_header_int_value(fits_header_unit, "NAXIS3", num_stokes, "length of data axis 3"); // number of stokes parameters (polarisation products)
    set_header_int_value(fits_header_unit, "NAXIS4", num_freq, "length of data axis 4"); // number of frequency channels
    set_header_int_value(fits_header_unit, "WCSAXES", 4, "number of coordinate axes"); // number of coordinate axes
    set_header_double_value(fits_header_unit, "CRPIX1", ceil((image_width+1)/2.0), "pixel coordinate of reference point"); // central reference point
    set_header_double_value(fits_header_unit, "CRPIX2", ceil((image_height+1)/2.0), "pixel coordinate of reference point"); // central reference point
    set_header_double_value(fits_header_unit, "CRPIX3", ceil((num_stokes+1)/2.0), "pixel coordinate of reference point"); // central reference point
    set_header_double_value(fits_header_unit, "CRPIX4", ceil((num_freq+1)/2.0), "pixel coordinate of reference point"); // central reference point
}


/**********************************************************************
 * Appends the FITS headers for each CDELTi and each CUNITi
 **********************************************************************/
void set_fits_axes_increments(char *fits_header_unit, double pixel_size, bool right_ascension, double frequency_inc_hz)
{
    const double pixel_size_degrees = pixel_size*180.0/M_PI; // convert to degrees
    set_header_double_value(fits_header_unit, "CDELT1", right_ascension ? -pixel_size_degrees : pixel_size_degrees, "[deg] coordinate increment at reference point");
    set_header_double_value(fits_header_unit, "CDELT2", pixel_size_degrees, "[deg] coordinate increment at reference point");
    set_header_double_value(fits_header_unit, "CDELT3", 1.0, "coordinate increment at reference point");
    set_header_double_value(fits_header_unit, "CDELT4", frequency_inc_hz, "[Hz] coordinate increment at reference point");
    set_header_quoted_value(fits_header_unit, "CUNIT1", "deg", "units of coordinate increment and value");
    set_header_quoted_value(fits_header_unit, "CUNIT2", "deg", "units of coordinate increment and value");
//    set_header_quoted_value(fits_header_unit, "CUNIT3", "", "units of coordinate increment and value"); // note CUNIT3 absent in RASCIL sample FITS
    set_header_quoted_value(fits_header_unit, "CUNIT4", "Hz", "units of coordinate increment and value");
}


/**********************************************************************
 * Appends the FITS headers for each CTYPEi, each CRVALi, LONPOLE, and LATPOLE
 **********************************************************************/
void set_fits_world_coordinate_system(char *fits_header_unit, double coord_ref_width, double coord_ref_height,
    double frequency_hz, double celestial_pole_long_deg, double celestial_pole_lat_deg)
{
    set_header_quoted_value(fits_header_unit, "CTYPE1", "RA---SIN", "right ascension, orthographic/synthesis proj");
    set_header_quoted_value(fits_header_unit, "CTYPE2", "DEC--SIN", "declination, orthographic/synthesis proj");
    set_header_quoted_value(fits_header_unit, "CTYPE3", "STOKES", "coordinate type");
    set_header_quoted_value(fits_header_unit, "CTYPE4", "FREQ", "frequency (linear)");
    set_header_double_value(fits_header_unit, "CRVAL1", coord_ref_width, "[deg] coordinate value at reference point");
    set_header_double_value(fits_header_unit, "CRVAL2", coord_ref_height, "[deg] coordinate value at reference point");
    set_header_double_value(fits_header_unit, "CRVAL3", 1.0, "coordinate value at reference point");
    set_header_double_value(fits_header_unit, "CRVAL4", frequency_hz, "[Hz] coordinate value at reference point");
    set_header_double_value(fits_header_unit, "LONPOLE", celestial_pole_long_deg, "[deg] native longitude of celestial pole");
    set_header_double_value(fits_header_unit, "LATPOLE", celestial_pole_lat_deg, "[deg] native latitude of celestial pole");
    set_header_quoted_value(fits_header_unit, "RADESYS", "ICRS", "equatorial coordinate system");
}


/**********************************************************************
 * Appends the FITS headers for DATEREF, MJDREFI, MJDREFF
 * If date_reference is NULL then uses current date 
 **********************************************************************/
void set_fits_date_reference(char *fits_header_unit, char *date_reference)
{
    if (date_reference == NULL)
    {
        // use current local system time as the date reference
        time_t current_time = time(NULL);
        struct tm *local_time = localtime(&current_time);
        char current_date[11]; // will store date in format YYYY-MM-DD with NUL string terminator
        memset(current_date, '\0', sizeof(date_reference));
        sprintf(current_date, "%04d-%02d-%02d", local_time->tm_year+1900, local_time->tm_mon+1, local_time->tm_mday);
        set_header_quoted_value(fits_header_unit, "DATEREF", current_date, "ISO-8601 fiducial time");
    }
    else
    {
        set_header_quoted_value(fits_header_unit, "DATEREF", date_reference, "ISO-8601 fiducial time");
    }
    set_header_double_value(fits_header_unit, "MJDREFI", 0.0, "[d] MJD of fiducial time, integer part");
    set_header_double_value(fits_header_unit, "MJDREFF", 0.0, "[d] MJD of fiducial time, fractional part");
}


/**********************************************************************
 * Appends the END header to the FITS header unit and pads header unit to be exactly a multple of 2880 characters
 * Then outputs the FITS header unit to the file and frees the FITS header unit
 * Returns the number of characters output for the header including FITS block padding
 **********************************************************************/
size_t output_fits_header_unit(char *fits_header_unit, FILE *fits_file)
{
    char fits_header[81];
    memset(fits_header, '\0', sizeof(fits_header));
    sprintf(fits_header, "%-80.80s", "END");
    strcat(fits_header_unit, fits_header);
    int num_header_blocks = (int)ceil(1.0*strlen(fits_header_unit)/2880);
    int padded_header_length = num_header_blocks * 2880;
    sprintf(fits_header_unit, "%-*.*s", padded_header_length, padded_header_length, fits_header_unit);
    fputs(fits_header_unit, fits_file);
    size_t num_header_chars = strlen(fits_header_unit);
    free(fits_header_unit);
    return num_header_chars;
}


/**********************************************************************
 * Converts the PRECISION image values to be in big endian order and outputs them row by row
 * and then by stokes and then by frequency
 * Returns the number of characters output for the data including FITS block padding
 **********************************************************************/
size_t output_fits_data_unit(PRECISION *image, int image_width, int image_height, int num_stokes, int num_freq, FILE *fits_file)
{
    size_t num_data_chars = 0;
    int num_pixels = image_width * image_height * num_stokes * num_freq;
    for (int i=0; i<num_pixels; i++)
    {
        PRECISION pixel = image[i];
        // convert the pixel to be big endian
        unsigned char *pixel_bytes = (unsigned char *)&pixel;
        int num_bytes_to_swap = sizeof(PRECISION)/2;
        for (int byte_index=0; byte_index<num_bytes_to_swap; byte_index++)
        {
            int swap_index = sizeof(PRECISION)-1-byte_index;
            unsigned char temp = pixel_bytes[byte_index];
            pixel_bytes[byte_index] = pixel_bytes[swap_index];
            pixel_bytes[swap_index] = temp;
        }
        num_data_chars += fwrite(pixel_bytes, sizeof(unsigned char), sizeof(PRECISION), fits_file);
    }
    // pad the primary data array with 0.0 values so that it is exactly a multiple of 2880 chars
    size_t num_data_blocks = (size_t)ceil(1.0*num_data_chars/2880);
    size_t num_padding_chars = num_data_blocks*2880 - num_data_chars;
    if (num_padding_chars > 0)
    {
        char padded_data[num_padding_chars];
        memset(padded_data, 0, sizeof(padded_data));
        num_data_chars += fwrite(padded_data, sizeof(char), num_padding_chars, fits_file);
    }

    return num_data_chars;
}

// AG hack function
void save_image_to_fits_file(Config *config, PRECISION *image, const char *file_name)
{
    // values that need to be specified that are specific to the image which are available in 
    const int image_width  = config->image_size;
    const int image_height = config->image_size;
    const double pixel_size = 1e-5;  //config->pixel_size; // (almost fov/imagesize) in radians
    const bool right_ascension = false; //config->right_ascension; 
    const double frequency_hz = 148000000.0; // Andrew: in SEP this is config->frequency_hz

    // Andrew: additional values which are not available in the SEP config
    const double coord_ref_width = 0.0; // coordinates of the reference (central) point
    const double coord_ref_height= -27.0; // coordinates of the reference (central) point
    const double celestial_pole_long_deg = 180.0; // native longitude of celestial pole in degrees
    const double celestial_pole_lat_deg = -27.0; // native latitude of celestial pole in degrees

    // Andrew: this FILE is named f in save_image_to_file in controller.cu
    //const char *buffer = "testdata.fits"; // Andrew: in SEP instead use the calculated buffer on lines 251-253
    FILE *fits_file; // Andrew: in SEP this is called f instead of fits_file
    fits_file = fopen(file_name, "wb");
    // generate some test pixel values in the correct precision
    ///const int numPixels = image_width*image_height;
    // PRECISION image[numPixels]; // Andrew: in SEP this is the image parameter on line 249
    // for (int i=0; i<numPixels; i++)
    // {
        // image[i] = ((PRECISION)i)/(numPixels-1);
    // }

    // Andrew: The following lines should appear in the save_image_to_file function in controller.cu
    // ***********************************************
    // assemble and write the FITS header to the file
    char *fits_header_unit = create_fits_header_unit();
    set_fits_axes_headers(fits_header_unit, image_width, image_height, 1, 1); // only one stokes polarization product and one frequency
    // note right_ascension presumed to flip pixel_size in horizontal direction, and for single frequency just take frequency increment to be the frequency
    set_fits_axes_increments(fits_header_unit, pixel_size, right_ascension, frequency_hz);
    set_fits_world_coordinate_system(fits_header_unit, coord_ref_width, coord_ref_height,
         frequency_hz, celestial_pole_long_deg, celestial_pole_lat_deg);
    set_fits_date_reference(fits_header_unit, NULL); // Note RASCIL FITS header uses "1858-11-17"
    set_header_text(fits_header_unit, "HISTORY", "Some history");
    set_header_text(fits_header_unit, "HISTORY", "Some more history");
    set_header_text(fits_header_unit, "COMMENT", "Created using the SEP Imaging and Calibration Pipeline");
    size_t saved = output_fits_header_unit(fits_header_unit, fits_file);
    // output the image pixels row by row to the FITS file
    saved += output_fits_data_unit(image, image_width, image_height, 1, 1, fits_file);
    // ***********************************************

    fclose(fits_file);
    printf("Wrote %ld chars\n", saved);
}
