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

#ifndef FITS_IMAGE_WRITER_H_
#define FITS_IMAGE_WRITER_H_ 

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

        int set_header_string_value(char *fits_header_unit, const char *keyword, const char *value, const char *comment);

        int set_header_quoted_value(char *fits_header_unit, const char *keyword, const char *value, const char *comment);

        int set_header_double_value(const char *fits_header_unit, const char *keyword, double value, const char *comment);

        int set_header_int_value(const char *fits_header_unit, const char *keyword, int value, const char *comment);

        int set_header_text(const char *fits_header_unit, const char *keyword, const char *text);

        char *create_fits_header_unit();

        void set_fits_axes_headers(char *fits_header_unit, int image_width, int image_height,
            int num_stokes, int num_freq);

        void set_fits_axes_increments(char *fits_header_unit, double pixel_size, bool right_ascension, double frequency_inc_hz);

        void set_fits_world_coordinate_system(char *fits_header_unit, double coord_ref_width, double coord_ref_height,
            double frequency_hz, double celestial_pole_long_deg, double celestial_pole_lat_deg);

        void set_fits_date_reference(char *fits_header_unit, char *date_reference);

        size_t output_fits_header_unit(char *fits_header_unit, FILE *fits_file);

        size_t output_fits_data_unit(PRECISION *image, int image_width, int image_height, int num_stokes, int num_freq, FILE *fits_file);
		
		void save_image_to_fits_file(Config *config, PRECISION *image, const char *file_name);

#ifdef __cplusplus
}
#endif

#endif /* FITS_IMAGE_WRITER_H_ */

