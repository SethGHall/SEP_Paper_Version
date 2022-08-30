
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

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
 
#ifndef DECONVOLUTION_H_
#define DECONVOLUTION_H_ 

	#include "common.h"
	#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

	void deconvolution_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings);

	void deconvolution_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void deconvolution_run(Config *config, Device_Mem_Handles *device);

	void deconvolution_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void deconvolution_clean_up(Device_Mem_Handles *device);

	void copy_psf_to_device(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void allocate_device_maximums(Config *config, Device_Mem_Handles *device);

	void copy_sources_to_host(Host_Mem_Handles *host, Device_Mem_Handles *device, unsigned int sources_found);

	void copy_sources_to_device(Host_Mem_Handles *host, Device_Mem_Handles *device, unsigned int sources_found);

	__global__ void grid_to_image_coords_conversion(PRECISION3 *sources, const PRECISION cell_size, const int half_grid_size,
		const int source_count);

	__global__ void image_to_grid_coords_conversion(PRECISION3 *sources, const PRECISION cell_size, const int half_grid_size,
	const int source_count);

	__global__ void scale_dirty_image_by_psf(PRECISION *image, PRECISION *psf, PRECISION reciprocal, const int grid_size);

	__global__ void find_max_source_row_reduction(const PRECISION *image, PRECISION3 *local_max,
		const int image_dim, const int search_dim);

	__global__ void find_max_source_col_reduction(PRECISION3 *sources, const PRECISION3 *local_max, const int cycle_number,
		const int image_dim, const int search_dim, const double loop_gain, const double weak_source_percent,
		const double noise_factor);

	__global__ void compress_sources(PRECISION3 *sources);

	__global__ void subtract_psf_from_image(PRECISION *image, PRECISION3 *sources, const PRECISION *psf, 
		const int cycle_number, const int grid_size, const PRECISION loop_gain);

	void save_sources_to_file(Source *sources, int number_of_sources, const char *path, const char *output_file, int cycle);

	void point_spread_function_free(Device_Mem_Handles *device);

	void local_maximums_free(Device_Mem_Handles *device);

#ifdef __cplusplus
}
#endif 
 
#endif /* DECONVOLUTION_H_ */
