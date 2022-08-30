
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

#ifndef CONTROLLER_H_
#define CONTROLLER_H_ 

#include "gains.h"
#include <pthread.h>
#include <unistd.h> 
#include <stdlib.h>
#include <cstdlib>
#include <malloc.h>
#include <stdio.h>

#include "common.h"
#include "complex.h"
#include "fft.h"
#include "wprojection.h"
#include "gridder.h"
#include "deconvolution.h"
#include "direct_fourier_transform.h"
#include "memory.h"
#include "timer.h"
#include "imaging.h"
#include "weighting.h"
#include "nifty.h"
// #include "system_test.h"

#include <libfyaml.h> 

#ifdef __cplusplus
extern "C" {
#endif

	#define NUM_MEM_REGIONS 1
	#define NUM_IMAGING_ITERATIONS 1
	#define SLEEP_READER_MICRO 100
	#define SLEEP_PIPELINE_MICRO 500

	typedef struct MemoryRegions 
	{
		int num_visibilities;
		VIS_PRECISION2 *memory_region[NUM_MEM_REGIONS];
	} MemoryRegions;

	int execute_controller(Config *config);

	void pretend_imaging_pipeline(Host_Mem_Handles *host_mem, int cycle, int num_vis);

	void populate_memory_megion(VIS_PRECISION2 *region, int num_vis,  char* filename);
	
	void *load_vis_into_memory_regions(void *args);

	void clean_up(Host_Mem_Handles *host);

	int exit_and_host_clean(const char *message, Host_Mem_Handles *host_mem);

	void save_image_to_file(Config *config, PRECISION *image, const char *file_name, int cycle);
	
	bool init_config_from_file(Config *config, char* config_file, bool required);

	void init_config(Config *config);

	void init_mem(Host_Mem_Handles *host);

	bool kernel_host_setup(Config *config, Host_Mem_Handles *host);
	
	bool visibility_UVW_host_setup(Config *config, Host_Mem_Handles *host);

	bool visibility_intensity_host_setup(Config *config, Host_Mem_Handles *host);

	bool allocate_host_images(Config *config, Host_Mem_Handles *host);

	bool gains_host_set_up(Config *config, Host_Mem_Handles *host);

	void calculate_receiver_pairs(Config *config, int2 *receiver_pairs);

	bool correction_set_up(Config *config, Host_Mem_Handles *host);

	void create_1D_half_prolate(PRECISION *prolate, int grid_size);

	void save_predicted_visibilities(Config *config, Host_Mem_Handles *host, const int cycle);

	void save_extracted_sources(Source *sources, int number_of_sources, const char *path, const char *identifier, const char *output_file, int cycle);

	bool parse_config_attribute(struct fy_document *fyd, const char *attrib_name, const char *format, void* data, bool required);

	bool parse_config_attribute_bool(struct fy_document *fyd, const char *attrib_name, const char *format, bool* data, bool required);

	void update_calculated_configs(Config *config);

#ifdef __cplusplus
}
#endif

#endif /* CONTROLLER_H_ */
