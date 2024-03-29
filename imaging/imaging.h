
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

#ifndef IMAGING_H_
#define IMAGING_H_  

#include "common.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

int execute_imaging_pipeline(Config *config, Host_Mem_Handles *host_mem);

int execute_imaging_pipeline_msmfs(Config *config, Host_Mem_Handles *host_mem);

void save_taylor_term_planes(Config *config, PRECISION *host_image, int taylor_terms, int dim);

void generate_psf(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timers);

void save_psf_to_file(Config *config, PRECISION *image, const char *file_name, int start_x, int start_y, int range_x, int range_y);

void init_gpu_mem(Device_Mem_Handles *device);

void init_timers(Timing *timers);

void report_timings(Timing *timers);

void clean_up_device(Device_Mem_Handles *device);

void extract_extracted_sources(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

void extract_predicted_visibilities(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

void extract_pipeline_image(PRECISION *host_image, PRECISION *device_image, const int grid_size);

void extract_pipeline_gains(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

void normalize_image_for_weighting(const int grid_size, PRECISION *dirty_image, PRECISION weighted_sum);

#ifdef __cplusplus
}
#endif 

#endif /* IMAGING_H_ */
