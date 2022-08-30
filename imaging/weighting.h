
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
 
#ifndef WEIGHTING_H_
#define WEIGHTING_H_ 

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

    void visibility_weighting_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

    void generate_weight_sum_maps(Config *config, Device_Mem_Handles *device);

    void perform_weight_scaling(Config *config, Device_Mem_Handles *device);

    __global__ void parallel_scale_vis_weights(const PRECISION3 *vis_uvw,
        VIS_PRECISION2 *vis, VIS_PRECISION *vis_weights, PRECISION *weight_plane, 
        const int grid_size, const int num_vis, const enum weighting_scheme scheme,
        const double f, const double uv_scale);

    __global__ void weight_mapping(PRECISION *weight_plane, const PRECISION3 *vis_uvw,
        const VIS_PRECISION *vis_weights, const int num_vis, const int grid_size, 
        const double uv_scale);

    void visibility_weighting_setup(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

    void visibility_weighting_cleanup(Device_Mem_Handles *device);

    __global__ void sum(PRECISION *input, PRECISION *output, const int num_elements, bool squared);
	
	__global__ void sum_vis(VIS_PRECISION *input, PRECISION *output, const int num_elements, bool squared);
    
    void clean_device_weight_map(Device_Mem_Handles *device);

    void transfer_imaging_weights_to_host(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

    void transfer_scaled_visibilities_to_host(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

#ifdef __cplusplus
}
#endif

#endif /* WEIGHTING_H_ */
