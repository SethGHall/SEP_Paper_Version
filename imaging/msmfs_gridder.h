
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

#ifndef MSMFS_GRIDDER_H_
#define MSMFS_GRIDDER_H_ 

#include "common.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

	//used to run w-projection solver 
	void gridding_execute_msmfs(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings);
	
	//called to initialize device bound memory
	void msmfs_gridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void gridding_run_msmfs(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void pull_msmfs_image_cube(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void msmfs_gridding_clean_up(Device_Mem_Handles *device);
	
	__global__ void execute_convolution_correction_msmfs(PRECISION *image, const PRECISION *prolate, const int t_plane, const int image_size);
	
	__global__ void accumulate_taylor_term_plane(PRECISION *plane, PRECISION *image_cube, const int t_plane, const int image_dim);
	
	
	__global__ void gridding_msmfs(PRECISION2 *grid, const VIS_PRECISION2 *kernel, const int2 *supports,
									const PRECISION3 *vis_uvw, const VIS_PRECISION2 *vis, const int num_vis, const int oversampling,
									const int grid_size, const double uv_scale, const double w_scale, const int num_w_kernels, 
									const bool psf, const VIS_PRECISION *vis_weights, const int num_channels, const int num_baselines, 
									const PRECISION freq, const PRECISION freqInc, const PRECISION vref, const int taylor_term);

#ifdef __cplusplus
}
#endif

#endif /* MSMFS_GRIDDER_H_ */
