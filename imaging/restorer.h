
// Copyright 2021 Anthony Griffin, Adam Campbell, Seth Hall, Andrew Ensor
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

#ifndef RESTORER_H_
#define RESTORER_H_ 

#include "common.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

	/**
	 * Sets up the required device memory needed to perform image restoration. This includes such items, but not limited
	 * to the image, estimated beam, source list.. 
	 * 
	 * <em>Note that if using high-spec GPU mode, this function is still called, 
	 * but device memory allocation and memory copy will be skipped.</em>
	 * 
	 * @param config Pointer to a struct containing configuration values for the current pipeline execution
	 * @param host Pointer to a struct containing pointers to allocated host memory
	 * @param device Pointer to a struct containing pointers to allocated device memory
	 * 
	 */
	void restoring_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void restoring_run(Config *config, Device_Mem_Handles *device);
	
	void do_image_restoration(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void restoring_clean_up(Device_Mem_Handles *device);

	void restoring_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *memory, Timing *timings);
	
	/**
	 * CUDA kernel for performing the W-Projection based convolutional gridding of a single visibility.
	 * Each thread will execute a copy of this kernel, such that one thread will perform the gridding of one visibility.
	 * 
	 * @param grid Pointer to the uv-plane (the grid), stored as a flattened one-dimensional array where each element represents one complex grid point.
	 * @param kernel Pointer to the set of bound W-projection convolution kernels. The kernels are stacked from plane 0 to max plane, and are flattened into a one-dimensional array.
	 * @param supports Pointer to a one-dimensional array of W-Projection kernel metadata. Each element in the array is a two-element vector, where the first element describes the support of a convolution kernel, and the second element describes the starting index of the kernel in the flattened kernel array.
	 * @param vis_uvw Pointer to a one-dimensional array storing the uvw coordinates for the bound set of visibilities. Each PRECISION3 represents one visibilities uvw coordinates.
	 * @param vis Pointer to a one-dimensional array storing the bound set of visibilities. Each PRECISION2 represents one complex visibility.
	 * @param num_vis The number of visibilities being processed in this execution of the pipeline.
	 * @param oversampling The oversampling factor used during creation and application of W-Projection convolution kernels.
	 * @param grid_size The size of the uv-plane (or grid) in one dimension.
	 * @param uv_scale Scalar value used to map visibility uv coordinates from metres to grid coordinates.
	 * @param w_scale Scalar value used to map visibility w coordinate to one of the stored W-Projection convolution kernels.
	 */
	__global__ void convolve_source_with_beam(PRECISION *image, const PRECISION *beam, const int2 beam_support,
		const PRECISION3 *source_list_lmv, const int num_sources, 
		const int image_size);
		

#ifdef __cplusplus
}
#endif

#endif /* GRIDDER_H_ */
