
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

#ifndef GRIDDER_H_
#define GRIDDER_H_ 

#include "common.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

	/**
	 * Sets up the required device memory needed to perform convolutional gridding. This includes such items, but not limited
	 * to the grid, kernels, visibilities, visibility uvw coordinates, etc. 
	 * 
	 * <em>Note that if using high-spec GPU mode, this function is still called, 
	 * but device memory allocation and memory copy will be skipped.</em>
	 * 
	 * @param config Pointer to a struct containing configuration values for the current pipeline execution
	 * @param host Pointer to a struct containing pointers to allocated host memory
	 * @param device Pointer to a struct containing pointers to allocated device memory
	 * 
	 */
	void gridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void degridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void gridding_run(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void degridding_run(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void psf_normalization(Config *config, Device_Mem_Handles *device);

	void fft_run(Config *config, Device_Mem_Handles *device);
	
	void fft_run_degridding(Config *config, Device_Mem_Handles *device);

	void convolution_correction_run(Config *config, Device_Mem_Handles *device);

	void gridding_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void psf_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void degridding_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void gridding_clean_up(Device_Mem_Handles *device);
	
	void degridding_clean_up(Device_Mem_Handles *device);

	void gridding_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *memory, Timing *timings);
	
	void degridding_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *memory, Timing *timings);
	
	void psf_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	bool copy_kernels_to_device(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *memory);

	bool copy_visibilities_to_device(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void source_list_to_image_run(Config *config,Device_Mem_Handles *device);

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
	__global__ void gridding(PRECISION2 *grid, const VIS_PRECISION2 *kernel, const int2 *supports,
		const PRECISION3 *vis_uvw, const VIS_PRECISION2 *vis, const int num_vis, const int oversampling,
		const int grid_size, const double uv_scale, const double w_scale, const int num_w_kernels,
		bool psf, const VIS_PRECISION *vis_weights, const int num_channels, const int num_baselines, const PRECISION freq, const PRECISION precInc);
		
	__global__ void degridding(const PRECISION2 *grid, const VIS_PRECISION2 *kernel, const int2 *supports,
		const PRECISION3 *vis_uvw, VIS_PRECISION2 *vis, const int num_vis, const int oversampling,
		const int grid_size, const double uv_scale, const double w_scale, const int num_w_kernels,
		const int num_channels, const int num_baselines, const PRECISION freq, const PRECISION precInc);


	__global__ void find_psf_max(PRECISION *max_psf, const PRECISION *psf, const int image_size);
	
	__global__ void psf_normalization_kernel(PRECISION max_psf, PRECISION *psf, const int image_size);

	__device__ VIS_PRECISION2 complex_mult(const VIS_PRECISION2 z1, const VIS_PRECISION2 z2);

	__global__ void fft_shift_complex_to_complex(PRECISION2 *grid, const int width);
	
	__global__ void fft_shift_complex_to_real(PRECISION2 *grid, PRECISION *image, const int grid_dim, const int image_dim);

	__global__ void fft_shift_real_to_complex(PRECISION *image, PRECISION2 *grid, const int image_dim, const int grid_dim);

	__global__ void execute_convolution_correction(PRECISION *image, const PRECISION *prolate, const int image_size);

	__global__ void execute_weight_map_normalization(PRECISION *image, const PRECISION *weight_map, const int image_size);
	
	 __global__ void execute_sources_to_image(const PRECISION3 *sources, const int num_sources, 
                    PRECISION *image, const int image_size, const PRECISION cell_size);

	void clean_device_vis_weights(Device_Mem_Handles *device);

#ifdef __cplusplus
}
#endif

#endif /* GRIDDER_H_ */
