
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
 	
#include "../direct_fourier_transform.h"

void dft_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{
	// Set up - allocate device mem, transfer host mem to device
	dft_set_up(config, host, device);

	start_timer(&timings->predict, false);
	dft_run(config, device, timings);
	stop_timer(&timings->predict, false);

	if(!config->retain_device_mem)
	{
		// Transfer device mem back to host (only required data, and for retain data flag)
		dft_memory_transfer(config, host, device);

		// Clean up device
		dft_clean_up(device);
	}
}

void dft_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	// Sources (major * minor)
	if(device->d_sources == NULL)
	{
		printf("UPDATE >>> Allocating Device Source Buffer of size %d...\n\n", config->num_sources);
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_sources), sizeof(PRECISION3) * config->num_sources));
    	if(config->num_sources > 0) // occurs only if has sources from previous major cycle
			CUDA_CHECK_RETURN(cudaMemcpy(device->d_sources, host->h_sources, sizeof(Source) * config->num_sources, cudaMemcpyHostToDevice));
    	cudaDeviceSynchronize();
	}

	// Predicted Vis
	if(device->d_visibilities == NULL)
	{	printf("UPDATE >>> Allocating VISIBILITY BUFFER %d...\n\n", config->num_host_visibilities);
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_visibilities), sizeof(VIS_PRECISION2) * config->num_host_visibilities));
		CUDA_CHECK_RETURN(cudaMemset(device->d_visibilities, 0, sizeof(VIS_PRECISION2) * config->num_host_visibilities));
	    cudaDeviceSynchronize();
	}

	// Visibility UVW coordinates
	if(device->d_vis_uvw_coords == NULL)
	{	printf("UPDATE >>> Allocating UVW BUFFER %d...\n\n", config->num_host_uvws);
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_uvw_coords), sizeof(PRECISION3) * config->num_host_uvws));
	    CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_uvw_coords, host->vis_uvw_coords, sizeof(PRECISION3) * config->num_host_uvws,
        	cudaMemcpyHostToDevice));
    	cudaDeviceSynchronize();
	}
}

void dft_run(Config *config, Device_Mem_Handles *device, Timing *timings)
{
	int max_threads_per_block = min(config->gpu_max_threads_per_block, config->num_visibilities);
	int num_blocks = (int) ceil((double) config->num_visibilities / max_threads_per_block);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);

	printf("UPDATE >>> Executing the Direct Fourier Transform algorithm...\n\n");
	printf("UPDATE >>> DFT distributed over %d blocks, consisting of %d threads...\n\n", num_blocks, max_threads_per_block);


	PRECISION freq_inc = 0.0;
	if(config->number_frequency_channels > 1)
		freq_inc = PRECISION(config->frequency_bandwidth) / (config->number_frequency_channels-1); 

	start_timer(&timings->dft, false);
	direct_fourier_transform<<<kernel_blocks, kernel_threads>>>
	(
		device->d_vis_uvw_coords,
		device->d_visibilities,
		config->num_visibilities,
		device->d_sources,
		config->num_sources,
		config->number_frequency_channels, 
		config->num_baselines, 
		config->frequency_hz_start, 
		freq_inc
	);
	cudaDeviceSynchronize();
	stop_timer(&timings->dft, false);

	printf("UPDATE >>> Direct Fourier Transform complete...\n\n");
}

void dft_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
    CUDA_CHECK_RETURN(cudaMemcpy(host->visibilities, device->d_visibilities, 
        config->num_visibilities * sizeof(VIS_PRECISION2), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

void dft_clean_up(Device_Mem_Handles *device)
{
	// Sources
  	if(device->d_sources != NULL) 
        CUDA_CHECK_RETURN(cudaFree(device->d_sources));
    device->d_sources = NULL;

	if(device->d_visibilities != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_visibilities));
	device->d_visibilities = NULL;

	if(device->d_vis_uvw_coords != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_vis_uvw_coords));
	device->d_vis_uvw_coords = NULL;	
}

//execute direct fourier transform on GPU
__global__ void direct_fourier_transform(const PRECISION3 *vis_uvw, VIS_PRECISION2 *predicted_vis,
	const int vis_count, const PRECISION3 *sources, const int source_count,
	const int num_channels, const int num_baselines, const PRECISION freq, const PRECISION freqInc)
{
	const int vis_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(vis_index >= vis_count)
		return;


	int timeStepOffset = vis_index/(num_channels*num_baselines);
	int uvwIndex = (vis_index % num_baselines) + (timeStepOffset*num_baselines);
	PRECISION3 local_uvw = vis_uvw[uvwIndex];
	
	//chnnelNum - visIndex 
	int channelNumber = (vis_index / num_baselines) % num_channels; 
	
	PRECISION freqScale = (freq + channelNumber*freqInc) / PRECISION(SPEED_OF_LIGHT); //meters to wavelengths conversion
	
	local_uvw.x *= freqScale;
	local_uvw.y *= freqScale;
	local_uvw.z *= freqScale;



	const PRECISION two_PI = PI + PI;
	
	PRECISION3 src;
	PRECISION2 theta_complex = MAKE_PRECISION2(0.0, 0.0);
	PRECISION2 source_sum = MAKE_PRECISION2(0.0, 0.0);

	// For all sources
	for(int src_indx = 0; src_indx < source_count; ++src_indx)
	{	
		src = sources[src_indx];
		
		#if APPROXIMATE_DFT
			// approximation formula (faster but less accurate)
			PRECISION term = 0.5 * ((src.x * src.x) + (src.y * src.y));
			PRECISION w_correction = -term;
			PRECISION image_correction = 1.0 - term;
		#else
			// square root formula (most accurate method)
			PRECISION term = SQRT(1.0 - (src.x * src.x) - (src.y * src.y));
			PRECISION image_correction = term;
			PRECISION w_correction = term - 1.0;
		#endif

		PRECISION src_correction = src.z / image_correction;
		PRECISION theta = (local_uvw.x * src.x + local_uvw.y * src.y + local_uvw.z * w_correction) * two_PI;
		SINCOS(theta, &(theta_complex.y), &(theta_complex.x));
		source_sum.x += theta_complex.x * src_correction;
		source_sum.y += -theta_complex.y * src_correction;
	}
	predicted_vis[vis_index] = MAKE_VIS_PRECISION2((VIS_PRECISION) source_sum.x, (VIS_PRECISION) source_sum.y);
}