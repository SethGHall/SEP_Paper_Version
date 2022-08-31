
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted pvided that the following conditions are met:

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
 	
#include "../gridder.h"

void gridding_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{
	// Set up - allocate device mem, transfer host mem to device
	gridding_set_up(config, host, device);

	// Perform gridding
	start_timer(&timings->solver, false);
	gridding_run(config, host, device);
	// Perform Inverse FFT
	fft_run(config, device);
	// Perform Convolution Correction
	convolution_correction_run(config, device);
	stop_timer(&timings->solver, false);	
	// Reset UV-plane for next major cycle
	if(!config->retain_device_mem)
	{
		// Transfer device mem back to host (only required data, and for retain data flag)
		gridding_memory_transfer(config, host, device);

		// Clean up device
		gridding_clean_up(device);
	}
}

void degridding_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{
	// Set up - allocate device mem, transfer host mem to device
	degridding_set_up(config, host, device);

	source_list_to_image_run(config,device);
	// Perform gridding
	start_timer(&timings->predict, false);	
	// Perform Convolution Correction
	convolution_correction_run(config, device);
	// Perform Inverse FFT
	fft_run_degridding(config, device);
	degridding_run(config, host, device);
	stop_timer(&timings->predict, false);	
	// Reset UV-plane for next major cycle
	
	CUDA_CHECK_RETURN(cudaMemset(device->d_uv_grid, 0, config->grid_size * config->grid_size * sizeof(PRECISION2)));
	cudaDeviceSynchronize();

	if(!config->retain_device_mem)
	{
		degridding_clean_up(device);
	}
}

void source_list_to_image_run(Config *config,Device_Mem_Handles *device)
{
	
//RUN GPU KERNEL
    int max_threads_per_block_conversion = min(config->gpu_max_threads_per_block, config->num_sources);
    int num_blocks_conversion = (int) ceil((double) config->num_sources / max_threads_per_block_conversion);
    dim3 conversion_blocks(num_blocks_conversion, 1, 1);
    dim3 conversion_threads(config->gpu_max_threads_per_block, 1, 1);
    execute_sources_to_image<<<conversion_blocks, conversion_threads>>>(device->d_sources, config->num_sources,
                                                device->d_image, config->image_size, config->cell_size_rad);
	cudaDeviceSynchronize();
}

void degridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	if(device->d_image != NULL)
    {  
        CUDA_CHECK_RETURN(cudaFree((*device).d_image));
    }

    //reset image
    int image_square = config->image_size * config->image_size;
    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_image), sizeof(PRECISION) * image_square));
    CUDA_CHECK_RETURN(cudaMemset(device->d_image, 0, image_square * sizeof(PRECISION)));
    cudaDeviceSynchronize();

    if(device->d_sources == NULL)
    {	
    	printf("UPDATE >>> Allocating Device Source Buffer of size %d...\n\n", config->num_sources);
        CUDA_CHECK_RETURN(cudaMalloc(&(device->d_sources), sizeof(PRECISION3) * config->num_sources));
        if(config->num_sources > 0) // occurs only if has sources from previous major cycle
            CUDA_CHECK_RETURN(cudaMemcpy(device->d_sources, host->h_sources, sizeof(Source) * config->num_sources, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }
	if(device->d_visibilities == NULL)
	{
	    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_visibilities), sizeof(VIS_PRECISION2) * config->vis_batch_size));
	    cudaDeviceSynchronize();
	}

	// Visibility UVW coordinates
	if(device->d_vis_uvw_coords == NULL)
	{
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_uvw_coords), sizeof(PRECISION3) * config->uvw_batch_size));
    	cudaDeviceSynchronize();
	}
	
	if(device->d_prolate == NULL)
    {   
    	printf("UPDATE >>> Allocating the prolate..\n\n");
        CUDA_CHECK_RETURN(cudaMalloc(&(device->d_prolate), sizeof(PRECISION) * (config->grid_size/2 + 1)));
	    CUDA_CHECK_RETURN(cudaMemcpy(device->d_prolate, host->prolate, sizeof(PRECISION) * config->grid_size / 2, cudaMemcpyHostToDevice));
    	cudaDeviceSynchronize();
    }

	int grid_square = config->grid_size * config->grid_size;
	if(device->d_uv_grid == NULL)
	{
	    printf("UPDATE >>> Allocating new UV-grid device memory of size %d squared complex values...\n\n", config->grid_size);
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_uv_grid), sizeof(PRECISION2) * grid_square));
	}

	CUDA_CHECK_RETURN(cudaMemset(device->d_uv_grid, 0, grid_square * sizeof(PRECISION2)));
	
	if(device->d_kernels == NULL)
	{
		printf("UPDATE >>> Copying kernels to device, number of samples: %d...\n\n",config->total_kernel_samples);
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_kernels), sizeof(VIS_PRECISION2) * config->total_kernel_samples));
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_kernels, host->kernels, sizeof(VIS_PRECISION2) * config->total_kernel_samples,
			cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	}

	if(device->d_kernel_supports == NULL)
	{
		printf("UPDATE >>> Copying kernels supports to device, number of planes: %d...\n\n",config->num_kernels);
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_kernel_supports), sizeof(int2) * config->num_kernels));
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_kernel_supports, host->kernel_supports, sizeof(int2) * config->num_kernels,
			cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	}

	if(device->fft_plan == NULL)
	{
		printf("UPDATE >>> Setup FFT plan and allocate output device image memory...\n\n");
		device->fft_plan = (cufftHandle*) calloc(1, sizeof(cufftHandle));
		CUFFT_SAFE_CALL(cufftPlan2d(device->fft_plan, config->grid_size, config->grid_size, CUFFT_C2C_PLAN));
	}
}

void psf_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	printf("UPDATE >>> Executing Gridding Pipeline for PSF generation...\n\n");
	gridding_set_up(config, host, device);

	gridding_run(config, host, device);
	fft_run(config, device);
	convolution_correction_run(config, device);

	psf_normalization(config, device);
	printf("UPDATE >>> PSF generation Complete...\n\n");
	// Reset UV-plane for next major cycle
	CUDA_CHECK_RETURN(cudaMemset(device->d_uv_grid, 0, config->grid_size * config->grid_size * sizeof(PRECISION2)));
	cudaDeviceSynchronize();

	// No longer require weights in subsequent imaging
	clean_device_vis_weights(device);

	if(!config->retain_device_mem)
	{
		// Transfer device mem back to host (only required data, and for retain data flag)
		psf_memory_transfer(config, host, device);

		// Clean up device
		gridding_clean_up(device);
	}
}





void clean_device_vis_weights(Device_Mem_Handles *device)
{
    if(device->d_vis_weights != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_vis_weights));
    device->d_vis_weights = NULL;
}

void gridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	// UV-Plane
	int grid_square = config->grid_size * config->grid_size;
	if(device->d_uv_grid == NULL)
	{
	    printf("UPDATE >>> Allocating new UV-grid device memory of size %d squared complex values...\n\n", config->grid_size);
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_uv_grid), sizeof(PRECISION2) * grid_square));	
	}
	CUDA_CHECK_RETURN(cudaMemset(device->d_uv_grid, 0, grid_square * sizeof(PRECISION2)));
	// Convolution Kernels
	if(device->d_kernels == NULL)
	{
		printf("UPDATE >>> Copying kernels to device, number of samples: %d...\n\n",config->total_kernel_samples);
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_kernels), sizeof(VIS_PRECISION2) * config->total_kernel_samples));
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_kernels, host->kernels, sizeof(VIS_PRECISION2) * config->total_kernel_samples,
			cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	}

	if(device->d_kernel_supports == NULL)
	{
		printf("UPDATE >>> Copying kernels supports to device, number of planes: %d...\n\n",config->num_kernels);
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_kernel_supports), sizeof(int2) * config->num_kernels));
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_kernel_supports, host->kernel_supports, sizeof(int2) * config->num_kernels,
			cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	}

	// Visibilities
	if(device->d_visibilities == NULL)
	{
	    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_visibilities), sizeof(VIS_PRECISION2) * config->vis_batch_size));
	    cudaDeviceSynchronize();
	}

	// Visibility UVW coordinates
	if(device->d_vis_uvw_coords == NULL)
	{
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_uvw_coords), sizeof(PRECISION3) * config->uvw_batch_size));
    	cudaDeviceSynchronize();
	}

	// Image
	if(!config->enable_psf && device->d_image == NULL)
	{
	    int image_square = config->image_size * config->image_size;
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_image), sizeof(PRECISION) * image_square));
    	CUDA_CHECK_RETURN(cudaMemset(device->d_image, 0, image_square * sizeof(PRECISION)));
    	cudaDeviceSynchronize();
	}

	if(device->fft_plan == NULL)
	{
		printf("UPDATE >>> Setup FFT plan and allocate output device image memory...\n\n");
		device->fft_plan = (cufftHandle*) calloc(1, sizeof(cufftHandle));
		CUFFT_SAFE_CALL(cufftPlan2d(device->fft_plan, config->grid_size, config->grid_size, CUFFT_C2C_PLAN));
	}

	// Prolate Spheroidal
	if(device->d_prolate == NULL)
	{
	    // Bind prolate spheroidal to gpu
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_prolate), sizeof(PRECISION) * config->grid_size / 2));
    	CUDA_CHECK_RETURN(cudaMemcpy(device->d_prolate, host->prolate, sizeof(PRECISION) * config->grid_size / 2, cudaMemcpyHostToDevice));
    	cudaDeviceSynchronize();
	}
	
	if(config->enable_psf && device->d_psf == NULL)
	{
	    // Bind prolate spheroidal to gpu
    	int image_square = config->image_size * config->image_size;
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_psf), sizeof(PRECISION) * image_square));
    	CUDA_CHECK_RETURN(cudaMemset(device->d_psf, 0, image_square * sizeof(PRECISION)));
    	cudaDeviceSynchronize();
	}

	if(device->d_vis_weights == NULL)
	{
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_weights), sizeof(VIS_PRECISION) * config->vis_batch_size));
    	//CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_weights, host->vis_weights,
		//	sizeof(VIS_PRECISION) * config->vis_batch_size, cudaMemcpyHostToDevice));
    	cudaDeviceSynchronize();
	}
}


void gridding_run(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	int total_num_batches = (int)CEIL(double(config->num_timesteps) / double(config->timesteps_per_batch));
		
	int visLeftToProcess = config->num_host_visibilities;
	int uwvLeftToProcess = config->num_host_uvws;
	
	PRECISION freq_inc = 0.0;
	if(config->number_frequency_channels > 1)
		freq_inc = PRECISION(config->frequency_bandwidth) / (config->number_frequency_channels-1); 
	
	for(int ts_batch=0;ts_batch<total_num_batches; ts_batch++)
	{	//copy vis UVW here
		//fill UVW and VIS
		printf("Griddding batch number %d ...\n",ts_batch);
		
		int current_vis_batch_size = min(visLeftToProcess, config->vis_batch_size);
		int current_uvw_batch_size = min(uwvLeftToProcess, config->uvw_batch_size);
		
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_weights, host->vis_weights+(ts_batch*config->vis_batch_size),
						sizeof(VIS_PRECISION) * current_vis_batch_size, cudaMemcpyHostToDevice));
		
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_visibilities, host->visibilities+(ts_batch*config->vis_batch_size), 
						sizeof(VIS_PRECISION2) * current_vis_batch_size, cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_uvw_coords, host->vis_uvw_coords+(ts_batch*config->uvw_batch_size), 
						sizeof(PRECISION3) * current_uvw_batch_size, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
		
		
		int max_threads_per_block = min(config->gpu_max_threads_per_block, current_vis_batch_size);
		int num_blocks = (int) ceil((double) current_vis_batch_size / max_threads_per_block);
		dim3 kernel_blocks(num_blocks, 1, 1);
		dim3 kernel_threads(max_threads_per_block, 1, 1);
				
		gridding<<<kernel_blocks, kernel_threads>>>(device->d_uv_grid, device->d_kernels, device->d_kernel_supports, 
					device->d_vis_uvw_coords, device->d_visibilities, current_vis_batch_size, config->oversampling,
					config->grid_size, config->uv_scale, config->w_scale, config->num_kernels, config->enable_psf,  
					device->d_vis_weights, config->number_frequency_channels, config->num_baselines, config->frequency_hz_start, freq_inc);
		cudaDeviceSynchronize();
		
		
		visLeftToProcess -= config->vis_batch_size;
		uwvLeftToProcess -= config->uvw_batch_size;
	}
	CUDA_CHECK_RETURN(cudaFree(device->d_vis_weights));
		device->d_vis_weights = NULL;
	CUDA_CHECK_RETURN(cudaFree(device->d_visibilities));
		device->d_visibilities = NULL;
	CUDA_CHECK_RETURN(cudaFree(device->d_vis_uvw_coords));
		device->d_vis_uvw_coords = NULL;		
	
	printf("UPDATE >>> Gridding complete...\n\n");
}

void degridding_run(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	
	int total_num_batches = (int)CEIL(double(config->num_timesteps) / double(config->timesteps_per_batch));
		
	int visLeftToProcess = config->num_host_visibilities;
	int uwvLeftToProcess = config->num_host_uvws;
	
	PRECISION freq_inc = 0.0;
	if(config->number_frequency_channels > 1)
		freq_inc = PRECISION(config->frequency_bandwidth) / (config->number_frequency_channels-1); 
	
	for(int ts_batch=0;ts_batch<total_num_batches; ts_batch++)
	{	//copy vis UVW here
		//fill UVW and VIS
		printf("Degriddding with batch number %d ...\n",ts_batch);
		
		int current_vis_batch_size = min(visLeftToProcess, config->vis_batch_size);
		int current_uvw_batch_size = min(uwvLeftToProcess, config->uvw_batch_size);
		
		CUDA_CHECK_RETURN(cudaMemset(device->d_visibilities, 0, current_vis_batch_size * sizeof(VIS_PRECISION2)));

		CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_uvw_coords, host->vis_uvw_coords+(ts_batch*config->uvw_batch_size), 
						sizeof(PRECISION3) * current_uvw_batch_size, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
		
		
		int max_threads_per_block = min(config->gpu_max_threads_per_block, current_vis_batch_size);
		int num_blocks = (int) ceil((double) current_vis_batch_size / max_threads_per_block);
		dim3 kernel_blocks(num_blocks, 1, 1);
		dim3 kernel_threads(max_threads_per_block, 1, 1);
		

		degridding<<<kernel_blocks, kernel_threads>>>(device->d_uv_grid, device->d_kernels, device->d_kernel_supports, 
				device->d_vis_uvw_coords, device->d_visibilities, current_vis_batch_size, config->oversampling,
				config->grid_size, config->uv_scale, config->w_scale, config->num_kernels, 
				config->number_frequency_channels, config->num_baselines, config->frequency_hz_start, freq_inc);
		cudaDeviceSynchronize();
		
	
		CUDA_CHECK_RETURN(cudaMemcpy(host->visibilities+(ts_batch*config->vis_batch_size), device->d_visibilities, 
						sizeof(VIS_PRECISION2) * current_vis_batch_size, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
	
		visLeftToProcess -= config->vis_batch_size;
		uwvLeftToProcess -= config->uvw_batch_size;
	}
	
	
	CUDA_CHECK_RETURN(cudaFree(device->d_visibilities));
	device->d_visibilities = NULL;
	CUDA_CHECK_RETURN(cudaFree(device->d_vis_uvw_coords));
	device->d_vis_uvw_coords = NULL;		
	
	printf("UPDATE >>> Degridding complete...\n\n");
	
}

void psf_normalization(Config *config, Device_Mem_Handles *device)
{
	PRECISION* d_max_psf_found;
	PRECISION max_psf_found;
	
	CUDA_CHECK_RETURN(cudaMalloc(&d_max_psf_found, sizeof(PRECISION)));
	
	find_psf_max<<<1,1>>>(d_max_psf_found, device->d_psf, config->image_size);
	cudaDeviceSynchronize();
	
	cudaMemcpy(&max_psf_found, d_max_psf_found, sizeof(PRECISION), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("UPDATE >>> FOUND PSF MAX OF %f ",max_psf_found);
	cudaFree(d_max_psf_found);
	
	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->image_size);
	int num_blocks_per_dimension = (int) ceil((double) config->image_size / max_threads_per_block_dimension);
	dim3 blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);
	
	psf_normalization_kernel<<<blocks, threads>>>(max_psf_found, device->d_psf, config->image_size);
	cudaDeviceSynchronize();
	
	config->psf_max_value = max_psf_found;
}

   
__global__ void find_psf_max(PRECISION *max_psf, const PRECISION *psf, const int image_size)
{
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index >= 1)
		return;

	*max_psf = psf[image_size * (image_size/2) + (image_size/2)];
}

__global__ void psf_normalization_kernel(PRECISION max_psf, PRECISION *psf, const int image_size)
{
	const int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    const int col_index = threadIdx.x + blockDim.x * blockIdx.x;

    if(row_index >= image_size || col_index >= image_size)
    	return;

    psf[row_index * image_size + col_index] /= max_psf;
}


void fft_run(Config *config, Device_Mem_Handles *device)
{
	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->grid_size);
	int num_blocks_per_dimension = (int) ceil((double) config->grid_size / max_threads_per_block_dimension);
	dim3 c2c_shift_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 c2c_shift_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	printf("UPDATE >>> Shifting grid data for FFT...\n\n");
	// Perform 2D FFT shift
	fft_shift_complex_to_complex<<<c2c_shift_blocks, c2c_shift_threads>>>(device->d_uv_grid, config->grid_size);
	cudaDeviceSynchronize();

	printf("UPDATE >>> Performing iFFT...\n\n");
	CUFFT_SAFE_CALL(CUFFT_EXECUTE_C2C(*(device->fft_plan), device->d_uv_grid, device->d_uv_grid, CUFFT_INVERSE));
	cudaDeviceSynchronize();

	printf("UPDATE >>> Shifting grid data back and converting to output image...\n\n");

	max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->image_size);
	num_blocks_per_dimension = (int) ceil((double) config->image_size / max_threads_per_block_dimension);
	dim3 c2r_shift_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 c2r_shift_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	if(config->enable_psf)
		fft_shift_complex_to_real<<<c2r_shift_blocks, c2r_shift_threads>>>(
			device->d_uv_grid, device->d_psf, config->grid_size, config->image_size
		);
	else
		fft_shift_complex_to_real<<<c2r_shift_blocks, c2r_shift_threads>>>(
			device->d_uv_grid, device->d_image, config->grid_size, config->image_size
		);
	cudaDeviceSynchronize();

	printf("UPDATE >>> FFT COMPLETE...\n\n");
}

void fft_run_degridding(Config *config, Device_Mem_Handles *device)
{
	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->image_size);
	int num_blocks_per_dimension = (int) ceil((double) config->image_size / max_threads_per_block_dimension);
	dim3 r2c_shift_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 r2c_shift_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	printf("UPDATE >>> Shifting image data to grid data for FFT...\n\n");
	// Perform 2D FFT shift
	fft_shift_real_to_complex<<<r2c_shift_blocks, r2c_shift_threads>>>(
		device->d_image,
		device->d_uv_grid,
		config->image_size,
		config->grid_size
	);
	cudaDeviceSynchronize();

	printf("UPDATE >>> Performing FFT...\n\n");
	CUFFT_SAFE_CALL(CUFFT_EXECUTE_C2C(*(device->fft_plan), device->d_uv_grid, device->d_uv_grid, CUFFT_FORWARD));
	cudaDeviceSynchronize();

	max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->grid_size);
	num_blocks_per_dimension = (int) ceil((double) config->grid_size / max_threads_per_block_dimension);
	dim3 c2c_shift_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 c2c_shift_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	printf("UPDATE >>> Shifting grid data back...\n\n");
	// Perform 2D FFT shift back
	fft_shift_complex_to_complex<<<c2c_shift_blocks, c2c_shift_threads>>>(device->d_uv_grid, config->grid_size);
	cudaDeviceSynchronize();
	printf("UPDATE >>> FFT COMPLETE...\n\n");
}


void convolution_correction_run(Config *config, Device_Mem_Handles *device)
{
	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->image_size);
	int num_blocks_per_dimension = (int) ceil((double) config->image_size / max_threads_per_block_dimension);
	dim3 cc_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 cc_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	if(config->enable_psf)
		execute_convolution_correction<<<cc_blocks, cc_threads>>>(device->d_psf, device->d_prolate, config->image_size);
	else
		execute_convolution_correction<<<cc_blocks, cc_threads>>>(device->d_image, device->d_prolate, config->image_size);
	cudaDeviceSynchronize();
}

void gridding_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
    int image_square = config->image_size * config->image_size;
    CUDA_CHECK_RETURN(cudaMemcpy(host->dirty_image, device->d_image, image_square * sizeof(PRECISION),
        cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}

void psf_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
    int image_square = config->image_size * config->image_size;
    CUDA_CHECK_RETURN(cudaMemcpy(host->h_psf, device->d_psf, image_square * sizeof(PRECISION),
        cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}

void gridding_clean_up(Device_Mem_Handles *device)
{
	printf("UPDATE >>> Freeing allocated device memory for Gridding / FFT / Convolution Correction...\n\n");

    if(device->d_uv_grid != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_uv_grid));
    device->d_uv_grid = NULL;

    if(device->d_kernel_supports != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_kernel_supports));
	device->d_kernel_supports = NULL;

	if(device->d_kernels != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_kernels));
	device->d_kernels = NULL;

	if(device->d_visibilities != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_visibilities));
	device->d_visibilities = NULL;

	if(device->d_vis_uvw_coords != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_vis_uvw_coords));
	device->d_vis_uvw_coords = NULL;	

	if(device->d_image != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_image));
	device->d_image = NULL;

	if(device->d_psf != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_psf));
	device->d_psf = NULL;

	if(device->fft_plan != NULL)
	{
		CUFFT_SAFE_CALL(cufftDestroy(*(device->fft_plan)));
		free(device->fft_plan);
	}
	device->fft_plan = NULL;

	if(device->d_prolate != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_prolate));
	device->d_prolate = NULL;
}

void degridding_clean_up(Device_Mem_Handles *device)
{
	printf("UPDATE >>> Freeing allocated device memory for DEGridding / FFT / Convolution Correction...\n\n");

	if(device->d_sources == NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_sources));
    device->d_sources = NULL;
	
    if(device->d_uv_grid != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_uv_grid));
    device->d_uv_grid = NULL;

    if(device->d_kernel_supports != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_kernel_supports));
	device->d_kernel_supports = NULL;

	if(device->d_kernels != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_kernels));
	device->d_kernels = NULL;

	if(device->d_image != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_image));
	device->d_image = NULL;

	if(device->fft_plan != NULL)
	{
		CUFFT_SAFE_CALL(cufftDestroy(*(device->fft_plan)));
		free(device->fft_plan);
	}
	device->fft_plan = NULL;

	if(device->d_prolate != NULL)
		CUDA_CHECK_RETURN(cudaFree(device->d_prolate));
	device->d_prolate = NULL;
}



//applys w-projection to the grid, snapping to nearest W plane and UV oversample
__global__ void gridding(PRECISION2 *grid, const VIS_PRECISION2 *kernel, const int2 *supports,
	const PRECISION3 *vis_uvw, const VIS_PRECISION2 *vis, const int num_vis, const int oversampling,
	const int grid_size, const double uv_scale, const double w_scale, const int num_w_kernels, 
	const bool psf, const VIS_PRECISION *vis_weights, const int num_channels, const int num_baselines, const PRECISION freq, const PRECISION freqInc)
{
	const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(vis_index >= num_vis)
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
	
	//convert local_UVW meters to wavelengths based on frequency
	
	// Represents index of w-projection kernel in supports array
	int w_kernel_index = (int) ROUND(SQRT(ABS(local_uvw.z * w_scale)));
	// Clamp to max possible kernel (prevent illegal memory exception)
	w_kernel_index = max(min(w_kernel_index, num_w_kernels - 1), 0);

	// Scale visibility uvw into grid coordinate space
	const PRECISION2 grid_coord = MAKE_PRECISION2(
		local_uvw.x * uv_scale,
		local_uvw.y * uv_scale
	);

	const int half_grid_size = grid_size / 2;
	const int half_support = supports[w_kernel_index].x;

	VIS_PRECISION conjugate = (local_uvw.z > 0.0) ? -1.0 : 1.0;

	const PRECISION2 snapped_grid_coord = MAKE_PRECISION2(
		ROUND(grid_coord.x * oversampling) / oversampling,
		ROUND(grid_coord.y * oversampling) / oversampling
	);

	const PRECISION2 min_grid_point = MAKE_PRECISION2(
		max(CEIL(snapped_grid_coord.x - half_support), (PRECISION)(-half_grid_size + 1)),
		max(CEIL(snapped_grid_coord.y - half_support), (PRECISION)(-half_grid_size + 1))
	);

	const PRECISION2 max_grid_point = MAKE_PRECISION2(
		min(FLOOR(snapped_grid_coord.x + half_support), (PRECISION)(half_grid_size - 1)),
		min(FLOOR(snapped_grid_coord.y + half_support), (PRECISION)(half_grid_size - 1))
	);

	PRECISION2 grid_point = MAKE_PRECISION2(0.0, 0.0);
	VIS_PRECISION2 convolved = MAKE_VIS_PRECISION2(0.0, 0.0);
	VIS_PRECISION2 kernel_sample = MAKE_VIS_PRECISION2(0.0, 0.0);
	int2 kernel_uv_index = make_int2(0, 0);

	int grid_index = 0;
	int kernel_index = 0;
	int w_kernel_offset = supports[w_kernel_index].y;
	
	VIS_PRECISION2 vis_value = 
		(!psf) ? vis[vis_index]: MAKE_VIS_PRECISION2((VIS_PRECISION) vis_weights[vis_index], 0.0);

	for(int grid_v = min_grid_point.y; grid_v <= max_grid_point.y; ++grid_v)
	{	
		kernel_uv_index.y = abs((int)ROUND((grid_v - snapped_grid_coord.y) * oversampling));
		
		for(int grid_u = min_grid_point.x; grid_u <= max_grid_point.x; ++grid_u)
		{
			kernel_uv_index.x = abs((int)ROUND((grid_u - snapped_grid_coord.x) * oversampling));

			kernel_index = w_kernel_offset + kernel_uv_index.y * (half_support + 1)
				* oversampling + kernel_uv_index.x;

			grid_index = (grid_v + half_grid_size) * grid_size + (grid_u + half_grid_size);

			kernel_sample = kernel[kernel_index];
			kernel_sample.y *= conjugate;

			convolved = complex_mult(vis_value, kernel_sample);

			atomicAdd(&(grid[grid_index].x), (PRECISION) convolved.x);
			atomicAdd(&(grid[grid_index].y), (PRECISION) convolved.y);
		}
	}
}

//applys w-projection to the grid, snapping to nearest W plane and UV oversample
__global__ void degridding(const PRECISION2 *grid, const VIS_PRECISION2 *kernel, const int2 *supports,
	const PRECISION3 *vis_uvw, VIS_PRECISION2 *vis, const int num_vis, const int oversampling,
	const int grid_size, const double uv_scale, const double w_scale, const int num_w_kernels,
	const int num_channels, const int num_baselines, const PRECISION freq, const PRECISION freqInc)
{
	const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(vis_index >= num_vis)
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

	// Represents index of w-projection kernel in supports array
	int w_kernel_index = (int) ROUND(SQRT(ABS(local_uvw.z * w_scale)));
	// Clamp to max possible kernel (prevent illegal memory exception)
	w_kernel_index = max(min(w_kernel_index, num_w_kernels - 1), 0);

	// Scale visibility uvw into grid coordinate space
	const PRECISION2 grid_coord = MAKE_PRECISION2(
		local_uvw.x * uv_scale,
		local_uvw.y * uv_scale
	);

	const int half_grid_size = grid_size / 2;
	const int half_support = supports[w_kernel_index].x;

	VIS_PRECISION conjugate = (local_uvw.z > 0.0) ? -1.0 : 1.0;
	
	//In stand alone degridder we had to flip the conjugate again is this correct???
	//conjugate *= (VIS_PRECISION)-1.0;

	const PRECISION2 snapped_grid_coord = MAKE_PRECISION2(
		ROUND(grid_coord.x * oversampling) / oversampling,
		ROUND(grid_coord.y * oversampling) / oversampling
	);

	const PRECISION2 min_grid_point = MAKE_PRECISION2(
		max(CEIL(snapped_grid_coord.x - half_support), (PRECISION)(-half_grid_size + 1)),
		max(CEIL(snapped_grid_coord.y - half_support), (PRECISION)(-half_grid_size + 1))
	);

	const PRECISION2 max_grid_point = MAKE_PRECISION2(
		min(FLOOR(snapped_grid_coord.x + half_support), (PRECISION)(half_grid_size - 1)),
		min(FLOOR(snapped_grid_coord.y + half_support), (PRECISION)(half_grid_size - 1))
	);

	VIS_PRECISION2 convolved = MAKE_VIS_PRECISION2(0.0, 0.0);
	VIS_PRECISION2 kernel_sample = MAKE_VIS_PRECISION2(0.0, 0.0);
	int2 kernel_uv_index = make_int2(0, 0);

	int grid_index = 0;
	int kernel_index = 0;
	int w_kernel_offset = supports[w_kernel_index].y;
	
	VIS_PRECISION2 vis_value = MAKE_VIS_PRECISION2(0.0, 0.0);
	VIS_PRECISION2 grid_point = MAKE_VIS_PRECISION2(0.0, 0.0);	

	for(int grid_v = min_grid_point.y; grid_v <= max_grid_point.y; ++grid_v)
	{	
		kernel_uv_index.y = abs((int)ROUND((grid_v - snapped_grid_coord.y) * oversampling));
		
		for(int grid_u = min_grid_point.x; grid_u <= max_grid_point.x; ++grid_u)
		{
			kernel_uv_index.x = abs((int)ROUND((grid_u - snapped_grid_coord.x) * oversampling));

			kernel_index = w_kernel_offset + kernel_uv_index.y * (half_support + 1)
				* oversampling + kernel_uv_index.x;

			grid_index = (grid_v + half_grid_size) * grid_size + (grid_u + half_grid_size);

			kernel_sample = kernel[kernel_index];
			kernel_sample.y *= conjugate;

			grid_point.x = (VIS_PRECISION)grid[grid_index].x;
			grid_point.y = (VIS_PRECISION)grid[grid_index].y;
			convolved = complex_mult(grid_point, kernel_sample);

			vis_value.x += convolved.x;
			vis_value.y += convolved.y;
		}
	}
	vis[vis_index] = MAKE_VIS_PRECISION2(vis_value.x,vis_value.y);
}



__global__ void fft_shift_complex_to_complex(PRECISION2 *grid, const int width)
{
    int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
    if(row_index >= width || col_index >= width)
        return;
 
    int a = 1 - 2 * ((row_index + col_index) & 1);
    grid[row_index * width + col_index].x *= a;
    grid[row_index * width + col_index].y *= a;
}
//Note passing in output image - Future work do Complex to Real transform
__global__ void fft_shift_complex_to_real(PRECISION2 *grid, PRECISION *image, const int grid_dim, const int image_dim)
{
    int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
    if(row_index >= image_dim || col_index >= image_dim)
        return;

    int grid_dim_offset = int((grid_dim - image_dim) * 0.5);
 	int grid_index = (row_index+grid_dim_offset) * grid_dim + col_index+grid_dim_offset;
 	int image_index = row_index * image_dim + col_index;
    int a = 1 - 2 * ((row_index + col_index) & 1);
    image[image_index] =  grid[grid_index].x * a;
}

__global__ void fft_shift_real_to_complex(PRECISION *image, PRECISION2 *grid, const int image_dim, const int grid_dim)
{
    int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
    if(row_index >= image_dim || col_index >= image_dim)
        return;

    int grid_dim_offset = int((grid_dim - image_dim) * 0.5);
 	int grid_index = (row_index+grid_dim_offset) * grid_dim + col_index+grid_dim_offset;
 	int image_index = row_index * image_dim + col_index;
    int a = 1 - 2 * ((row_index + col_index) & 1);
    grid[grid_index].x =  image[image_index] * a;
}

//execute convolution correction on GPU - use row and col indices to look up prolate sample and divide image
__global__ void execute_convolution_correction(PRECISION *image, const PRECISION *prolate, const int image_size)
{
	const int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    const int col_index = threadIdx.x + blockDim.x * blockIdx.x;

    if(row_index >= image_size || col_index >= image_size)
    	return;

    const int image_index = row_index * image_size + col_index;
    const int half_image_size = image_size / 2;
    const int taper_x_index = abs(col_index - half_image_size);
    const int taper_y_index = abs(row_index - half_image_size);

    // patch to fix out of bounds taper access (masking with zero for first row/col)
    // TODO: review a better technique to solve this, possibly increased taper width by one?
    const PRECISION taper = (taper_x_index >= half_image_size || taper_y_index >= half_image_size) 
                                ? 0.0 : prolate[taper_x_index] * prolate[taper_y_index];
    image[image_index] = (ABS(taper) > (1E-10)) ? image[image_index] / taper  : 0.0;
}

 __global__ void execute_sources_to_image(const PRECISION3 *sources, const int num_sources, 
                    PRECISION *image, const int image_size, const PRECISION cell_size)
{
    const int source_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(source_index >= num_sources)
        return;

    //Convert image coordinate (l,m) to image pixel coordinates
    PRECISION3 source = sources[source_index];
	source.x = ROUND((source.x / cell_size) + (image_size/2)); 
    source.y = ROUND((source.y / cell_size) + (image_size/2));
	
	int image_index = (source.y * image_size) + source.x;

    atomicAdd(&(image[image_index]), (PRECISION) source.z);
}   

__global__ void execute_weight_map_normalization(PRECISION *image, const PRECISION *weight_map, const int image_size)
{
	const int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    const int col_index = threadIdx.x + blockDim.x * blockIdx.x;

    if(row_index >= image_size || col_index >= image_size)
    	return;

	const int pixel_index = row_index * image_size + col_index;
	// Should we be setting image pixel to zero in this case?...
	image[pixel_index] = (weight_map[pixel_index] > 0.0) ? image[pixel_index] / weight_map[pixel_index] : 0.0;
}


__device__ VIS_PRECISION2 complex_mult(const VIS_PRECISION2 z1, const VIS_PRECISION2 z2)
{
    return MAKE_VIS_PRECISION2(z1.x * z2.x - z1.y * z2.y, z1.y * z2.x + z1.x * z2.y);
}
