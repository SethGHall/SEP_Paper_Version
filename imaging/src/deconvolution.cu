
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
 	
#include "../deconvolution.h"

__device__ bool d_exit_early = false;
__device__ unsigned int d_source_counter = 0;

void deconvolution_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{
	// Set up - allocate device mem, transfer host mem to device
	deconvolution_set_up(config, host, device);

	start_timer(&timings->deconvolution, false);
	deconvolution_run(config, device);
	stop_timer(&timings->deconvolution, false);

	if(!config->retain_device_mem)
	{
		// Transfer device mem back to host (only required data, and for retain data flag)
		deconvolution_memory_transfer(config, host, device);

		// Clean up device
		deconvolution_clean_up(device);
	}
}

void deconvolution_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	// Image
	if(device->d_image == NULL)
	{
	    int image_square = config->image_size * config->image_size;
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_image), sizeof(PRECISION) * image_square));
    	CUDA_CHECK_RETURN(cudaMemcpy(device->d_image, host->dirty_image, 
	        image_square * sizeof(PRECISION), cudaMemcpyHostToDevice));
    	cudaDeviceSynchronize();
	}

	// PSF
	if(device->d_psf == NULL)
	{	
		int psf_size_square = config->image_size * config->image_size;
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_psf), sizeof(PRECISION) * psf_size_square));
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_psf, host->h_psf,
			sizeof(PRECISION) * psf_size_square, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	}

	// Sources (major * minor)
	if(device->d_sources == NULL)
	{
		int number_minor_cycles = (config->perform_gain_calibration) ? config->number_minor_cycles_cal : config->number_minor_cycles_img;
		printf("UPDATE >>> Allocating Device Source Buffer of size %d...\n\n", number_minor_cycles * config->num_major_cycles);
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_sources), sizeof(PRECISION3) * number_minor_cycles* config->num_major_cycles));
    	CUDA_CHECK_RETURN(cudaMemset(device->d_sources, 0, sizeof(PRECISION3) * number_minor_cycles* config->num_major_cycles));
    	if(config->num_sources > 0) // occurs only if has sources from previous major cycle
		{	
			CUDA_CHECK_RETURN(cudaMemcpy(device->d_sources, host->h_sources, sizeof(Source) * config->num_sources, cudaMemcpyHostToDevice));
    	}
    	cudaDeviceSynchronize();
	}
}

void deconvolution_run(Config *config, Device_Mem_Handles *device)
{
	int image_search_dim = (int)ceil((config->image_size * config->search_region_percent) / 100.0);
	//To ensure image_search_dim is divisible by two because also need to have a "half image_search_dim"
	if(image_search_dim % 2 != 0)
		image_search_dim -= 1;

	printf("INFO >>> Performing deconvolution in inner %f%% (inner %d pixels of actual %d pixels)...\n\n",
		config->search_region_percent, image_search_dim, config->image_size);

	PRECISION3 *max_locals = NULL;
	CUDA_CHECK_RETURN(cudaMalloc(&max_locals, sizeof(PRECISION3) * image_search_dim));
	CUDA_CHECK_RETURN(cudaMemset(max_locals, 0, sizeof(PRECISION3) * image_search_dim));

	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->image_size);
	int num_blocks_per_dimension = (int) ceil((double) config->image_size / max_threads_per_block_dimension);
	dim3 scale_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 scale_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	printf(">>> PSF MAX VALUE IS %lf...\n\n", config->psf_max_value);

	scale_dirty_image_by_psf<<<scale_blocks, scale_threads>>>(device->d_image, device->d_psf, config->psf_max_value, config->image_size);
	cudaDeviceSynchronize();

	// row reduction configuration
	int max_threads_per_block = min(config->gpu_max_threads_per_block, image_search_dim);
	int num_blocks = (int) ceil((double) image_search_dim / max_threads_per_block);
	dim3 reduction_blocks(num_blocks, 1, 1);
	dim3 reduction_threads(max_threads_per_block, 1, 1);

	// PSF subtraction configuration
	int max_psf_threads_per_block_dim = min(config->gpu_max_threads_per_block_dimension, config->image_size);
	int num_blocks_psf = (int) ceil((double) config->image_size / max_psf_threads_per_block_dim);
	dim3 psf_blocks(num_blocks_psf, num_blocks_psf, 1);
	dim3 psf_threads(max_psf_threads_per_block_dim, max_psf_threads_per_block_dim, 1);

	unsigned int cycle_number = 0;
	bool exit_early = false;

	if(config->perform_gain_calibration)
		config->num_sources = 0;
	
	// Reset exit early clause in case of multiple major cycles
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_exit_early, &exit_early, sizeof(bool), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_source_counter, &(config->num_sources), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	//convert existing sources to grid coords
	if(config->num_sources > 0)
	{	
		printf("UPDATE >>> Performing grid conversion on previously found Source coordinates...\n\n");

		int max_threads_per_block_conversion = min(config->gpu_max_threads_per_block, config->num_sources);
		int num_blocks_conversion = (int) ceil((double) config->num_sources / max_threads_per_block_conversion);
		dim3 conversion_blocks(num_blocks_conversion, 1, 1);
		dim3 conversion_threads(config->gpu_max_threads_per_block, 1, 1);

		image_to_grid_coords_conversion<<<conversion_blocks, conversion_threads>>>(device->d_sources, config->cell_size_rad,
			config->image_size / 2, config->num_sources);
		cudaDeviceSynchronize();
	}

	int number_minor_cycles = (config->perform_gain_calibration) ? config->number_minor_cycles_cal : config->number_minor_cycles_img;
	printf("UPDATE >>> Performing deconvolution, up to %d minor cycles...\n\n",number_minor_cycles);

	double weak_source_percent = (config->perform_gain_calibration) ? config->weak_source_percent_gc : config->weak_source_percent_img;
	while(cycle_number < number_minor_cycles)
	{
		if(cycle_number % 10 == 0)
			printf("UPDATE >>> Performing minor cycle number: %u...\n\n", cycle_number);

		// Find local row maximum via reduction
		find_max_source_row_reduction<<<reduction_blocks, reduction_threads>>>
			(device->d_image, max_locals, config->image_size, image_search_dim);
		cudaDeviceSynchronize();

		// Find final image maximum via column reduction (local maximums array)
		find_max_source_col_reduction<<<1, 1>>>
			(device->d_sources, max_locals, cycle_number, config->image_size, image_search_dim, config->loop_gain, 
		 		weak_source_percent, config->noise_factor);
		cudaDeviceSynchronize();

		subtract_psf_from_image<<<psf_blocks, psf_threads>>>
				(device->d_image, device->d_sources, device->d_psf, cycle_number, config->image_size, config->loop_gain);
		cudaDeviceSynchronize();

		compress_sources<<<1, 1>>>(device->d_sources);
		cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&exit_early, d_exit_early, sizeof(bool), 0, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

		if(exit_early)
		{
			printf(">>> UPDATE: Terminating minor cycles as now just cleaning noise, cycle number %u...\n\n", cycle_number);
			break;
		}

		cycle_number++;
	}

	// Determine how many compressed sources were found
	cudaMemcpyFromSymbol(&(config->num_sources), d_source_counter, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	if(config->num_sources > 0)
	{
		printf("UPDATE >>> Performing conversion on Source coordinates...\n\n");

		int max_threads_per_block_conversion = min(config->gpu_max_threads_per_block, config->num_sources);
		int num_blocks_conversion = (int) ceil((double) config->num_sources / max_threads_per_block_conversion);
		dim3 conversion_blocks(num_blocks_conversion, 1, 1);
		dim3 conversion_threads(config->gpu_max_threads_per_block, 1, 1);

		grid_to_image_coords_conversion<<<conversion_blocks, conversion_threads>>>(device->d_sources, config->cell_size_rad,
			config->image_size / 2, config->num_sources);
		cudaDeviceSynchronize();
	}

    if(max_locals != NULL) 
    	CUDA_CHECK_RETURN(cudaFree(max_locals));
    max_locals = NULL;
}

void deconvolution_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	if(host->h_sources)
		free(host->h_sources);

	host->h_sources = (Source*) calloc(config->num_sources, sizeof(Source));

    CUDA_CHECK_RETURN(cudaMemcpy(host->h_sources, device->d_sources, config->num_sources * sizeof(Source),
        cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(host->residual_image, device->d_image, config->image_size * config->image_size * sizeof(PRECISION),
        cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

void deconvolution_clean_up(Device_Mem_Handles *device)
{
	// Image
  	if(device->d_image != NULL) 
        CUDA_CHECK_RETURN(cudaFree(device->d_image));
    device->d_image = NULL;

	// PSF
  	if(device->d_psf != NULL) 
        CUDA_CHECK_RETURN(cudaFree(device->d_psf));
    device->d_psf = NULL;

	// Sources
  	if(device->d_sources != NULL) 
        CUDA_CHECK_RETURN(cudaFree(device->d_sources));
    device->d_sources = NULL;
}

__global__ void grid_to_image_coords_conversion(PRECISION3 *sources, const PRECISION cell_size, const int half_grid_size,
	const int source_count)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if(index >= source_count)
		return;

	sources[index].x = (sources[index].x - half_grid_size) * cell_size;
	sources[index].y = (sources[index].y - half_grid_size) * cell_size;
}

__global__ void image_to_grid_coords_conversion(PRECISION3 *sources, const PRECISION cell_size, const int half_grid_size,
	const int source_count)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if(index >= source_count)
		return;

	sources[index].x = ROUND((sources[index].x / cell_size) + half_grid_size); 
	sources[index].y = ROUND((sources[index].y / cell_size) + half_grid_size);
}



__global__ void scale_dirty_image_by_psf(PRECISION *image, PRECISION *psf, PRECISION psf_max, const int image_size)
{
	int col_index = blockIdx.x*blockDim.x + threadIdx.x;
	int row_index = blockIdx.y*blockDim.y + threadIdx.y;

	if(col_index >= image_size || row_index >= image_size)
		return;

	image[row_index * image_size + col_index] /= psf_max;
}

__global__ void find_max_source_row_reduction(const PRECISION *image, PRECISION3 *local_max,
	const int image_dim, const int search_dim)
{
	unsigned int row_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(row_index >= search_dim)
		return;

	const int image_dim_offset = int((image_dim - search_dim) * 0.5);

	// l, m, intensity
	PRECISION3 max = MAKE_PRECISION3(0.0, 0.0, 0.0);

	for(int col_index = 0; col_index < search_dim; ++col_index)
	{
 		int image_index = (row_index+image_dim_offset) * image_dim + col_index+image_dim_offset;
		PRECISION current = image[image_index];
		max.y += ABS(current); // "borrow" y/m to store running average for this row
		
		if(ABS(current) > ABS(max.z))
		{
			// update l and intensity
			max.x = (PRECISION) col_index + image_dim_offset;
			max.z = current;
		}
	}
	
	local_max[row_index] = max;
}

__global__ void find_max_source_col_reduction(PRECISION3 *sources, const PRECISION3 *local_max, const int cycle_number,
	const int image_dim, const int search_dim, const double loop_gain, const double weak_source_percent, 
	const double noise_factor)
{
	const unsigned int col_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(col_index >= 1) // only single threaded
		return;

	//obtain max from row and col and clear the y (row) coordinate.
	PRECISION3 max = MAKE_PRECISION3(0.0, 0.0, 0.0);
	PRECISION3 current = MAKE_PRECISION3(0.0, 0.0, 0.0);
	PRECISION running_avg = 0.0;
	const int image_dim_offset = int((image_dim - search_dim) * 0.5);
	
	for(int index = 0; index < search_dim; ++index)
	{
		current = local_max[index];
		running_avg += current.y; // sum average across all previous row averages	
		current.y = index + image_dim_offset; // now set y/m coordinate correctly

		if(ABS(current.z) > ABS(max.z))
			max = current;
	}

	running_avg /= (search_dim * search_dim); 
	max.z *= loop_gain;
	
	// determine whether we drop out and ignore this source
	bool extracting_noise = ABS(max.z) < noise_factor * running_avg * loop_gain;
	bool weak_source = ABS(max.z) < (ABS(sources[0].z) * weak_source_percent);
	d_exit_early = extracting_noise || weak_source;

	if(d_exit_early)	
		return;	

	// source was reasonable, so we keep it
	sources[d_source_counter] = max;
	++d_source_counter;
}

//used to compress found sources at the same coordinate into one 
__global__ void compress_sources(PRECISION3 *sources)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index >= 1  || d_source_counter == 0) // only single threaded
		return;

	PRECISION3 last_source = sources[d_source_counter - 1];
	for(int i = d_source_counter - 2; i >= 0; --i)
	{
		if((int)last_source.x == (int)sources[i].x && (int)last_source.y == (int)sources[i].y)
		{
			sources[i].z += last_source.z;
			--d_source_counter;
			break;
		}
	}
}

//subtracts overlaid psf from the image
__global__ void subtract_psf_from_image(PRECISION *image, PRECISION3 *sources, const PRECISION *psf, 
	const int cycle_number, const int image_size, const PRECISION loop_gain)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	// thread out of bounds
	if(idx >= image_size || idy >= image_size || d_source_counter == 0)
		return;

	const int half_image_size = image_size / 2;

	// Determine image coordinates relative to source location
	int2 image_coord = make_int2(
		sources[d_source_counter-1].x - half_image_size + idx,
		sources[d_source_counter-1].y - half_image_size + idy
	);
	
	// image coordinates fall out of bounds
	if(image_coord.x < 0 || image_coord.x >= image_size || image_coord.y < 0 || image_coord.y >= image_size)
		return;

	// Get required psf sample for subtraction
	const PRECISION psf_weight = psf[idy * image_size + idx];

	// Subtract shifted psf sample from image
	image[image_coord.y * image_size + image_coord.x] -= psf_weight  * sources[d_source_counter-1].z;
}