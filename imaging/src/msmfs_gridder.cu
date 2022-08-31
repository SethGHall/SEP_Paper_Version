

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
#include "../msmfs_gridder.h"


void gridding_execute_msmfs(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{
	printf("UPDATE >> MSMFS Gridding for %d Taylor terms ", config->mf_num_moments);
	
	msmfs_gridding_set_up(config,host,device);
	start_timer(&timings->solver, false);
	gridding_run_msmfs(config,host,device);
	stop_timer(&timings->solver, false);
}

void msmfs_gridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	// UV-Plane
	int grid_square = config->grid_size * config->grid_size;
	if(device->d_uv_grid == NULL)
	{
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
	if(device->fft_plan == NULL)
	{
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
	if(device->d_vis_weights == NULL)
	{
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_weights), sizeof(VIS_PRECISION) * config->vis_batch_size));
    	cudaDeviceSynchronize();
	}
	
	if(config->enable_psf)
	{
		int num_taylors = 2*config->mf_num_moments - 1;
		
		if(device->d_psf != NULL)
		{	cudaFree(device->d_psf);
		}
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_psf), sizeof(PRECISION) * config->psf_size * config->psf_size * num_taylors));
	}
	else{
		if(device->d_image != NULL)
		{	cudaFree(device->d_image);
		}
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_image), sizeof(PRECISION) * config->image_size * config->image_size * config->mf_num_moments));
	}
}

void msmfs_gridding_clean_up(Device_Mem_Handles *device)
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


void gridding_run_msmfs(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	
	int total_num_batches = (int)CEIL(double(config->num_timesteps) / double(config->timesteps_per_batch));
		
	int visLeftToProcess = config->num_host_visibilities;
	int uwvLeftToProcess = config->num_host_uvws;
	
	PRECISION freq_inc = 0.0;
	if(config->number_frequency_channels > 1)
		freq_inc = PRECISION(config->frequency_bandwidth) / (config->number_frequency_channels-1); 
	
	int num_taylors = config->mf_num_moments;
	int temp_size = config->image_size;
	if(config->enable_psf)
	{
		num_taylors = 2*config->mf_num_moments - 1;
		temp_size = config->psf_size;
	}
	
	PRECISION *d_temp_image;
	CUDA_CHECK_RETURN(cudaMalloc(&(d_temp_image), sizeof(PRECISION) * temp_size * temp_size));
	
	for(int ts_batch=0;ts_batch<total_num_batches; ts_batch++)
	{	//co

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
		
		for(int t=0;t<num_taylors;t++)
		{			
			gridding_msmfs<<<kernel_blocks, kernel_threads>>>(device->d_uv_grid, device->d_kernels, device->d_kernel_supports, 
						device->d_vis_uvw_coords, device->d_visibilities, current_vis_batch_size, config->oversampling,
						config->grid_size, config->uv_scale, config->w_scale, config->num_kernels, config->enable_psf,  
						device->d_vis_weights, config->number_frequency_channels, config->num_baselines, 
						config->frequency_hz_start, freq_inc, config->mf_reference_hz,t);
			cudaDeviceSynchronize();   
			
			int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->grid_size);
			int num_blocks_per_dimension = (int) ceil((double) config->grid_size / max_threads_per_block_dimension);
			dim3 c2c_shift_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
			dim3 c2c_shift_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

			// Perform 2D FFT shift
			fft_shift_complex_to_complex<<<c2c_shift_blocks, c2c_shift_threads>>>(device->d_uv_grid, config->grid_size);
			cudaDeviceSynchronize();
  
			CUFFT_SAFE_CALL(CUFFT_EXECUTE_C2C(*(device->fft_plan), device->d_uv_grid, device->d_uv_grid, CUFFT_INVERSE));
			cudaDeviceSynchronize();

			max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, temp_size);
			num_blocks_per_dimension = (int) ceil((double) temp_size / max_threads_per_block_dimension);
			dim3 c2r_shift_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
			dim3 c2r_shift_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

			
			fft_shift_complex_to_real<<<c2r_shift_blocks, c2r_shift_threads>>>(device->d_uv_grid, d_temp_image, config->grid_size, temp_size);
			cudaDeviceSynchronize();

			accumulate_taylor_term_plane<<<c2r_shift_blocks, c2r_shift_threads>>>(d_temp_image, config->enable_psf ? device->d_psf : device->d_image, t, temp_size);
			cudaDeviceSynchronize();	
			
			CUDA_CHECK_RETURN(cudaMemset(device->d_uv_grid, 0, config->grid_size * config->grid_size * sizeof(PRECISION2)));
			CUDA_CHECK_RETURN(cudaMemset(d_temp_image, 0, temp_size * temp_size * sizeof(PRECISION)));			   
		}
		
		visLeftToProcess -= config->vis_batch_size;
		uwvLeftToProcess -= config->uvw_batch_size;
		
		
	}
	
	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, temp_size);
	int num_blocks_per_dimension = (int) ceil((double) temp_size / max_threads_per_block_dimension);
	dim3 cc_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 cc_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);
	//CC on each plane:
	for(int t=0;t<num_taylors;t++)
	{
		execute_convolution_correction_msmfs<<<cc_blocks, cc_threads>>>(config->enable_psf ? device->d_psf : device->d_image, device->d_prolate, t, temp_size);
	}
	
	CUDA_CHECK_RETURN(cudaFree(d_temp_image));
	
	CUDA_CHECK_RETURN(cudaFree(device->d_vis_weights));
		device->d_vis_weights = NULL;
	CUDA_CHECK_RETURN(cudaFree(device->d_visibilities));
		device->d_visibilities = NULL;
	CUDA_CHECK_RETURN(cudaFree(device->d_vis_uvw_coords));
		device->d_vis_uvw_coords = NULL;		
	
	printf("UPDATE >>> MSMFS Gridding complete...\n\n");
}


void pull_msmfs_image_cube(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{	
	int cube_dim = 0;
	
	if(config->enable_psf)
		cube_dim = config->psf_size * config->psf_size * (2 * config->mf_num_moments - 1);
	else
		cube_dim = config->image_size * config->image_size * config->mf_num_moments;
	
	printf("UPDATE >>> Pulling image cube  %d .. \n\n",cube_dim);
	
    CUDA_CHECK_RETURN(cudaMemcpy(config->enable_psf ? host->h_psf : host->dirty_image, config->enable_psf ? device->d_psf : device->d_image, cube_dim * sizeof(PRECISION),
        cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();	
	
	
}

//applys w-projection to the grid, snapping to nearest W plane and UV oversample
__global__ void gridding_msmfs(PRECISION2 *grid, const VIS_PRECISION2 *kernel, const int2 *supports,
	const PRECISION3 *vis_uvw, const VIS_PRECISION2 *vis, const int num_vis, const int oversampling,
	const int grid_size, const double uv_scale, const double w_scale, const int num_w_kernels, 
	const bool psf, const VIS_PRECISION *vis_weights, const int num_channels, const int num_baselines, 
	const PRECISION freq, const PRECISION freqInc, const PRECISION vref, const int taylor_term)
{
	const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(vis_index >= num_vis)
		return;

	int timeStepOffset = vis_index/(num_channels*num_baselines);
	int uvwIndex = (vis_index % num_baselines) + (timeStepOffset*num_baselines);
	PRECISION3 local_uvw = vis_uvw[uvwIndex];
	
	//chnnelNum - visIndex 
	int channelNumber = (vis_index / num_baselines) % num_channels; 
	
	PRECISION frequency = (freq + channelNumber*freqInc);
	
	PRECISION freqUVWScale = frequency / PRECISION(SPEED_OF_LIGHT); //meters to wavelengths conversion
	
	PRECISION vis_freq_scale = 1.0;
	
	if(ABS(frequency - vref) > 1E-6)
		vis_freq_scale = POW((frequency - vref)/vref, taylor_term);
	
	//convert local_UVW meters to wavelengths based on frequency
	local_uvw.x *= freqUVWScale;
	local_uvw.y *= freqUVWScale;
	local_uvw.z *= freqUVWScale;
	
	
	VIS_PRECISION2 vis_value = 
		(!psf) ? vis[vis_index]: MAKE_VIS_PRECISION2((VIS_PRECISION) vis_weights[vis_index], 0.0);
	
	vis_value.x = VIS_PRECISION(vis_freq_scale * (PRECISION)vis_value.x);
	vis_value.y = VIS_PRECISION(vis_freq_scale * (PRECISION)vis_value.y);
	
	
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

	VIS_PRECISION conjugate = (local_uvw.z < 0.0) ? -1.0 : 1.0;

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



__global__ void execute_convolution_correction_msmfs(PRECISION *image, const PRECISION *prolate, const int t_plane, const int image_size)
{
	const int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    const int col_index = threadIdx.x + blockDim.x * blockIdx.x;

    if(row_index >= image_size || col_index >= image_size)
    	return;

	int t_plane_offset = t_plane * image_size * image_size;
	
    const int image_index = t_plane_offset + row_index * image_size + col_index;
    const int half_image_size = image_size / 2;
    const int taper_x_index = abs(col_index - half_image_size);
    const int taper_y_index = abs(row_index - half_image_size);

    // patch to fix out of bounds taper access (masking with zero for first row/col)
    // TODO: review a better technique to solve this, possibly increased taper width by one?
    const PRECISION taper = (taper_x_index >= half_image_size || taper_y_index >= half_image_size) 
                                ? 0.0 : prolate[taper_x_index] * prolate[taper_y_index];
    image[image_index] = (ABS(taper) > (1E-10)) ? image[image_index] / taper  : 0.0;
}

__global__ void accumulate_taylor_term_plane(PRECISION *plane, PRECISION *image_cube, const int t_plane, const int image_dim)
{
    int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
    if(row_index >= image_dim || col_index >= image_dim)
        return;

    int t_plane_offset = t_plane * image_dim * image_dim;
 	int plane_index = row_index * image_dim + col_index;
    image_cube[t_plane_offset+plane_index] +=  plane[plane_index];
}



