
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

#include "../weighting.h"

void visibility_weighting_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
    // Set up
    visibility_weighting_setup(config, host, device);

    // Perform preliminary processes (ie: generate vis weight map)
    if(config->weighting == UNIFORM || config->weighting == ROBUST)
        generate_weight_sum_maps(config, device);
    
    // Scale visibility weights for imaging
    perform_weight_scaling(config, device);

    // Free up weight map (dont need to retain going forward)
    clean_device_weight_map(device);

    // Get back the visibility weight map (density)
    if(!config->retain_device_mem)
    {   
        // Need to transfer newly scaled visibilities back to host
        transfer_scaled_visibilities_to_host(config, host, device);
        // and weights for generating weighted PSF
        transfer_imaging_weights_to_host(config, host, device);
        // Clean up
        visibility_weighting_cleanup(device);
    }
}

void clean_device_weight_map(Device_Mem_Handles *device)
{
    if(device->d_weight_map != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_weight_map));
    device->d_weight_map = NULL;
}

void transfer_imaging_weights_to_host(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
    CUDA_CHECK_RETURN(cudaMemcpy(host->vis_weights, device->d_vis_weights,
        config->num_visibilities * sizeof(PRECISION), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}

void transfer_scaled_visibilities_to_host(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
    CUDA_CHECK_RETURN(cudaMemcpy(host->measured_vis, device->d_measured_vis,
        config->num_visibilities * sizeof(VIS_PRECISION2), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}

void generate_weight_sum_maps(Config *config, Device_Mem_Handles *device)
{
    int num_blocks = (int) ceil((double) config->num_visibilities / config->gpu_max_threads_per_block);
    int max_threads_per_block = min(config->gpu_max_threads_per_block, config->num_visibilities);
    dim3 kernel_blocks(num_blocks, 1, 1);
    dim3 kernel_threads(max_threads_per_block, 1, 1);

	printf("UPDATE >>> Weight mapping using %d blocks, %d threads, for %d visibilities...\n\n",
        num_blocks, max_threads_per_block, config->num_visibilities);

    // Execute mapping kernel
    weight_mapping<<<kernel_blocks, kernel_threads>>>(device->d_weight_map, 
        device->d_vis_uvw_coords, device->d_vis_weights,
        config->num_visibilities, config->grid_size, config->uv_scale);
    cudaDeviceSynchronize();
	printf("UPDATE >>> Weight mapping complete...\n\n");
}

void perform_weight_scaling(Config *config, Device_Mem_Handles *device)
{
    double f = 0.0;

    if(config->weighting == ROBUST)
    {
        PRECISION weight_plane_sum_of_squares = 0.0;
        PRECISION* d_weight_plane_sum_square;
        CUDA_CHECK_RETURN(cudaMalloc(&d_weight_plane_sum_square, sizeof(PRECISION)));
        CUDA_CHECK_RETURN(cudaMemset(d_weight_plane_sum_square, 0, sizeof(PRECISION)));
        cudaDeviceSynchronize();
        
        // Gather sum of real weight plane squared
        int grid_size_square = config->grid_size * config->grid_size;
        printf("Grid size for weighting is %d ... \n", config->grid_size);
        dim3 grid_sum_blocks((int) ceil((double) grid_size_square / config->gpu_max_threads_per_block), 1, 1);
        dim3 grid_sum_threads(min(config->gpu_max_threads_per_block, grid_size_square), 1, 1);
        sum<<<grid_sum_blocks, grid_sum_threads>>>
            (device->d_weight_map, d_weight_plane_sum_square, grid_size_square, true);
        cudaDeviceSynchronize();

        // Get result back from GPU        
        CUDA_CHECK_RETURN(cudaMemcpy(&weight_plane_sum_of_squares, d_weight_plane_sum_square, 
            sizeof(PRECISION),cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        // Clean up
        CUDA_CHECK_RETURN(cudaFree(d_weight_plane_sum_square));
        d_weight_plane_sum_square = NULL;

        // Gather sum of visibility weights (data valid count)
        PRECISION visibility_weights_sum = 0.0;
        PRECISION* d_visibility_weights_sum = NULL;
        CUDA_CHECK_RETURN(cudaMalloc(&d_visibility_weights_sum, sizeof(PRECISION)));
        CUDA_CHECK_RETURN(cudaMemset(d_visibility_weights_sum, 0, sizeof(PRECISION)));
        cudaDeviceSynchronize();
        
        // Gather sum of visibility weights
        dim3 vis_weight_sum_blocks((int) ceil((double) config->num_visibilities / config->gpu_max_threads_per_block), 1, 1);
        dim3 vis_weight_sum_threads(min(config->gpu_max_threads_per_block, config->num_visibilities), 1, 1);
        sum_vis<<<vis_weight_sum_blocks, vis_weight_sum_threads>>>
            (device->d_vis_weights, d_visibility_weights_sum, config->num_visibilities, false);
        cudaDeviceSynchronize();

        // Get result back from GPU        
        CUDA_CHECK_RETURN(cudaMemcpy(&visibility_weights_sum, d_visibility_weights_sum, 
            sizeof(PRECISION),cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        // Clean up
        CUDA_CHECK_RETURN(cudaFree(d_visibility_weights_sum));
        d_visibility_weights_sum = NULL;

        // Perform robust scaling of visibility imaging weights (data valid count from correlator)
        f = pow((5.0 * pow(10.0, -(config->robustness))), 2.0) * (visibility_weights_sum / weight_plane_sum_of_squares);
    }

    // Perform designated reweighting of visibilities
    dim3 vis_weight_scale_blocks((int) ceil((double) config->num_visibilities / config->gpu_max_threads_per_block), 1, 1);
    dim3 vis_weight_scale_threads(min(config->gpu_max_threads_per_block, config->num_visibilities), 1, 1);

    printf("UPDATE >>> Perform weight scale using %d blocks, %d threads, for %d visibilities...\n\n",
            vis_weight_scale_blocks.x, vis_weight_scale_threads.x, config->num_visibilities);

    parallel_scale_vis_weights<<<vis_weight_scale_blocks, vis_weight_scale_threads>>>
        (device->d_vis_uvw_coords, device->d_measured_vis, device->d_vis_weights, device->d_weight_map,
        config->grid_size, config->num_visibilities, config->weighting, f, config->uv_scale);
    cudaDeviceSynchronize();

    // Gather the sum of all now scaled visibility weights
    PRECISION* d_visibility_scaled_weights_sum = NULL;
    CUDA_CHECK_RETURN(cudaMalloc(&d_visibility_scaled_weights_sum, sizeof(PRECISION)));
    CUDA_CHECK_RETURN(cudaMemset(d_visibility_scaled_weights_sum, 0, sizeof(PRECISION)));
    cudaDeviceSynchronize();
    
    // Gather sum of visibility weights
    dim3 vis_weight_sum_blocks((int) ceil((double) config->num_visibilities / config->gpu_max_threads_per_block), 1, 1);
    dim3 vis_weight_sum_threads(min(config->gpu_max_threads_per_block, config->num_visibilities), 1, 1);
    sum_vis<<<vis_weight_sum_blocks, vis_weight_sum_threads>>>
        (device->d_vis_weights, d_visibility_scaled_weights_sum, config->num_visibilities, false);
    cudaDeviceSynchronize();

    // Get result back from GPU        
    CUDA_CHECK_RETURN(cudaMemcpy(&(config->visibility_scaled_weights_sum), d_visibility_scaled_weights_sum, 
        sizeof(PRECISION), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // Clean up
    CUDA_CHECK_RETURN(cudaFree(d_visibility_scaled_weights_sum));
    d_visibility_scaled_weights_sum = NULL;

    printf("Sum of vis weights: %f\n", config->visibility_scaled_weights_sum);
}

__global__ void parallel_scale_vis_weights(const PRECISION3 *vis_uvw,
    VIS_PRECISION2 *vis, VIS_PRECISION *vis_weights, PRECISION *weight_plane, 
    const int grid_size, const int num_vis, const enum weighting_scheme scheme,
    const double f, const double uv_scale)
{
    const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(vis_index >= num_vis)
        return;

    double weight_map_sample = 0.0;
	double weight_to_apply = vis_weights[vis_index];
	
    if(scheme == UNIFORM || scheme == ROBUST)
    {
        const int half_grid_size = grid_size / 2;
        
        const int2 snapped_grid_coord = make_int2(
            (int) ROUND((vis_uvw[vis_index].x * uv_scale)),
            (int) ROUND((vis_uvw[vis_index].y * uv_scale))
        );
        
        const int grid_point_index = (snapped_grid_coord.y + half_grid_size) * grid_size + (snapped_grid_coord.x + half_grid_size);
        weight_map_sample = weight_plane[grid_point_index];
        
        if(scheme == UNIFORM && weight_map_sample > 0.0)
            weight_to_apply /= weight_map_sample;
    
        if(scheme == ROBUST)
            weight_to_apply /= (1.0 + f * weight_map_sample);
    }

    // scale visibilities for PSF/gridding stages
    vis[vis_index].x = VIS_PRECISION(weight_to_apply) * vis[vis_index].x;
    vis[vis_index].y = VIS_PRECISION(weight_to_apply) * vis[vis_index].y;
	vis_weights[vis_index] = (VIS_PRECISION) weight_to_apply;
}

__global__ void weight_mapping(PRECISION *weight_plane, const PRECISION3 *vis_uvw,
    const VIS_PRECISION *vis_weights, const int num_vis, const int grid_size, 
    const double uv_scale)
{
	const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(vis_index >= num_vis)
		return;

	const int half_grid_size = grid_size / 2;

    const int2 snapped_grid_coord = make_int2(
		(int) ROUND((vis_uvw[vis_index].x * uv_scale)),
		(int) ROUND((vis_uvw[vis_index].y * uv_scale))
	);
    
    const int grid_point_index = (snapped_grid_coord.y + half_grid_size) * grid_size + (snapped_grid_coord.x + half_grid_size);
    atomicAdd(&(weight_plane[grid_point_index]), vis_weights[vis_index]);
}

void visibility_weighting_setup(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{   
    // Visibilities
	if(device->d_measured_vis == NULL)
	{
	    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_measured_vis), sizeof(VIS_PRECISION2) * config->num_visibilities));
	    CUDA_CHECK_RETURN(cudaMemcpy(device->d_measured_vis, host->measured_vis, sizeof(VIS_PRECISION2) * config->num_visibilities,
	        cudaMemcpyHostToDevice));
	    cudaDeviceSynchronize();
	}

    // Visibility weights
	if(device->d_vis_weights == NULL)
	{
	    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_weights), sizeof(PRECISION) * config->num_visibilities));
	    CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_weights, host->vis_weights, sizeof(PRECISION) * config->num_visibilities,
            cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
	}

    // Natural weighting does not require mapping of weights
    if(config->weighting == UNIFORM || config->weighting == ROBUST)
    {
        // Sum of weights
        if(device->d_weight_map == NULL)
        {
            int grid_square = config->grid_size * config->grid_size;
            CUDA_CHECK_RETURN(cudaMalloc(&(device->d_weight_map), sizeof(PRECISION) * grid_square));
            CUDA_CHECK_RETURN(cudaMemset(device->d_weight_map, 0, grid_square * sizeof(PRECISION)));
            cudaDeviceSynchronize();
        }

        // Visibility UVW coordinates
        if(device->d_vis_uvw_coords == NULL)
        {
            CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_uvw_coords), sizeof(PRECISION3) * config->num_visibilities));
            CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_uvw_coords, host->vis_uvw_coords, sizeof(PRECISION3) * config->num_visibilities,
                cudaMemcpyHostToDevice));
            cudaDeviceSynchronize();
        }        
    }
}

void visibility_weighting_cleanup(Device_Mem_Handles *device)
{
    if(device->d_measured_vis != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_measured_vis));
    device->d_measured_vis = NULL;

    if(device->d_vis_weights != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_vis_weights));
    device->d_vis_weights = NULL;

    if(device->d_vis_uvw_coords != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_vis_uvw_coords));
    device->d_vis_uvw_coords = NULL;
}

__global__ void sum(PRECISION *input, PRECISION *output, const int num_elements, bool squared)
{
    __shared__ PRECISION temp[MAX_THREADS_PER_BLOCK];
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= num_elements)
        return;

    int local_idx = threadIdx.x;
    temp[local_idx] = (squared) ? input[idx] * input[idx] : input[idx];
    int i = (int) ceilf((float) blockDim.x / 2.0);
    __syncthreads();
    while(i!=0)
    {
        if(idx+i < num_elements && local_idx<i)
                temp[local_idx] += temp[local_idx+i];
        i/=2;
        __syncthreads();

    }
    // Accumulate local thread blocks into one term
    if(local_idx == 0)
        atomicAdd(&(output[0]), temp[0]);
}

__global__ void sum_vis(VIS_PRECISION *input, PRECISION *output, const int num_elements, bool squared)
{
    __shared__ PRECISION temp[MAX_THREADS_PER_BLOCK];
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= num_elements)
        return;

    int local_idx = threadIdx.x;
    temp[local_idx] = (squared) ? input[idx] * input[idx] : input[idx];
    int i = (int) ceilf((float) blockDim.x / 2.0);
    __syncthreads();
    while(i!=0)
    {
        if(idx+i < num_elements && local_idx<i)
                temp[local_idx] += temp[local_idx+i];
        i/=2;
        __syncthreads();

    }
    // Accumulate local thread blocks into one term
    if(local_idx == 0)
        atomicAdd(&(output[0]), temp[0]);
}