
// Copyright 2021 Adam Campbell, Anthony Griffin, Andrew Ensor, Seth Hall
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

#include "../nifty.h"
#include "../msmfs_nifty.h"

void msmfs_nifty_gridding_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{
	printf("UPDATE >>> Executing MSMFS Nifty Gridder as solver, Assuming host side setup in controller...\n\n");

	msmfs_nifty_gridding_set_up(config, host, device);
	//perform nifty - nifty_gridding_run, fft_run ect ect
	start_timer(&timings->solver, false);
	msmfs_nifty_gridding_run(config,host,device);
	stop_timer(&timings->solver, false);

    //if(config->enable_psf)
     //   psf_normalization_nifty(config, device);

	
}
   

void msmfs_nifty_gridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{		
	if(device->d_visibilities == NULL && !config->enable_psf)
	{
	    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_visibilities), sizeof(VIS_PRECISION2) * config->vis_batch_size));
	}
	if(device->d_vis_uvw_coords == NULL)
	{  
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_uvw_coords), sizeof(PRECISION3) * config->uvw_batch_size));
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
	
	if(device->d_w_grid_stack == NULL)
    {   printf("UPDATE >>> Allocating GRID STACK : %d square, %d num planes...\n\n", config->grid_size, config->nifty_config.num_w_grids_batched);
        uint32_t num_w_grid_stack_cells = config->grid_size * config->grid_size * config->nifty_config.num_w_grids_batched;
        CUDA_CHECK_RETURN(cudaMalloc(&(device->d_w_grid_stack), sizeof(PRECISION2) * num_w_grid_stack_cells));
        CUDA_CHECK_RETURN(cudaMemset(device->d_w_grid_stack, 0, num_w_grid_stack_cells * sizeof(PRECISION2)));
    }
	if(device->d_vis_weights == NULL)
    { 
        CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_weights), sizeof(VIS_PRECISION) * config->vis_batch_size));
       // CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_weights, host->vis_weights, sizeof(VIS_PRECISION) * config->num_visibilities, cudaMemcpyHostToDevice));
    }
	if(device->d_prolate == NULL)
    {   printf("UPDATE >>> Allocating the prolate..\n\n");
        CUDA_CHECK_RETURN(cudaMalloc(&(device->d_prolate), sizeof(PRECISION) * (config->image_size/2 + 1)));
	    CUDA_CHECK_RETURN(cudaMemcpy(device->d_prolate, host->prolate,
           sizeof(PRECISION) * (config->image_size/2 + 1), cudaMemcpyHostToDevice));
    }

    if(device->fft_plan == NULL)
    {   printf("UPDATE >>> Allocating the FFT plan..\n\n");
        //CUDA_CHECK_RETURN(cudaHostAlloc(&(device->fft_plan), sizeof(cufftHandle), cudaHostAllocDefault));
        device->fft_plan = (cufftHandle*) calloc(1, sizeof(cufftHandle));
        // Init the plan based on pre-allocated cufft handle (from host mem handles)
        int n[2] = {(int32_t)config->grid_size, (int32_t)config->grid_size};
        CUFFT_SAFE_CALL(cufftPlanMany(
            device->fft_plan,
            2, // number of dimensions per FFT
            n, // number of elements per FFT dimension
            NULL,
            1,
            (int32_t) (config->grid_size * config->grid_size), // num elements per 2d w grid
            NULL,
            1,
            (int32_t) (config->grid_size * config->grid_size), // num elements per 2d w grid
            CUFFT_C2C_PLAN, // Complex to complex
            (int32_t) config->nifty_config.num_w_grids_batched
        ));
    }

	printf("UPDATE >>> copy quadrature kernels to device as symbol ...\n\n");
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(quadrature_nodes,   host->quadrature_nodes,   QUADRATURE_SUPPORT_BOUND*sizeof(PRECISION), 0));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(quadrature_weights, host->quadrature_weights, QUADRATURE_SUPPORT_BOUND*sizeof(PRECISION), 0));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(quadrature_kernel,  host->quadrature_kernel,  QUADRATURE_SUPPORT_BOUND*sizeof(PRECISION), 0));

    printf("UPDATE >>> Finished setting up device side NIFTY buffers ...\n\n");
}  

void msmfs_nifty_clean_up(Device_Mem_Handles *device)
{
	printf("UPDATE >>> Freeing allocated device memory for NIFTY Gridding / FFT / Convolution Correction...\n\n");

    if(device->d_w_grid_stack != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_w_grid_stack));
    device->d_w_grid_stack = NULL;


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

void msmfs_nifty_gridding_run(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	// Determine the block/thread distribution per w correction and summation batch
    // note: based on image size, not grid size, as we take inner region and discard padding
    
	
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
	
	dim3 w_correction_threads(
        min((uint32_t)32, (temp_size + 1) / 2),
        min((uint32_t)32, (temp_size + 1) / 2)
    );
    dim3 w_correction_blocks(
        (temp_size/2 + 1 + w_correction_threads.x - 1) / w_correction_threads.x,  // allow extra in negative x quadrants, for asymmetric image centre
        (temp_size/2 + 1 + w_correction_threads.y - 1) / w_correction_threads.y   // allow extra in negative y quadrants, for asymmetric image centre
    );
	
	
	PRECISION *d_temp_image;
	CUDA_CHECK_RETURN(cudaMalloc(&(d_temp_image), sizeof(PRECISION) * temp_size * temp_size));
	
	for(int ts_batch=0;ts_batch<total_num_batches; ts_batch++)
	{	//copy vis UVW here
		//fill UVW and VIS
		printf("Griddding batch number %d ...\n",ts_batch);
		
		int current_vis_batch_size = min(visLeftToProcess, config->vis_batch_size);
		int current_uvw_batch_size = min(uwvLeftToProcess, config->uvw_batch_size);
		
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_weights, host->vis_weights+(ts_batch*config->vis_batch_size),
						sizeof(VIS_PRECISION) * current_vis_batch_size, cudaMemcpyHostToDevice));
		
		if(!config->enable_psf)
		{	CUDA_CHECK_RETURN(cudaMemcpy(device->d_visibilities, host->visibilities+(ts_batch*config->vis_batch_size), 
						sizeof(VIS_PRECISION2) * current_vis_batch_size, cudaMemcpyHostToDevice));
		}
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_uvw_coords, host->vis_uvw_coords+(ts_batch*config->uvw_batch_size), 
						sizeof(PRECISION3) * current_uvw_batch_size, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	
	
		int min_cuda_grid_size;
		int cuda_block_size;
		cudaOccupancyMaxPotentialBlockSize(&min_cuda_grid_size, &cuda_block_size, nifty_gridding, 0, 0);
		int cuda_grid_size = (current_vis_batch_size + cuda_block_size - 1) / cuda_block_size;  // create 1 thread per visibility to be gridded
		printf("UPDATE >>> Nifty Gridder using grid size %d where minimum grid size available is %d and using block size %d..\n\n", cuda_grid_size,min_cuda_grid_size,cuda_block_size);

		uint32_t total_w_grid_batches = (config->nifty_config.num_total_w_grids + config->nifty_config.num_w_grids_batched - 1) / config->nifty_config.num_w_grids_batched;
		uint32_t num_w_grid_stack_cells = config->grid_size * config->grid_size * config->nifty_config.num_w_grids_batched;

		// Determine how many w grid subset batches to process in total
	
		//loop for each taylor term
		for(int t=0;t<num_taylors;t++)
		{
			
			for(int batch = 0; batch < total_w_grid_batches; batch++)
			{
				uint32_t num_w_grids_subset = min(
					config->nifty_config.num_w_grids_batched,
					config->nifty_config.num_total_w_grids - ((batch * config->nifty_config.num_w_grids_batched) % config->nifty_config.num_total_w_grids)
				);

				int32_t grid_start_w = batch*config->nifty_config.num_w_grids_batched;

				printf("Gridder calling nifty_gridding kernel for w grids %d to %d (%d planes in current batch)\n\n", 
						grid_start_w, (grid_start_w + (int32_t)num_w_grids_subset - 1), num_w_grids_subset);
				
				// Perform gridding on a "chunk" of w grids
				msmfs_nifty_gridding<<<cuda_grid_size, cuda_block_size>>>(
					device->d_visibilities,
					device->d_vis_weights,
					device->d_vis_uvw_coords,
					current_vis_batch_size,
					device->d_w_grid_stack,
					config->grid_size,
					grid_start_w,
					num_w_grids_subset,
					config->nifty_config.support,
					config->nifty_config.beta,
					config->nifty_config.upsampling,
					config->uv_scale,
					config->w_scale, 
					config->nifty_config.min_plane_w,
					config->number_frequency_channels, 
					config->num_baselines, 
					config->frequency_hz_start, 
					freq_inc,
					config->mf_reference_hz,
					t,
					config->enable_psf,
					config->nifty_config.perform_shift_fft,
					true
				);

				/**************************************************************************************
				 * TESTING FUNCTIONALITY ONLY - TESTING FUNCTIONALITY ONLY - TESTING FUNCTIONALITY ONLY
				 *************************************************************************************/
				// // Copy back and save each w grid to file for review of accuracy
				// std::cout << "Copying back grid stack from device to host..." << std::endl; 
				// proof_of_concept_copy_w_grid_stack_to_host(host, device, config);
				// // Save each w grid into single file of complex type, use python to render via matplotlib
				// std::cout << "Saving grid stack to file..." << std::endl;
				// proof_of_concept_save_w_grids_to_file(host, config, batch);
				/**************************************************************************************
				 * TESTING FUNCTIONALITY ONLY - TESTING FUNCTIONALITY ONLY - TESTING FUNCTIONALITY ONLY
				 *************************************************************************************/

				// Perform 2D FFT on each bound w grid
				printf("UPDATE >>> Nifty Gridder calling CUFFT...\n\n");
				CUFFT_SAFE_CALL(CUFFT_EXECUTE_C2C(*(device->fft_plan), device->d_w_grid_stack, device->d_w_grid_stack, CUFFT_INVERSE));

				/**************************************************************************************
				 * TESTING FUNCTIONALITY ONLY - TESTING FUNCTIONALITY ONLY - TESTING FUNCTIONALITY ONLY
				 *************************************************************************************/
				// // Copy back and save each w grid to file for review of accuracy
				// std::cout << "Copying back grid stack from device to host..." << std::endl; 
				// proof_of_concept_copy_w_grid_stack_to_host(host, device, config);
				// // Save each w grid into single file of complex type, use python to render via matplotlib
				// std::cout << "Saving grid stack to file..." << std::endl;
				// proof_of_concept_save_w_images_to_file(host, config, batch);

				// zero dirty image  (debugging option)
				// CUDA_CHECK_RETURN(cudaMemset(device->d_dirty_image, 0, config->image_size * config->image_size * sizeof(PRECISION)));

				/**************************************************************************************
				 * TESTING FUNCTIONALITY ONLY - TESTING FUNCTIONALITY ONLY - TESTING FUNCTIONALITY ONLY
				 *************************************************************************************/		
				// Perform phase shift on a "chunk" of planes and sum into single real plane
				apply_w_screen_and_sum<<<w_correction_blocks, w_correction_threads>>>(
					d_temp_image,
					temp_size,
					config->cell_size_rad,
					device->d_w_grid_stack,
					config->grid_size,   
					grid_start_w,
					num_w_grids_subset,
					PRECISION(1.0)/config->w_scale,
					config->nifty_config.min_plane_w,
					config->nifty_config.perform_shift_fft
				);

				printf("UPDATE >>> Resetting device w grid stack memory for next batch..\n\n");
				CUDA_CHECK_RETURN(cudaMemset(device->d_w_grid_stack, 0, num_w_grid_stack_cells * sizeof(PRECISION2)));
			
			}
			//stack plane on taylor cube
			int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, temp_size);
			int num_blocks_per_dimension = (int) ceil((double) temp_size / max_threads_per_block_dimension);
			dim3 accumulate_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
			dim3 accumulate_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);
			nifty_accumulate_taylor_term_plane<<<accumulate_blocks, accumulate_threads>>>(d_temp_image, config->enable_psf ? device->d_psf : device->d_image, t, temp_size);
			cudaDeviceSynchronize();	
			
			CUDA_CHECK_RETURN(cudaMemset(d_temp_image, 0, temp_size * temp_size * sizeof(PRECISION)));		
		}
		visLeftToProcess -= config->vis_batch_size;
		uwvLeftToProcess -= config->uvw_batch_size;
	}
	
	CUDA_CHECK_RETURN(cudaFree(device->d_vis_weights));
		device->d_vis_weights = NULL;
	CUDA_CHECK_RETURN(cudaFree(device->d_visibilities));
		device->d_visibilities = NULL;
	CUDA_CHECK_RETURN(cudaFree(device->d_vis_uvw_coords));
		device->d_vis_uvw_coords = NULL;		
	
	
    // Need to determine final scaling factor for scaling dirty image by w grid accumulation
    PRECISION inv_w_range = 1.0 / (config->nifty_config.max_plane_w - config->nifty_config.min_plane_w);

    // Perform convolution correction and final scaling on single real plane
    // note: can recycle same block/thread dims as w correction kernel
	
	
	printf("UPDATE:: YO IS THIS SCREWWED UP %f \n\n",config->visibility_scaled_weights_sum);
	config->visibility_scaled_weights_sum = config->num_host_visibilities;
	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, temp_size);
	int num_blocks_per_dimension = (int) ceil((double) temp_size / max_threads_per_block_dimension);
	dim3 cc_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 cc_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);
	for(int t=0;t<num_taylors;t++)
	{
		msmfs_conv_corr_and_scaling<<<cc_blocks, cc_threads>>>(
			(config->enable_psf) ? device->d_psf : device->d_image,
			temp_size,
			config->cell_size_rad,
			config->nifty_config.support,
			config->nifty_config.conv_corr_norm_factor,
			device->d_prolate,
			inv_w_range,
			config->visibility_scaled_weights_sum,
			PRECISION(1.0)/config->w_scale,
			t,
			true
		);
	}

}

/**********************************************************************
 * Performs the gridding (or degridding) of visibilities across a subset of w planes
 * Parallelised so each CUDA thread processes a single visibility
 **********************************************************************/
 __global__ void msmfs_nifty_gridding(
    
    VIS_PRECISION2 *visibilities, // INPUT(gridding) OR OUTPUT(degridding): complex visibilities
    const VIS_PRECISION *vis_weights, // INPUT: weight for each visibility
    const PRECISION3 *uvw_coords, // INPUT: (u, v, w) coordinates for each visibility
    const uint32_t num_visibilities, // total num visibilities

    PRECISION2 *w_grid_stack, // OUTPUT: flat array containing 2D computed w grids, presumed initially clear
    const uint32_t grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
    const int32_t grid_start_w, // signed index of first w grid in current subset stack
    const uint32_t num_w_grids_subset, // number of w grids bound in current subset stack

    const uint32_t support, // full support for gridding kernel
    const PRECISION beta, // beta constant used in exponential of semicircle kernel
    const PRECISION upsampling, // upscaling factor for determining grid size (sigma)
    const PRECISION uv_scale, // scaling factor for conversion of uv coords to grid coordinates (grid_size * cell_size)
    const PRECISION w_scale, // scaling factor for converting w coord to signed w grid index
    const PRECISION min_plane_w, // w coordinate of smallest w plane  
	const int num_channels,
	const int num_baselines,
	const PRECISION metres_wavelength_scale, // for w coordinate	
	const PRECISION freqInc,
	const PRECISION vref, 
	const int taylor_term,
    const bool generating_psf, // flag for enabling/disabling creation of PSF using same gridding code
    const bool perform_shift_fft, // flag to (equivalently) rearrange each grid so origin is at lower-left corner for FFT
    const bool solving // flag to enable degridding operations instead of gridding
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < (int32_t)num_visibilities)
    {
		
		int timeStepOffset = i/(num_channels*num_baselines);
		int uvwIndex = (i % num_baselines) + (timeStepOffset*num_baselines);
		PRECISION3 local_uvw = uvw_coords[uvwIndex];
		
		//chnnelNum - visIndex 
		int channelNumber = (i / num_baselines) % num_channels; 
		
		PRECISION frequency = (metres_wavelength_scale + channelNumber*freqInc); //meters to wavelengths conversion
		
	
		PRECISION freqUVWScale = frequency / PRECISION(SPEED_OF_LIGHT); //meters to wavelengths conversion
	
		PRECISION vis_freq_scale = 1.0;
	
		if(ABS(frequency - vref) > 1E-6)
			vis_freq_scale = POW((frequency - vref)/vref, taylor_term);
		
		
		local_uvw.x *= freqUVWScale;
		local_uvw.y *= freqUVWScale;
		local_uvw.z *= freqUVWScale;
		
		
        const PRECISION half_support = PRECISION(support)/2.0; // NOTE confirm everyone's understanding of what support means eg when even/odd
        const int32_t grid_min_uv = -(int32_t)grid_size/2; // minimum coordinate on grid along u or v axes
        const int32_t grid_max_uv = ((int32_t)grid_size-1)/2; // maximum coordinate on grid along u or v axes

        // Determine whether to flip visibility coordinates, so w is usually positive
        VIS_PRECISION flip = (local_uvw.z < 0.0) ? -1.0 : 1.0; 

        // Calculate bounds of where gridding kernel will be applied for this visibility
        PRECISION3 uvw_coord = MAKE_PRECISION3(
            local_uvw.x * uv_scale * (PRECISION)flip,
            local_uvw.y * uv_scale * (PRECISION)flip,
            (local_uvw.z * (PRECISION)flip - min_plane_w) * w_scale
        );

        int32_t grid_u_least = max((int32_t)CEIL(uvw_coord.x-(PRECISION)half_support), grid_min_uv);
        int32_t grid_u_largest = min((int32_t)FLOOR(uvw_coord.x+(PRECISION)half_support), grid_max_uv);
        int32_t grid_v_least = max((int32_t)CEIL(uvw_coord.y-(PRECISION)half_support), grid_min_uv);
        int32_t grid_v_largest = min((int32_t)FLOOR(uvw_coord.y+(PRECISION)half_support), grid_max_uv);
        int32_t grid_w_least = max((int32_t)CEIL(uvw_coord.z-(PRECISION)half_support), grid_start_w);
        int32_t grid_w_largest = min((int32_t)FLOOR(uvw_coord.z+(PRECISION)half_support), grid_start_w+(int32_t)num_w_grids_subset-1);
        // perform w coord check first to help short-circuit CUDA kernel execution
        if ((grid_w_least>grid_w_largest) || (grid_u_least>grid_u_largest) || (grid_v_least>grid_v_largest))
        {
            return; // this visibility has no overlap with the current subset stack so avoid further calculations with this visibility
        }

        // calculate the necessary kernel values along u and v directions for this uvw_coord
        VIS_PRECISION inv_half_support = (VIS_PRECISION)1.0/(VIS_PRECISION)half_support;

        // bound above the maximum possible support for use in nifty_gridding kernel when precalculating kernel values
        VIS_PRECISION kernel_u[KERNEL_SUPPORT_BOUND];
        uint32_t kernel_index_u = 0;
        for (int32_t grid_coord_u=grid_u_least; grid_coord_u<=grid_u_largest; grid_coord_u++)
        {
            kernel_u[kernel_index_u++] = exp_semicircle(beta, (VIS_PRECISION)(grid_coord_u-uvw_coord.x)*inv_half_support);
        }

        VIS_PRECISION kernel_v[KERNEL_SUPPORT_BOUND];
        uint32_t kernel_index_v = 0;
        for (int32_t grid_coord_v=grid_v_least; grid_coord_v<=grid_v_largest; grid_coord_v++)
        {
            kernel_v[kernel_index_v++] = exp_semicircle(VIS_PRECISION(beta), (VIS_PRECISION)(grid_coord_v-uvw_coord.y)*inv_half_support);
        }

        VIS_PRECISION2 vis_weighted = MAKE_VIS_PRECISION2(0.0, 0.0);
        if(solving)
        {
            vis_weighted = (generating_psf) ? MAKE_VIS_PRECISION2(1.0 * vis_freq_scale, 0.0) :
								MAKE_VIS_PRECISION2(PRECISION(visibilities[i].x) * vis_freq_scale, PRECISION(visibilities[i].y) * vis_freq_scale);
            vis_weighted.x *= vis_weights[i];
            vis_weighted.y *= vis_weights[i] * flip; // complex conjugate for negative w coords
        }

        // iterate through each w-grid
        const int32_t origin_offset_uv = (int32_t)(grid_size/2); // offset of origin along u or v axes
		
        for (int32_t grid_coord_w=grid_w_least; grid_coord_w<=grid_w_largest; grid_coord_w++)
        {
            VIS_PRECISION kernel_w = exp_semicircle(beta, (VIS_PRECISION)(grid_coord_w-uvw_coord.z)*inv_half_support);
            int32_t grid_index_offset_w = (grid_coord_w-grid_start_w)*(int32_t)(grid_size*grid_size);
            kernel_index_v = 0;
            for (int32_t grid_coord_v=grid_v_least; grid_coord_v<=grid_v_largest; grid_coord_v++)
            {  
                int32_t grid_index_offset_vw = grid_index_offset_w + (grid_coord_v+origin_offset_uv)*(int32_t)grid_size;
                kernel_index_u = 0;
                for (int32_t grid_coord_u=grid_u_least; grid_coord_u<=grid_u_largest; grid_coord_u++)
                {
                    // apply the separable kernel to the weighted visibility and accumulate at the grid_coord
                    VIS_PRECISION kernel_value = kernel_u[kernel_index_u] * kernel_v[kernel_index_v] * kernel_w;
                    bool odd_grid_coordinate = ((grid_coord_u + grid_coord_v) & (int32_t)1) != (int32_t)0;
                    kernel_value = (perform_shift_fft && odd_grid_coordinate) ? -kernel_value : kernel_value;

                    int32_t grid_offset_uvw = grid_index_offset_vw + (grid_coord_u+origin_offset_uv);

				

                    if(solving) // accumulation of visibility onto w-grid plane
                    {	
                        // Atomic add of type double only supported on NVIDIA GPU architecture Pascal and above
                        atomicAdd(&w_grid_stack[grid_offset_uvw].x, (PRECISION)(vis_weighted.x * kernel_value));
                        atomicAdd(&w_grid_stack[grid_offset_uvw].y, (PRECISION)(vis_weighted.y * kernel_value));
                    }
                    else // extraction of visibility from w-grid plane
                    {
						VIS_PRECISION2 accumulation = MAKE_VIS_PRECISION2(w_grid_stack[grid_offset_uvw].x * (PRECISION)kernel_value,
									w_grid_stack[grid_offset_uvw].y * (PRECISION) kernel_value);
                        vis_weighted.x += accumulation.x;
                        vis_weighted.y += accumulation.y;
                    }

                    kernel_index_u++;
                }
                kernel_index_v++;
            }
        }

        if(!solving) // degridding
        {	//visibilities[i].x = 0.0;
			//visibilities[i].y = 0.0;
            visibilities[i].x += vis_weighted.x;
            visibilities[i].y += (vis_weighted.y * flip);
        }
    }
}

__global__ void nifty_accumulate_taylor_term_plane(PRECISION *plane, PRECISION *image_cube, const int t_plane, const int image_dim)
{
    int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
    if(row_index >= image_dim || col_index >= image_dim)
        return;

    int t_plane_offset = t_plane * image_dim * image_dim;
 	int plane_index = row_index * image_dim + col_index;
    image_cube[t_plane_offset+plane_index] +=  plane[plane_index];
}
/**********************************************************************
 * Performs convolution correction and final scaling of dirty image
 * using precalculated and runtime calculated correction values.
 * See conv_corr device function for more details
 * Note precalculated convolutional correction for (l, m) are normalised to max of 1,
 * but value for n is calculated at runtime, therefore normalised at runtime by C(0)
 **********************************************************************/
 
 
 __global__ void msmfs_conv_corr_and_scaling(
    PRECISION *dirty_image,
    const uint32_t image_size,
    const PRECISION pixel_size,
    const uint32_t support,
    const PRECISION conv_corr_norm_factor,
    const PRECISION *conv_corr_kernel,
    const PRECISION inv_w_range,
    const PRECISION weight_channel_product,
    const PRECISION inv_w_scale,
    const bool solving,
	const int t_plane
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	
	if(i >= image_size || j >= image_size)
    	return;

	int t_plane_offset = t_plane * image_size * image_size;
	
    const int half_image_size = image_size / 2;
    const int taper_x_index = abs(i - half_image_size);
    const int taper_y_index = abs(j - half_image_size);
	
	
	PRECISION l = pixel_size * taper_x_index;
	PRECISION m = pixel_size * taper_y_index;
	PRECISION n = SQRT(PRECISION(1.0)-l*l-m*m) - PRECISION(1.0);
	PRECISION l_conv = conv_corr_kernel[taper_x_index];
	PRECISION m_conv = conv_corr_kernel[taper_y_index];
	PRECISION n_conv = conv_corr((PRECISION)support, n*inv_w_scale) * conv_corr_norm_factor * conv_corr_norm_factor;

	// Note: scaling (everything after division) does not appear to be present in reference NIFTY code
	// so it may need to be removed if testing this code against the reference code
	// repo: https://gitlab.mpcdf.mpg.de/ift/nifty_gridder
	PRECISION correction = (l_conv * m_conv * n_conv); // / ((n + PRECISION(1.0)) * inv_w_range);
	//PRECISION correction = conv_corr((PRECISION)support, n*inv_w_scale); 
	
	//if(solving)
	//	correction = PRECISION(1.0)/(correction*weight_channel_product);
	//else
		correction = PRECISION(1.0)/(correction);
		//correction = PRECISION(1.0)/conv_corr((PRECISION)support, n*inv_w_scale); 
	dirty_image[t_plane_offset+i*image_size+j] *= correction; 
}
 
 /*
__global__ void msmfs_conv_corr_and_scaling(
    PRECISION *dirty_image,
    const uint32_t image_size,
    const PRECISION pixel_size,
    const uint32_t support,
    const PRECISION conv_corr_norm_factor,
    const PRECISION *conv_corr_kernel,
    const PRECISION inv_w_range,
    const PRECISION weight_channel_product,
    const PRECISION inv_w_scale,
    const bool solving,
	const int t_plane
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t half_image_size = image_size / 2;
	
	int t_plane_offset = t_plane * image_size * image_size;

    if(i <= (int32_t)half_image_size && j <= (int32_t)half_image_size)
    {
        PRECISION l = pixel_size * i;
        PRECISION m = pixel_size * j;
        PRECISION n = SQRT(PRECISION(1.0)-l*l-m*m) - PRECISION(1.0);
        PRECISION l_conv = conv_corr_kernel[i];
        PRECISION m_conv = conv_corr_kernel[j];
        PRECISION n_conv = conv_corr((PRECISION)support, n*inv_w_scale) * conv_corr_norm_factor * conv_corr_norm_factor;
        
		
		
        // Note: scaling (everything after division) does not appear to be present in reference NIFTY code
        // so it may need to be removed if testing this code against the reference code
        // repo: https://gitlab.mpcdf.mpg.de/ift/nifty_gridder
        PRECISION correction = (l_conv * m_conv * n_conv); // / ((n + PRECISION(1.0)) * inv_w_range);
        //PRECISION correction = conv_corr((PRECISION)support, n*inv_w_scale); 
        
       // if(solving)
            //correction = PRECISION(1.0)/(correction*weight_channel_product);
        //else
            correction = PRECISION(1.0)/(correction);
            //correction = PRECISION(1.0)/conv_corr((PRECISION)support, n*inv_w_scale); 

        // Going to need offsets to stride from pixel to pixel for this thread
        const int32_t origin_offset_image_centre = (int32_t)half_image_size; // offset of origin (in dirty image) along l or m axes
        const int32_t image_index_offset_image_centre = origin_offset_image_centre*((int32_t)image_size) + origin_offset_image_centre;
        
		if(i ==0 && j == 0)
			printf("GPU: CENTRE FOUND %f \n ",dirty_image[t_plane_offset+image_index_offset_image_centre - j*((int32_t)image_size) - i]);
		
        if(i < (int32_t)half_image_size && j < (int32_t)half_image_size)
        {
            dirty_image[t_plane_offset+image_index_offset_image_centre + j*((int32_t)image_size) + i] *= correction; 
        }
        // Special cases along centre of image doesn't update four pixels
        if(i > 0 && j < (int32_t)half_image_size)
        {
            dirty_image[t_plane_offset+image_index_offset_image_centre + j*((int32_t)image_size) - i] *= correction; 
        }
        if(j > 0 && i < (int32_t)half_image_size)
        {
            dirty_image[t_plane_offset+image_index_offset_image_centre - j*((int32_t)image_size) + i] *= correction; 
        }
        if(i > 0 && j > 0)
        {
            dirty_image[t_plane_offset+image_index_offset_image_centre - j*((int32_t)image_size) - i] *= correction; 
        }
    }
}
*/
