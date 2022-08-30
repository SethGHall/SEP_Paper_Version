
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

// Use of constant buffers instead of cudaMalloc'd memory allows for
// more efficient caching, and "broadcasting" across threads
__constant__ PRECISION quadrature_nodes[QUADRATURE_SUPPORT_BOUND];
__constant__ PRECISION quadrature_weights[QUADRATURE_SUPPORT_BOUND];
__constant__ PRECISION quadrature_kernel[QUADRATURE_SUPPORT_BOUND];

void nifty_gridding_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{
	printf("UPDATE >>> Executing Nifty Gridder as solver, Assuming host side setup in controller...\n\n");
	
	
    printf("UPDATE >>> Setting up NIFTY, Assuming host side setup in controller...\n\n");
	nifty_gridding_set_up(config, host, device);
	

    printf("UPDATE >>> Executing Nifty Gridder run...\n\n");
	//perform nifty - nifty_gridding_run, fft_run ect ect
	start_timer(&timings->solver);
	nifty_gridding_run(config,device);
	stop_timer(&timings->solver);

    if(config->enable_psf)
        psf_normalization_nifty(config, device);

	printf("UPDATE >>> Cleaning up NIFTY memory...\n\n");
	//clean up
	if(!config->retain_device_mem)
	{
		// Transfer device mem back to host (only required data, and for retain data flag)
		nifty_memory_transfer(config, host, device);
		nifty_clean_up(device);
	}
}

void nifty_degridding_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{   printf("UPDATE >>> Executing Nifty Gridder as PREDICT...\n\n");
    
    
    nifty_degridding_set_up(config, host, device);
	start_timer(&timings->predict);
    nifty_degridding_run(config,device);
	stop_timer(&timings->predict);

    if(!config->retain_device_mem)
    {	// Transfer device mem back to host (only required data, and for retain data flag)
        nifty_visibility_transfer(config, host, device);
        nifty_clean_up(device);
    }

}


void nifty_degridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	if(device->d_visibilities == NULL)
	{	printf("UPDATE >>> Copying predicted visibilities to device, number of visibilities: %d...\n\n",config->num_visibilities);
	    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_visibilities), sizeof(VIS_PRECISION2) * config->num_visibilities));
		CUDA_CHECK_RETURN(cudaMemset(device->d_visibilities, 0, config->num_visibilities * sizeof(VIS_PRECISION2)));
	    cudaDeviceSynchronize();
	}
	else
		CUDA_CHECK_RETURN(cudaMemset(device->d_visibilities, 0, config->num_visibilities * sizeof(VIS_PRECISION2)));
	
	if(device->d_vis_uvw_coords == NULL)
	{  	printf("UPDATE >>> Copying UVW coords: %d...\n\n",config->num_visibilities);
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_uvw_coords), sizeof(PRECISION3) * config->num_visibilities));
	    CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_uvw_coords, host->vis_uvw_coords, sizeof(PRECISION3) * config->num_visibilities,
        	cudaMemcpyHostToDevice));
    	cudaDeviceSynchronize();
	}
	if(device->d_w_grid_stack == NULL)
    {   printf("UPDATE >>> Allocating GRID STACK : %d square, %d num planes...\n\n", config->grid_size, config->nifty_config.num_w_grids_batched);
        uint32_t num_w_grid_stack_cells = config->grid_size * config->grid_size * config->nifty_config.num_w_grids_batched;
        CUDA_CHECK_RETURN(cudaMalloc(&(device->d_w_grid_stack), sizeof(PRECISION2) * num_w_grid_stack_cells));
        CUDA_CHECK_RETURN(cudaMemset(device->d_w_grid_stack, 0, num_w_grid_stack_cells * sizeof(PRECISION2)));
    }
	else
	{	uint32_t num_w_grid_stack_cells = config->grid_size * config->grid_size * config->nifty_config.num_w_grids_batched;
		CUDA_CHECK_RETURN(cudaMemset(device->d_w_grid_stack, 0, num_w_grid_stack_cells * sizeof(PRECISION2)));
	}
	if(device->d_vis_weights == NULL)
    {   printf("UPDATE >>> Allocating weights %d..\n\n", config->num_visibilities);
        CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_weights), sizeof(VIS_PRECISION) * config->num_visibilities));
        CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_weights, host->vis_weights, sizeof(VIS_PRECISION) * config->num_visibilities, cudaMemcpyHostToDevice));
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


void nifty_visibility_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	CUDA_CHECK_RETURN(cudaMemcpy(host->visibilities, device->d_visibilities, 
        config->num_visibilities * sizeof(VIS_PRECISION2), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();	
}


void execute_source_list_to_image(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{

    if(device->d_image != NULL)
    {  
        CUDA_CHECK_RETURN(cudaFree((*device).d_image));
    }
    //reset image
    int grid_square = config->image_size * config->image_size;
    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_image), sizeof(PRECISION) * grid_square));
    CUDA_CHECK_RETURN(cudaMemset(device->d_image, 0, grid_square * sizeof(PRECISION)));
    cudaDeviceSynchronize();

    if(device->d_sources == NULL)
    {
        printf("UPDATE >>> Allocating Device Source Buffer of size %d...\n\n", config->num_sources);
        CUDA_CHECK_RETURN(cudaMalloc(&(device->d_sources), sizeof(PRECISION3) * config->num_sources));
        if(config->num_sources > 0) // occurs only if has sources from previous major cycle
            CUDA_CHECK_RETURN(cudaMemcpy(device->d_sources, host->h_sources, sizeof(Source) * config->num_sources, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }


//RUN GPU KERNEL
    int max_threads_per_block_conversion = min(config->gpu_max_threads_per_block, config->num_sources);
    int num_blocks_conversion = (int) ceil((double) config->num_sources / max_threads_per_block_conversion);
    dim3 conversion_blocks(num_blocks_conversion, 1, 1);
    dim3 conversion_threads(config->gpu_max_threads_per_block, 1, 1);
    source_list_to_image<<<conversion_blocks, conversion_threads>>>(device->d_sources, config->num_sources,
                                                device->d_image, config->image_size, config->cell_size_rad);

    if(!config->retain_device_mem)
    {   //transfer memory then cleanup
        //if(host->dirty_image == NULL)
        //    printf("ERROR !!!! THE DIRTY IMAGE ON HOST IS NULL... \n\n");
        //int image_square = config->image_size * config->image_size;
     //   CUDA_CHECK_RETURN(cudaMemcpy(host->dirty_image, device->d_image, image_square * sizeof(PRECISION),
      //          cudaMemcpyDeviceToHost));
        
        CUDA_CHECK_RETURN(cudaMemcpy(host->h_sources, device->d_sources, config->num_sources * sizeof(PRECISION3),
                cudaMemcpyDeviceToHost));

        //device->d_image = NULL;
        device->d_sources = NULL;
    }
}

void nifty_degridding_run(Config *config, Device_Mem_Handles *device)
{
	int min_cuda_grid_size;
    int cuda_block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_cuda_grid_size, &cuda_block_size, nifty_gridding, 0, 0);
    int cuda_grid_size = (config->num_visibilities + cuda_block_size - 1) / cuda_block_size;  // create 1 thread per visibility to be gridded
    printf("UPDATE >>> Nifty De-Gridder using grid size %d where minimum grid size available is %d and using block size %d..\n\n", cuda_grid_size,min_cuda_grid_size,cuda_block_size);
    // Determine the block/thread distribution per w correction and summation batch
    // note: based on image size, not grid size, as we take inner region and discard padding
    dim3 w_correction_threads(
        min((uint32_t)32, (config->image_size + 1) / 2),
        min((uint32_t)32, (config->image_size + 1) / 2)
    );
    dim3 w_correction_blocks(
        (config->image_size/2 + 1 + w_correction_threads.x - 1) / w_correction_threads.x,  // allow extra in negative x quadrants, for asymmetric image centre
        (config->image_size/2 + 1 + w_correction_threads.y - 1) / w_correction_threads.y   // allow extra in negative y quadrants, for asymmetric image centre
    );

    // Determine how many w grid subset batches to process in total
    uint32_t total_w_grid_batches = (config->nifty_config.num_total_w_grids + config->nifty_config.num_w_grids_batched - 1) / config->nifty_config.num_w_grids_batched;
    uint32_t num_w_grid_stack_cells = config->grid_size * config->grid_size * config->nifty_config.num_w_grids_batched;

    dim3 scalar_threads(
        min((uint32_t)32, config->grid_size),
        min((uint32_t)32, config->grid_size)
    );
    dim3 scalar_blocks(
        (int)ceil((double)config->grid_size / scalar_threads.x), 
        (int)ceil((double)config->grid_size / scalar_threads.y)
    );

	// 1. Undo convolution correction and scaling
	PRECISION inv_w_range = 1.0 / (config->nifty_config.max_plane_w - config->nifty_config.min_plane_w);
	
	conv_corr_and_scaling<<<w_correction_blocks, w_correction_threads>>>(
		device->d_image,
		config->image_size,
		config->cell_size_rad,
		config->nifty_config.support,
		config->nifty_config.conv_corr_norm_factor,
		device->d_prolate,
		inv_w_range,
		config->visibility_scaled_weights_sum,
		PRECISION(1.0)/config->w_scale,
		false // predicting
	);
	CUDA_CHECK_RETURN( cudaGetLastError() );
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );


	for(int batch = 0; batch < total_w_grid_batches; batch++)
	{
		uint32_t num_w_grids_subset = min(
			config->nifty_config.num_w_grids_batched,
			config->nifty_config.num_total_w_grids - ((batch * config->nifty_config.num_w_grids_batched) % config->nifty_config.num_total_w_grids)
		);

		int32_t grid_start_w = batch*config->nifty_config.num_w_grids_batched;

		printf("Reversing...\n");

		// 2. Undo w-stacking and dirty image accumulation
		reverse_w_screen_to_stack<<<w_correction_blocks, w_correction_threads>>>(
			device->d_image,
			config->image_size,
			config->cell_size_rad,
			device->d_w_grid_stack,
			config->grid_size,
			grid_start_w,
			num_w_grids_subset,
			config->w_scale,
			config->nifty_config.min_plane_w,
			config->nifty_config.perform_shift_fft
		);

		CUDA_CHECK_RETURN(cudaGetLastError());
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		printf("Reversing FINISHED...\n");

		// 3. Batch FFT
		CUFFT_SAFE_CALL(CUFFT_EXECUTE_C2C(*(device->fft_plan), device->d_w_grid_stack, device->d_w_grid_stack, CUFFT_FORWARD));
		CUDA_CHECK_RETURN( cudaGetLastError() );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

		//  scale_for_FFT<<<scalar_blocks, scalar_threads>>>(
		//     device->d_w_grid_stack, 
		//     num_w_grids_subset, 
		//     config->grid_size, 
		//     1.0/(config->grid_size*config->grid_size)
		// );

		nifty_gridding<<<cuda_grid_size, cuda_block_size>>>(
			device->d_visibilities,
			device->d_vis_weights,
			device->d_vis_uvw_coords,
			config->num_visibilities,
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
			config->frequency_hz / PRECISION(SPEED_OF_LIGHT),
			config->enable_psf,
			config->nifty_config.perform_shift_fft,
			false // degridding
		);
		CUDA_CHECK_RETURN( cudaGetLastError() );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	
		// Reset intermediate w grid buffer for next batch
		printf("UPDATE >>> Resetting device w grid stack memory for next batch...\n");
		CUDA_CHECK_RETURN(cudaMemset(device->d_w_grid_stack, 0, num_w_grid_stack_cells * sizeof(PRECISION2)));
		CUDA_CHECK_RETURN( cudaGetLastError() );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	}
}

void nifty_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
    printf("UPDATE >>> Transfer dirty image to host.. \n\n");

    if(host->dirty_image == NULL)
        printf("ERROR !!!! THE DIRTY IMAGE ON HOST IS NULL... \n\n");
    int image_square = config->image_size * config->image_size;
    if(config->enable_psf)
    {   CUDA_CHECK_RETURN(cudaMemcpy(host->h_psf, device->d_psf, image_square * sizeof(PRECISION),
            cudaMemcpyDeviceToHost));
    }
    else
    {   CUDA_CHECK_RETURN(cudaMemcpy(host->dirty_image, device->d_image, image_square * sizeof(PRECISION),
            cudaMemcpyDeviceToHost));
    }
	cudaDeviceSynchronize();
}

void nifty_clean_up(Device_Mem_Handles *device)
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



void nifty_gridding_run(Config *config, Device_Mem_Handles *device)
{
	int min_cuda_grid_size;
    int cuda_block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_cuda_grid_size, &cuda_block_size, nifty_gridding, 0, 0);
    int cuda_grid_size = (config->num_visibilities + cuda_block_size - 1) / cuda_block_size;  // create 1 thread per visibility to be gridded
    printf("UPDATE >>> Nifty Gridder using grid size %d where minimum grid size available is %d and using block size %d..\n\n", cuda_grid_size,min_cuda_grid_size,cuda_block_size);


    // Determine the block/thread distribution per w correction and summation batch
    // note: based on image size, not grid size, as we take inner region and discard padding
    dim3 w_correction_threads(
        min((uint32_t)32, (config->image_size + 1) / 2),
        min((uint32_t)32, (config->image_size + 1) / 2)
    );
    dim3 w_correction_blocks(
        (config->image_size/2 + 1 + w_correction_threads.x - 1) / w_correction_threads.x,  // allow extra in negative x quadrants, for asymmetric image centre
        (config->image_size/2 + 1 + w_correction_threads.y - 1) / w_correction_threads.y   // allow extra in negative y quadrants, for asymmetric image centre
    );

    // Determine how many w grid subset batches to process in total
    uint32_t total_w_grid_batches = (config->nifty_config.num_total_w_grids + config->nifty_config.num_w_grids_batched - 1) / config->nifty_config.num_w_grids_batched;
    uint32_t num_w_grid_stack_cells = config->grid_size * config->grid_size * config->nifty_config.num_w_grids_batched;

    for(int batch = 0; batch < total_w_grid_batches; batch++)
    {
        uint32_t num_w_grids_subset = min(
            config->nifty_config.num_w_grids_batched,
            config->nifty_config.num_total_w_grids - ((batch * config->nifty_config.num_w_grids_batched) % config->nifty_config.num_total_w_grids)
        );

        int32_t grid_start_w = batch*config->nifty_config.num_w_grids_batched;

        printf("Gridder calling nifty_gridding kernel for w grids %d to %d (%d planes in current batch)", 
				grid_start_w, (grid_start_w + (int32_t)num_w_grids_subset - 1), num_w_grids_subset);
        
        // Perform gridding on a "chunk" of w grids
        nifty_gridding<<<cuda_grid_size, cuda_block_size>>>(
            device->d_visibilities,
            device->d_vis_weights,
            device->d_vis_uvw_coords,
            config->num_visibilities,
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
            config->frequency_hz / PRECISION(SPEED_OF_LIGHT),
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
            (config->enable_psf) ? device->d_psf : device->d_image,
            config->image_size,
            config->cell_size_rad,
            device->d_w_grid_stack,
            config->grid_size,
            grid_start_w,
            num_w_grids_subset,
            PRECISION(1.0)/config->w_scale,
            config->nifty_config.min_plane_w,
            config->nifty_config.perform_shift_fft
        );

		// Copy back images, save to file  (debugging)
		// copy_dirty_image_to_host(host, device, config);
		// save_numbered_dirty_image_to_file(host, config, batch);

        // Reset intermediate w grid buffer for next batch
        //if(batch < total_w_grid_batches - 1)
       // {
            printf("UPDATE >>> Resetting device w grid stack memory for next batch..\n\n");
            CUDA_CHECK_RETURN(cudaMemset(device->d_w_grid_stack, 0, num_w_grid_stack_cells * sizeof(PRECISION2)));
        //}
    }

    // Need to determine final scaling factor for scaling dirty image by w grid accumulation
    PRECISION inv_w_range = 1.0 / (config->nifty_config.max_plane_w - config->nifty_config.min_plane_w);

    // Perform convolution correction and final scaling on single real plane
    // note: can recycle same block/thread dims as w correction kernel
    conv_corr_and_scaling<<<w_correction_blocks, w_correction_threads>>>(
        (config->enable_psf) ? device->d_psf : device->d_image,
        config->image_size,
        config->cell_size_rad,
        config->nifty_config.support,
        config->nifty_config.conv_corr_norm_factor,
        device->d_prolate,
        inv_w_range,
        config->visibility_scaled_weights_sum,
		PRECISION(1.0)/config->w_scale,
		true
    );
}

void psf_normalization_nifty(Config *config, Device_Mem_Handles *device)
{
    PRECISION* d_max_psf_found;
    PRECISION max_psf_found;
    
    CUDA_CHECK_RETURN(cudaMalloc(&d_max_psf_found, sizeof(PRECISION)));
    
    find_psf_nifty_max<<<1,1>>>(d_max_psf_found, device->d_psf, config->image_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&max_psf_found, d_max_psf_found, sizeof(PRECISION), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("UPDATE >>> FOUND PSF MAX OF %f ",max_psf_found);
    cudaFree(d_max_psf_found);
    
    int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->image_size);
    int num_blocks_per_dimension = (int) ceil((double) config->image_size / max_threads_per_block_dimension);
    dim3 blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
    dim3 threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);
    
    //USE RECIPROCAL???
    psf_normalization_nifty_kernel<<<blocks, threads>>>(max_psf_found, device->d_psf, config->image_size);
    cudaDeviceSynchronize();
    
    config->psf_max_value = max_psf_found;
}


void nifty_psf_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timers)
{
	nifty_gridding_execute(config,host,device, timers);
}

void nifty_host_side_setup(Config *config, Host_Mem_Handles *host)
{   
    printf("UPDATE >>> setting up Nifty specific host side parameters .. \n\n");
	//uint32_t num_dirty_image_pixels = config->image_size * config->image_size;
	
    //CUDA_CHECK_RETURN(cudaHostAlloc(&(host->dirty_image), num_dirty_image_pixels * sizeof(PRECISION), cudaHostAllocDefault));
   // host->dirty_image   = (PRECISION*) calloc(num_dirty_image_pixels, sizeof(PRECISION));

    //CUDA_CHECK_RETURN(cudaHostAlloc(&(host->prolate), (config->image_size/2 + 1) * sizeof(PRECISION), cudaHostAllocDefault));
    host->prolate = (PRECISION*) calloc(config->image_size/2 + 1, sizeof(PRECISION));
    host->quadrature_nodes   = (PRECISION*) calloc(QUADRATURE_SUPPORT_BOUND, sizeof(PRECISION));
    host->quadrature_weights = (PRECISION*) calloc(QUADRATURE_SUPPORT_BOUND, sizeof(PRECISION));
    host->quadrature_kernel  = (PRECISION*) calloc(QUADRATURE_SUPPORT_BOUND, sizeof(PRECISION));
	
    printf("UPDATE >>> Generate the Gauss-Legendre kernels ...\n\n");
	generate_gauss_legendre_conv_kernel(host, config);
    // printf("UPDATE >>> Host prolate at 0 %f ... \n\n", host->prolate[0]);
}

void nifty_gridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{		
	if(device->d_visibilities == NULL && !config->enable_psf)
	{	printf("UPDATE >>> Copying predicted visibilities to device, number of visibilities: %d...\n\n",config->num_visibilities);
	    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_visibilities), sizeof(VIS_PRECISION2) * config->num_visibilities));
	    CUDA_CHECK_RETURN(cudaMemcpy(device->d_visibilities, host->visibilities, sizeof(VIS_PRECISION2) * config->num_visibilities,
	        cudaMemcpyHostToDevice));
	    cudaDeviceSynchronize();
	}
	if(device->d_vis_uvw_coords == NULL)
	{  printf("UPDATE >>> Copying UVW coords: %d...\n\n",config->num_visibilities);
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_uvw_coords), sizeof(PRECISION3) * config->num_visibilities));
	    CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_uvw_coords, host->vis_uvw_coords, sizeof(PRECISION3) * config->num_visibilities,
        	cudaMemcpyHostToDevice));
    	cudaDeviceSynchronize();
	}
	if(!config->enable_psf && device->d_image == NULL)
	{  printf("UPDATE >>> Allocating IMAGE ON DEVICE: %d...\n\n", config->image_size);
	    int grid_square = config->image_size * config->image_size;
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_image), sizeof(PRECISION) * grid_square));
    	CUDA_CHECK_RETURN(cudaMemset(device->d_image, 0, grid_square * sizeof(PRECISION)));
    	cudaDeviceSynchronize();
	}
	if(config->enable_psf && device->d_psf == NULL)
	{  printf("UPDATE >>> Allocating PSF image: %d...\n\n", config->image_size);
	    // Bind prolate spheroidal to gpu
    	int grid_square = config->image_size * config->image_size;
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_psf), sizeof(PRECISION) * grid_square));
    	CUDA_CHECK_RETURN(cudaMemset(device->d_psf, 0, grid_square * sizeof(PRECISION)));
    	cudaDeviceSynchronize();
	}
	if(device->d_w_grid_stack == NULL)
    {   printf("UPDATE >>> Allocating GRID STACK : %d square, %d num planes...\n\n", config->grid_size, config->nifty_config.num_w_grids_batched);
        uint32_t num_w_grid_stack_cells = config->grid_size * config->grid_size * config->nifty_config.num_w_grids_batched;
        CUDA_CHECK_RETURN(cudaMalloc(&(device->d_w_grid_stack), sizeof(PRECISION2) * num_w_grid_stack_cells));
        CUDA_CHECK_RETURN(cudaMemset(device->d_w_grid_stack, 0, num_w_grid_stack_cells * sizeof(PRECISION2)));
    }
	if(device->d_vis_weights == NULL)
    {   printf("UPDATE >>> Allocating weights %d..\n\n", config->num_visibilities);
        CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_weights), sizeof(VIS_PRECISION) * config->num_visibilities));
        CUDA_CHECK_RETURN(cudaMemcpy(device->d_vis_weights, host->vis_weights, sizeof(VIS_PRECISION) * config->num_visibilities, cudaMemcpyHostToDevice));
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

void generate_gauss_legendre_conv_kernel(Host_Mem_Handles *host, Config *config)
{
    // p based on formula in first paragraph under equation 3.10, page 8 of paper:
    // A parallel non-uniform fast Fourier transform library based on an "exponential of semicircle" kernel
    uint32_t p = (uint32_t)CEIL(1.5 * config->nifty_config.support + 2.0);
    uint32_t n = (uint32_t)(p * 2);

    for (uint32_t i=1; i<=n/2; i++)
    {
        double w_i = 0.0;
        double x_i = calculate_legendre_root((int32_t)i, (int32_t)n, 1e-16, &w_i);
        double k_i = exp(config->nifty_config.beta*(sqrt(1.0 - x_i*x_i) - 1.0)); 
        host->quadrature_nodes[i-1] = (PRECISION)x_i;
        host->quadrature_weights[i-1] = (PRECISION)w_i;
        host->quadrature_kernel[i-1] = (PRECISION)k_i;
    }

    // Need to determine normalisation factor for scaling runtime calculated
    // conv correction values for coordinate n (where n = sqrt(1 - l^2 - m^2) - 1)
    double conv_corr_norm_factor = 0.0;
    for(uint32_t i = 0; i < p; i++)
        conv_corr_norm_factor += host->quadrature_kernel[i]*host->quadrature_weights[i];
    conv_corr_norm_factor *= (double)config->nifty_config.support;
    config->nifty_config.conv_corr_norm_factor  = conv_corr_norm_factor;

    // Precalculate one side of convolutional correction kernel out to one more than half the image size
    for(uint32_t l_m = 0; l_m <= config->image_size/2; l_m++)
    {
        double l_m_norm =  ((double)l_m)*(1.0/config->grid_size); // between 0.0 .. 0.5
        double correction = 0.0;

        for(uint32_t i = 0; i < p; i++)
            correction += host->quadrature_kernel[i]*cos(PI*l_m_norm*((double)config->nifty_config.support)*host->quadrature_nodes[i])*host->quadrature_weights[i];
        host->prolate[l_m] = (PRECISION)(correction*((double)config->nifty_config.support)) / conv_corr_norm_factor;

    } 
}

/**********************************************************************
* Calculate the n-th Legendre Polynomial P_n at x
* optionally also calculates the (first) derivate P_n' at x when -1<x<1
**********************************************************************/

double get_legendre(double x, int n, double *derivative)
{
    if (n==0)
    {
        if (derivative != NULL)
             *derivative = 0.0;
        return 1.0;
    }
    else if (n==1)
    {
        if (derivative != NULL)
             *derivative = 1.0;
        return x;
    }
    // else n>=2

    // recursively calculate P_n(x) = ((2n-1)/n)xP_{n-1}(x) - ((n-1)/n)P_{n-2}(x)
    // note this is same as P_n(x) = xP_{n-1}(x) + ((n-1)/n)(xP_{n-1}(x)-P_{n-2}(x))
    double p_im2 = 1.0; // P_{i-2}(x)
    double p_im1 = x; // P_{i-1}(x)
    double p_i;

    for (int32_t i=2; i<=n; i++)
    {
        p_i = x*p_im1 + (((double)i-1.0)/(double)i)*(x*p_im1-p_im2);
        // prepare for next iteration if required
        p_im2 = p_im1;
        p_im1 = p_i;
    }

    // optionally calculate the derivate of P_n at x
    // note this is given by P_n'(x) = n/(1-x*x)(P_{n-1}(x)-x*P_n(x))
    if (derivative != NULL && x>-1.0 && x<1.0)
    {
        *derivative = (double)n/(1.0-(x*x))*(p_im2-x*p_i); // note p_im2 currently holds P_{n-1}(x)
    }
    return p_i;
}
/**********************************************************************
* Calculate an initial estimate of i-th root of n-th Legendre Polynomial
* where i=1,2,...,n ordered from largest root to smallest
* Note this follows the approach using in equation 13 of the paper
* "Sugli zeri dei polinomi sferici ed ultrasferici" by Francesco G. Tricomi
* Annali di Matematica Pura ed Applicata volume 31, pages93â€“97 (1950)
**********************************************************************/
double get_approx_legendre_root(int32_t i, int32_t n)
{
    double improvement = 1.0 - (1.0-1.0/(double)n)/(8.0*(double)n*(double)n);
    double estimate = cos(M_PI*(4.0*(double)i-1.0)/(4.0*(double)n+2.0));
    return improvement*estimate;
}

/**********************************************************************
* Calculates an estimate of i-th root of n-th Legendre Polynomial
* where i=1,2,...,n ordered from largest root to smallest
* using Newton-Ralphson iterations until the specified
* accuracy is reached and then also calculate its weight for Gauss-Legendre quadrature
**********************************************************************/
double calculate_legendre_root(int32_t i, int32_t n, double accuracy, double *weight)
{
    double next_estimate = get_approx_legendre_root(i, n);
    double derivative;
    double estimate;
    int32_t iterations = 0;
    do
    {
        estimate = next_estimate;
        double p_n = get_legendre(estimate, n, &derivative);
        next_estimate = estimate - p_n/derivative;
        iterations++;
    }
    while (fdim(next_estimate,estimate)>accuracy && iterations<MAX_NEWTON_RAPHSON_ITERATIONS);

    // Gauss-Legendre quadrature weight for x is given by w = 2/((1-x*x)P_n'(x)*P_n'(x))
    double p_n = get_legendre(next_estimate, n, &derivative);
    *weight = 2.0/((1.0-next_estimate*next_estimate)*derivative*derivative);
    return next_estimate;
}


 /**********************************************************************
* Evaluates the convolutional correction C(k) in one dimension.
* As the exponential of semicircle gridding kernel psi(u,v,w) is separable its Fourier
* transform Fourier(psi)(l,m,n) is likewise separable into one-dimensional components C(l)C(m)C(n).
* As psi is even and zero outside its support the convolutional correction is given by:
*   C(k) = 2\integral_{0}^{supp/2} psi(u)cos(2\pi ku) du = supp\integral_{0}^{1}psi(supp*x)cos(\pi*k*supp*x) dx by change of variables
* This integral from 0 to 1 can be numerically approximated via a 2p-node Gauss-Legendre quadrature, as recommended 
* in equation 3.10 of ‘A Parallel Non-uniform Fast Fourier Transform Library Based on an “Exponential of Semicircle” Kernel’
* by Barnett, Magland, Klintenberg and only using the p positive nodes):
*   C(k) ~ Sum_{i=1}^{p} weight_i*psi(supp*node_i)*cos(\pi*k*supp*node_i)
* Note this convolutional correction is not normalised, 
* but is normalised during use in convolution correction by C(0) to get max of 1
**********************************************************************/
__device__ PRECISION conv_corr(PRECISION support, PRECISION k)
{
    PRECISION correction = 0.0;
    uint32_t p = (uint32_t)(CEIL(PRECISION(1.5)*support + PRECISION(2.0)));

    for(uint32_t i = 0; i < p; i++)
        correction += quadrature_kernel[i]*COS(PI*k*support*quadrature_nodes[i])*quadrature_weights[i];

    return correction*support;
}

/**********************************************************************
 * Calculates the exponential of semicircle
 * Note the parameter x must be normalised to be in range [-1,1]
 * Source Paper: A parallel non-uniform fast Fourier transform library based on an "exponential of semicircle" kernel
 * Address: https://arxiv.org/abs/1808.06736
 **********************************************************************/
__device__ PRECISION exp_semicircle(PRECISION beta, PRECISION x)
{
	PRECISION xx = x*x;

    if (xx > 1.0)  // to ensure we don't try to take the square root of a negative number
        return (0.0);
    else
		return VEXP(beta*(VSQRT(1.0 - xx) - 1.0));
        //return (VEXP(beta*(VSQRT(VIS_PRECISION(1.0) - (xx)) - VIS_PRECISION(1.0))));
} 

/**********************************************************************
 * Calculates complex phase shift for applying to each
 * w layer (note: w layer = iFFT(w grid))
 * Note: l and m are expected to be in the range  [-0.5, 0.5]
 **********************************************************************/
__device__ PRECISION2 phase_shift(PRECISION w, PRECISION l, PRECISION m, PRECISION signage)
{
    // calc the sum of squares
    PRECISION sos = l*l + m*m;
    PRECISION nm1 = (-sos)/(SQRT(PRECISION(1.0)-sos) + PRECISION(1.0));
    PRECISION x = PRECISION(2.0)*PI*w*(nm1);
    PRECISION xn = PRECISION(1.0)/(nm1 + PRECISION(1.0));
    PRECISION sinx;
    PRECISION cosx;
    // signage = -1.0 if solving, 1.0 if predicting
    SINCOS(x*signage, &sinx, &cosx); // calculates both sin and cos of x at same time (efficient)
    return (MAKE_PRECISION2(cosx*xn, sinx*xn));
} 

/**********************************************************************
 * Performs the gridding (or degridding) of visibilities across a subset of w planes
 * Parallelised so each CUDA thread processes a single visibility
 **********************************************************************/
 __global__ void nifty_gridding(
    
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
    const PRECISION metres_wavelength_scale, // for w coordinate
    const bool generating_psf, // flag for enabling/disabling creation of PSF using same gridding code
    const bool perform_shift_fft, // flag to (equivalently) rearrange each grid so origin is at lower-left corner for FFT
    const bool solving // flag to enable degridding operations instead of gridding
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < (int32_t)num_visibilities)
    {
        const PRECISION half_support = PRECISION(support)/2.0; // NOTE confirm everyone's understanding of what support means eg when even/odd
        const int32_t grid_min_uv = -(int32_t)grid_size/2; // minimum coordinate on grid along u or v axes
        const int32_t grid_max_uv = ((int32_t)grid_size-1)/2; // maximum coordinate on grid along u or v axes

        // Determine whether to flip visibility coordinates, so w is usually positive
        VIS_PRECISION flip = (uvw_coords[i].z < 0.0) ? -1.0 : 1.0; 

        // Calculate bounds of where gridding kernel will be applied for this visibility
        PRECISION3 uvw_coord = MAKE_PRECISION3(
            uvw_coords[i].x * uv_scale * (PRECISION)flip,
            uvw_coords[i].y * uv_scale * (PRECISION)flip,
            (uvw_coords[i].z * (PRECISION)flip * metres_wavelength_scale - min_plane_w) * w_scale
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
            vis_weighted = (generating_psf) ? MAKE_VIS_PRECISION2(1.0, 0.0) : visibilities[i];
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
        {
            visibilities[i].x += vis_weighted.x;
            visibilities[i].y += vis_weighted.y * flip;
        }
    }
}



__global__ void scale_for_FFT(PRECISION2 *w_grid_stack, const int num_w_planes, const int grid_size, const PRECISION scalar)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= grid_size || y >= grid_size)
        return;

    for (int32_t w_layer=0; w_layer < num_w_planes; w_layer++)
    {   uint32_t w_index = w_layer*grid_size*grid_size;

        uint32_t index = w_index + y*grid_size + x;
        w_grid_stack[index].x *= scalar;
        w_grid_stack[index].y *= scalar;
    }

}

/**********************************************************************
 * Applies the w screen phase shift and accumulation of each w layer onto dirty image
 * Parallelised so each CUDA thread processes one pixel in each quadrant of the dirty image
 **********************************************************************/
__global__ void apply_w_screen_and_sum(
    PRECISION *dirty_image, // INPUT & OUTPUT: real plane for accumulating phase corrected w layers across batches
    const uint32_t image_size, // one dimensional size of image plane (grid_size / sigma), assumed square
    const PRECISION pixel_size, // converts pixel index (x, y) to normalised image coordinate (l, m) where l, m between -0.5 and 0.5
    const PRECISION2 *w_grid_stack, // INPUT: flat array containing 2D computed w layers (w layer = iFFT(w grid))
    const uint32_t grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
    const int32_t grid_start_w, // index of first w grid in current subset stack
    const uint32_t num_w_grids_subset, // number of w grids bound in current subset stack
    const PRECISION inv_w_scale, // scaling factor for converting w coord to signed w grid index
    const PRECISION min_plane_w, // w coordinate of smallest w plane
    const bool perform_shift_fft  // flag to (equivalently) rearrange each grid so origin is at lower-left corner for FFT
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t half_image_size = image_size / 2;
    
    if(i <= (int32_t)half_image_size && j <= (int32_t)half_image_size)  // allow extra in negative x and y directions, for asymmetric image centre
    {
        // Init pixel sums for the four quadrants
        PRECISION pixel_sum_pos_pos = 0.0;
        PRECISION pixel_sum_pos_neg = 0.0;
        PRECISION pixel_sum_neg_pos = 0.0;
        PRECISION pixel_sum_neg_neg = 0.0;

        const int32_t origin_offset_grid_centre = (int32_t)(grid_size/2); // offset of origin (in w layer) along l or m axes
        const int32_t grid_index_offset_image_centre = origin_offset_grid_centre*((int32_t)grid_size) + origin_offset_grid_centre;

        for (int32_t grid_coord_w=grid_start_w; grid_coord_w < grid_start_w + (int32_t)num_w_grids_subset; grid_coord_w++)
        {
            PRECISION l = pixel_size * (PRECISION)i;
            PRECISION m = pixel_size * (PRECISION)j;
            PRECISION w = (PRECISION)grid_coord_w * inv_w_scale + min_plane_w; 
            PRECISION2 shift = phase_shift(w, l, m, PRECISION(-1.0));
            
            int32_t grid_index_offset_w = (grid_coord_w-grid_start_w)*((int32_t)(grid_size*grid_size));
            int32_t grid_index_image_centre = grid_index_offset_w + grid_index_offset_image_centre;
            
            // Calculate the real component of the complex w layer value multiplied by the complex phase shift
            // Note w_grid_stack presumed to be larger than dirty_image (sigma > 1) so has extra pixels around boundary
            PRECISION2 w_layer_pos_pos = w_grid_stack[grid_index_image_centre + j*((int32_t)grid_size) + i];
            pixel_sum_pos_pos += w_layer_pos_pos.x*shift.x - w_layer_pos_pos.y*shift.y;
            PRECISION2 w_layer_pos_neg = w_grid_stack[grid_index_image_centre - j*((int32_t)grid_size) + i];
            pixel_sum_pos_neg += w_layer_pos_neg.x*shift.x - w_layer_pos_neg.y*shift.y;
            PRECISION2 w_layer_neg_pos = w_grid_stack[grid_index_image_centre + j*((int32_t)grid_size) - i];
            pixel_sum_neg_pos += w_layer_neg_pos.x*shift.x - w_layer_neg_pos.y*shift.y;
            PRECISION2 w_layer_neg_neg = w_grid_stack[grid_index_image_centre - j*((int32_t)grid_size) - i];
            pixel_sum_neg_neg += w_layer_neg_neg.x*shift.x - w_layer_neg_neg.y*shift.y;
        }

        // Equivalently rearrange each grid so origin is at lower-left corner for FFT
        bool odd_grid_coordinate = ((i+j) & (int32_t)1) != (int32_t)0;
        if(perform_shift_fft && odd_grid_coordinate)
        {
            pixel_sum_pos_pos = -pixel_sum_pos_pos;
            pixel_sum_pos_neg = -pixel_sum_pos_neg;
            pixel_sum_neg_pos = -pixel_sum_neg_pos;
            pixel_sum_neg_neg = -pixel_sum_neg_neg;
        }

        // Add the four pixel sums to the dirty image taking care to be within bounds for positive x and y quadrants
        const int32_t origin_offset_image_centre = (int32_t)half_image_size; // offset of origin (in dirty image) along l or m axes
        const int32_t image_index_offset_image_centre = origin_offset_image_centre*((int32_t)image_size) + origin_offset_image_centre;
        // Special cases along centre or edges of image
        if(i < (int32_t)half_image_size && j < (int32_t)half_image_size)
            dirty_image[image_index_offset_image_centre + j*((int32_t)image_size) + i] += pixel_sum_pos_pos;
        if(i > 0 && j < (int32_t)half_image_size)
            dirty_image[image_index_offset_image_centre + j*((int32_t)image_size) - i] += pixel_sum_neg_pos;
        if(j > 0 && i < (int32_t)half_image_size)
            dirty_image[image_index_offset_image_centre - j*((int32_t)image_size) + i] += pixel_sum_pos_neg;
        if(i > 0 && j > 0)
            dirty_image[image_index_offset_image_centre - j*((int32_t)image_size) - i] += pixel_sum_neg_neg;
    }
}


/**********************************************************************
 * Reverses w screen phase shift for each w layer from dirty image
 * Parallelised so each CUDA thread processes one pixel in each quadrant of the dirty image
 **********************************************************************/
__global__ void reverse_w_screen_to_stack(
    const PRECISION *dirty_image, // INPUT: real plane for input dirty image
    const uint32_t image_size, // one dimensional size of image plane (grid_size / sigma), assumed square
    const PRECISION pixel_size, // converts pixel index (x, y) to normalised image coordinate (l, m) where l, m between -0.5 and 0.5
    PRECISION2 *w_grid_stack, // OUTPUT: flat array containing 2D computed w layers (w layer = iFFT(w grid))
    const uint32_t grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
    const int32_t grid_start_w, // index of first w grid in current subset stack
    const uint32_t num_w_grids_subset, // number of w grids bound in current subset stack
    const PRECISION w_scale, // scaling factor for converting w coord to signed w grid index
    const PRECISION min_plane_w, // w coordinate of smallest w plane
    const bool perform_shift_fft  // flag to (equivalently) rearrange each grid so origin is at lower-left corner for FFT
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t half_image_size = image_size / 2;

    if(i <= (int32_t)half_image_size && j <= (int32_t)half_image_size)  // allow extra in negative x and y directions, for asymmetric image centre
    {
        // Obtain four pixels from the dirty image, taking care to be within bounds for positive x and y quadrants
        const int32_t origin_offset_image_centre = (int32_t)half_image_size; // offset of origin (in dirty image) along l or m axes
        const int32_t image_index_offset_image_centre = origin_offset_image_centre*((int32_t)image_size) + origin_offset_image_centre;

        // Look up dirty pixels in four quadrants
        // TODO: Check bounds for i,j
        // BUG: Seems these reads are going out of bounds, but oddly only for double??
        PRECISION dirty_image_pos_pos = PRECISION(0.0);
        PRECISION dirty_image_neg_pos = PRECISION(0.0);
        PRECISION dirty_image_pos_neg = PRECISION(0.0);
        PRECISION dirty_image_neg_neg = PRECISION(0.0);

        if(j < (int32_t)half_image_size && i < (int32_t)half_image_size)
            dirty_image_pos_pos = dirty_image[image_index_offset_image_centre + j*((int32_t)image_size) + i];
        if(j < (int32_t)half_image_size)
            dirty_image_neg_pos = dirty_image[image_index_offset_image_centre + j*((int32_t)image_size) - i];
        if(i < (int32_t)half_image_size)
            dirty_image_pos_neg = dirty_image[image_index_offset_image_centre - j*((int32_t)image_size) + i];
        
        dirty_image_neg_neg = dirty_image[image_index_offset_image_centre - j*((int32_t)image_size) - i];

        // Equivalently rearrange each grid so origin is at lower-left corner for FFT
        bool odd_grid_coordinate = ((i+j) & (int32_t)1) != (int32_t)0;
        if(perform_shift_fft && odd_grid_coordinate)
        {
            dirty_image_pos_pos = -dirty_image_pos_pos;
            dirty_image_pos_neg = -dirty_image_pos_neg;
            dirty_image_neg_pos = -dirty_image_neg_pos;
            dirty_image_neg_neg = -dirty_image_neg_neg;
        }

        const int32_t origin_offset_grid_centre = (int32_t)(grid_size/2); // offset of origin (in w layer) along l or m axes
        const int32_t grid_index_offset_image_centre = origin_offset_grid_centre*((int32_t)grid_size) + origin_offset_grid_centre;

        for (int32_t grid_coord_w=grid_start_w; grid_coord_w < grid_start_w + (int32_t)num_w_grids_subset; grid_coord_w++)
        {
            PRECISION l = pixel_size * (PRECISION)i;
            PRECISION m = pixel_size * (PRECISION)j;
            PRECISION w = (PRECISION)grid_coord_w / w_scale + min_plane_w; 
            PRECISION2 shift = phase_shift(w, l, m, PRECISION(1.0));
            //shift.y = -shift.y; // inverse of original phase shift (equivalent to division)
            
            int32_t grid_index_offset_w = (grid_coord_w-grid_start_w)*((int32_t)(grid_size*grid_size));
            int32_t grid_index_image_centre = grid_index_offset_w + grid_index_offset_image_centre;
            
            // Calculate the complex product of the (real) dirty image by the complex phase shift
            // Special cases along centre or edges of image
            if(i < (int32_t)half_image_size && j < (int32_t)half_image_size)
                w_grid_stack[grid_index_image_centre + j*((int32_t)grid_size) + i] = MAKE_PRECISION2(shift.x*dirty_image_pos_pos, shift.y*dirty_image_pos_pos);
            if(j > 0 && i < (int32_t)half_image_size)
                w_grid_stack[grid_index_image_centre - j*((int32_t)grid_size) + i] = MAKE_PRECISION2(shift.x*dirty_image_pos_neg, shift.y*dirty_image_pos_neg);                
            if(i > 0 && j < (int32_t)half_image_size)
                w_grid_stack[grid_index_image_centre + j*((int32_t)grid_size) - i] = MAKE_PRECISION2(shift.x*dirty_image_neg_pos, shift.y*dirty_image_neg_pos);
            if(i > 0 && j > 0)
                w_grid_stack[grid_index_image_centre - j*((int32_t)grid_size) - i] = MAKE_PRECISION2(shift.x*dirty_image_neg_neg, shift.y*dirty_image_neg_neg);
        }
    }
}

/**********************************************************************
 * Performs convolution correction and final scaling of dirty image
 * using precalculated and runtime calculated correction values.
 * See conv_corr device function for more details
 * Note precalculated convolutional correction for (l, m) are normalised to max of 1,
 * but value for n is calculated at runtime, therefore normalised at runtime by C(0)
 **********************************************************************/
__global__ void conv_corr_and_scaling(
    PRECISION *dirty_image,
    const uint32_t image_size,
    const PRECISION pixel_size,
    const uint32_t support,
    const PRECISION conv_corr_norm_factor,
    const PRECISION *conv_corr_kernel,
    const PRECISION inv_w_range,
    const PRECISION weight_channel_product,
    const PRECISION inv_w_scale,
    const bool solving
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t half_image_size = image_size / 2;

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
        
        if(solving)
            correction = PRECISION(1.0)/(correction*weight_channel_product);
            //correction = PRECISION(1.0)/(correction);
        else
            correction = PRECISION(1.0)/(correction);
            //correction = PRECISION(1.0)/conv_corr((PRECISION)support, n*inv_w_scale); 

        // Going to need offsets to stride from pixel to pixel for this thread
        const int32_t origin_offset_image_centre = (int32_t)half_image_size; // offset of origin (in dirty image) along l or m axes
        const int32_t image_index_offset_image_centre = origin_offset_image_centre*((int32_t)image_size) + origin_offset_image_centre;
        
        if(i < (int32_t)half_image_size && j < (int32_t)half_image_size)
        {
            dirty_image[image_index_offset_image_centre + j*((int32_t)image_size) + i] *= correction; 
        }
        // Special cases along centre of image doesn't update four pixels
        if(i > 0 && j < (int32_t)half_image_size)
        {
            dirty_image[image_index_offset_image_centre + j*((int32_t)image_size) - i] *= correction; 
        }
        if(j > 0 && i < (int32_t)half_image_size)
        {
            dirty_image[image_index_offset_image_centre - j*((int32_t)image_size) + i] *= correction; 
        }
        if(i > 0 && j > 0)
        {
            dirty_image[image_index_offset_image_centre - j*((int32_t)image_size) - i] *= correction; 
        }
    }
}

__global__ void find_psf_nifty_max(PRECISION *max_psf, const PRECISION *psf, const int grid_size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= 1)
        return;
    
    int grid_index = grid_size * (grid_size /2) + (grid_size/2);
    *max_psf = psf[grid_index];
    
}

__global__ void psf_normalization_nifty_kernel(PRECISION max_psf, PRECISION *psf, const int image_size)
{
    const int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    const int col_index = threadIdx.x + blockDim.x * blockIdx.x;

    if(row_index >= image_size || col_index >= image_size)
        return;

    const int image_index = row_index * image_size + col_index;
   
    psf[image_index] /= max_psf;
    
}

 __global__ void source_list_to_image(const PRECISION3 *sources, const int num_sources, 
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

