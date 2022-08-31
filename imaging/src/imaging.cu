
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

#include "../imaging.h"
#include "../controller.h"
#include "../restorer.h"

/*
	Temporarily executing msmfs solver - This is just a placeholder and cannot major cycle nor clean at this stage
*/
int execute_imaging_pipeline_msmfs(Config *config, Host_Mem_Handles *host_mem)
{
	cudaDeviceReset();
	
	Device_Mem_Handles device_mem;
	init_gpu_mem(&device_mem);
	
	Timing timers;
	init_timers(&timers);
	
	//Generate  PSFs for MSMFS cleaning - added later 
	
#if SOLVER == NIFTY_GRIDDING
	int  taylor = (2 * config->mf_num_moments) - 1;
	config->enable_psf = true;
	msmfs_nifty_gridding_execute(config, host_mem, &device_mem, &timers);
	host_mem->h_psf = (PRECISION*) calloc(taylor*config->psf_size*config->psf_size, sizeof(PRECISION));
	pull_msmfs_image_cube(config, host_mem, &device_mem);
	save_taylor_term_planes(config, host_mem->h_psf, taylor, config->psf_size);
	msmfs_nifty_clean_up(&device_mem);
	
	config->enable_psf = false;
	taylor = config->mf_num_moments;
	gains_apply_execute(config, host_mem, &device_mem, &timers);
	msmfs_nifty_gridding_execute(config, host_mem, &device_mem, &timers);
	host_mem->dirty_image = (PRECISION*) calloc(taylor*config->image_size*config->image_size, sizeof(PRECISION));
	pull_msmfs_image_cube(config, host_mem, &device_mem);
	save_taylor_term_planes(config, host_mem->dirty_image, taylor, config->image_size);
	msmfs_nifty_clean_up(&device_mem);
#else 
	int  taylor = (2 * config->mf_num_moments) - 1;
	config->enable_psf = true;
	gridding_execute_msmfs(config,host_mem,&device_mem, &timers);
	host_mem->h_psf = (PRECISION*) calloc(taylor*config->psf_size*config->psf_size, sizeof(PRECISION));
	pull_msmfs_image_cube(config, host_mem, &device_mem);
	save_taylor_term_planes(config, host_mem->h_psf, taylor, config->psf_size);
	msmfs_gridding_clean_up(&device_mem);
	config->enable_psf = false;
	
	//Generate  Dirty Images for MSMFS cleaning - added later 
	taylor = config->mf_num_moments;
	gains_apply_execute(config, host_mem, &device_mem, &timers);
	gridding_execute_msmfs(config,host_mem,&device_mem, &timers);
	host_mem->dirty_image = (PRECISION*) calloc(taylor*config->image_size*config->image_size, sizeof(PRECISION));
	pull_msmfs_image_cube(config, host_mem, &device_mem);
	save_taylor_term_planes(config, host_mem->dirty_image, taylor, config->image_size);
	msmfs_gridding_clean_up(&device_mem);
#endif	
	
	
	clean_up_device(&device_mem); 
	
	return EXIT_SUCCESS;
}


void save_taylor_term_planes(Config *config, PRECISION *host_image, int taylor_terms, int dim)
{
	
	char buffer[MAX_LEN_CHAR_BUFF * 4];
	
	for(int t=0;t<taylor_terms;t++)
	{
		snprintf(buffer, MAX_LEN_CHAR_BUFF*4, "%s%s_Plane_%d_psf.bin", config->data_output_path,(config->enable_psf ? "psf" : "dirtyimage"), t);
		printf("UPDATE >>> Attempting to save image to %s... \n\n", buffer);
		FILE *f = fopen(buffer, "wb");
	
		int saved = fwrite(host_image+t*dim*dim, sizeof(PRECISION), dim * dim, f);
		printf(">>> GRID DIMS IS : %d\n", dim);
		printf(">>> SAVED TO FILE: %d\n", saved);
		fclose(f);
	}
}


int execute_imaging_pipeline(Config *config, Host_Mem_Handles *host_mem)
{	
	// Destroy any hanging device mem allocations
	cudaDeviceReset();

	Timing timers;
	init_timers(&timers);

	Device_Mem_Handles device_mem;
	init_gpu_mem(&device_mem);

	// IMPLEMENTING: Perform visibility weighting here (required for PSF and gridding)
	
	//ToDo - taken out weight  scaling until we develop a better MSMFS solution
	
	visibility_weighting_execute(config, host_mem, &device_mem); 

	// Create PSF HERE will depend what solver we are using
	generate_psf(config, host_mem, &device_mem, &timers);
	//extract_pipeline_image(host_mem->h_psf, device_mem.d_psf, config->image_size);
	 
	 //Scale pixels by sum of visibility weights used during gridding
	//normalize_image_for_weighting(config->image_size, host_mem->dirty_image, config->visibility_scaled_weights_sum);
	 //save_image_to_file(config, host_mem->h_psf, "residual_image.bin", 0);
//exit(0);
	for(int cycle = 0; cycle < config->num_major_cycles; cycle++)
	{
		if(config->perform_gain_calibration && cycle == config->number_cal_major_cycles)
		{	
			config->perform_gain_calibration = false;
			config->num_sources = 0;
			//reset host and device mem predicted
			if(host_mem->visibilities != NULL)
				memset(host_mem->visibilities,0,config->num_host_visibilities * sizeof(VIS_PRECISION2));
		}
		if(cycle < config->number_cal_major_cycles)
			printf("UPDATE >>> Executing Calibration: Major cycle number %d...\n\n", cycle);
		else
			printf("UPDATE >>> Executing Imaging: Major cycle number %d...\n\n", cycle);
		// Gains Application / Subtraction
		gains_apply_execute(config, host_mem, &device_mem, &timers);

		// Gridding / FFT / Convolution Correction

#if SOLVER == NIFTY_GRIDDING		
		printf("UPDATE >>> EXECUTING NIFTY GRIDDER...\n\n");
		nifty_gridding_execute(config, host_mem, &device_mem, &timers);
#else 
		printf("UPDATE >>> EXECUTING W_PROJECTION GRIDDER...\n\n");
		gridding_execute(config, host_mem, &device_mem, &timers);
#endif
		if(config->save_dirty_image)
		{	
			extract_pipeline_image(host_mem->dirty_image, device_mem.d_image, config->image_size);
			// Scale pixels by sum of visibility weights used during gridding
			// normalize_image_for_weighting(config->image_size, host_mem->dirty_image, config->visibility_scaled_weights_sum);
			save_image_to_file(config, host_mem->dirty_image, "dirty_image.bin", cycle);
		}

		// image restoration
		if (cycle == config->num_major_cycles-1)
		{
			printf("AG UPDATE >>> Performing image restoration...\n\n");
			//do_image_restoration(config, host_mem, &device_mem);
		}
		else
		{
			// Deconvolution
			deconvolution_execute(config, host_mem, &device_mem, &timers);

			printf("Number of sources found is %d \n\n",config->num_sources);
			
			if(config->save_residual_image)
			{	
				extract_pipeline_image(host_mem->residual_image, device_mem.d_image, config->image_size);
				save_image_to_file(config, host_mem->residual_image, "residual_image.bin", cycle);
			}

			if(config->save_extracted_sources && config->num_sources > 0)
			{
				extract_extracted_sources(config, host_mem, &device_mem);
				save_extracted_sources(host_mem->h_sources, config->num_sources, config->data_output_path,config->imaging_output_id,
					"model_sources.bin", cycle);
			}

			#if PREDICT == DFT_PREDICTION
				// DFT
				printf("UPDATE >>> PREDICT PIPELINE - DFT  ...\n");
				dft_execute(config, host_mem, &device_mem, &timers);
			#elif PREDICT == NIFTY_GRIDDING
				printf("UPDATE >>> PREDICT PIPELINE - NIFTY_DEGRIDDING  ...\n");
				//plot model sources on a grid.
				execute_source_list_to_image(config,host_mem,&device_mem);
				
				nifty_degridding_execute(config, host_mem, &device_mem, &timers);
			#else
				printf("UPDATE >>> PREDICT PIPELINE - W_PROJECTION DEGRIDDING...\n");
				//execute_source_list_to_image(config,host_mem,&device_mem);
				degridding_execute(config,host_mem,&device_mem,&timers);
			#endif

			// Gain Calibration
			if(config->perform_gain_calibration)
			{	
				gain_calibration_execute(config, host_mem, &device_mem, &timers);

				if(config->save_estimated_gains)
				{	extract_pipeline_gains(config, host_mem, &device_mem);
					rotateAndOutputGains(config, host_mem, cycle);
				}
			}
		}
	}

	report_timings(&timers);
	clean_up_device(&device_mem);
	return EXIT_SUCCESS;
}

void generate_psf(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timers)
{
	config->enable_psf = true;
	#if SOLVER == NIFTY_GRIDDING
		nifty_psf_execute(config, host, device, timers);
	#else //SOLVER MUST BE W_PROJ (FOR NOW)
		psf_execute(config, host, device);
	#endif
	config->enable_psf = false;
}

void normalize_image_for_weighting(const int grid_size, PRECISION *image, PRECISION weighted_sum)
{	
	// Avoid divide by zero
	if(weighted_sum > 0.0)
		for(int i = 0; i < grid_size * grid_size; i++)
			image[i] /= weighted_sum;
}

void save_psf_to_file(Config *config, PRECISION *image, const char *file_name, int start_x, int start_y, int range_x, int range_y)
{
	char buffer[MAX_LEN_CHAR_BUFF * 2];
	snprintf(buffer, MAX_LEN_CHAR_BUFF, "%s%s", config->data_output_path, file_name);
	printf("UPDATE >>> Attempting to save PSF to %s... \n\n", buffer);

	FILE *f = fopen(buffer, "w");

	if(f == NULL)
	{	
		printf(">>> ERROR: Unable to save image to file %s, check file/folder structure exists...\n\n", buffer);
		return;
	}

    for(int row = start_y; row < start_y + range_y; ++row)
    {
    	for(int col = start_x; col < start_x + range_x; ++col)
        {
            PRECISION pixel = image[row * config->grid_size + col];

			#if SINGLE_PRECISION
            	fprintf(f, "%f ", pixel);
			#else
            	fprintf(f, "%lf ", pixel);
			#endif  
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void init_gpu_mem(Device_Mem_Handles *device)
{
    (*device).d_gains           = NULL;
    (*device).d_kernels         = NULL;
    (*device).d_kernel_supports = NULL;
    (*device).d_image           = NULL;
	(*device).d_uv_grid         = NULL;
	(*device).d_weight_map      = NULL;
    (*device).d_vis_uvw_coords  = NULL;
    (*device).d_vis_weights     = NULL;
    (*device).d_visibilities    = NULL;
    (*device).d_prolate         = NULL;
    (*device).d_sources         = NULL;
    (*device).d_psf             = NULL;
    (*device).d_max_locals      = NULL;
    (*device).d_measured_vis    = NULL;
    (*device).d_receiver_pairs  = NULL;
    (*device).fft_plan          = NULL;
    (*device).d_w_grid_stack	= NULL;
}


void clean_up_device(Device_Mem_Handles *device)
{
	printf("UPDATE >>> Cleaning up all allocated device memory...\n");
    if((*device).d_gains)           CUDA_CHECK_RETURN(cudaFree((*device).d_gains));
    if((*device).d_kernels)         CUDA_CHECK_RETURN(cudaFree((*device).d_kernels));
    if((*device).d_kernel_supports) CUDA_CHECK_RETURN(cudaFree((*device).d_kernel_supports));
    if((*device).d_image)           CUDA_CHECK_RETURN(cudaFree((*device).d_image));
	if((*device).d_uv_grid)         CUDA_CHECK_RETURN(cudaFree((*device).d_uv_grid));
	if((*device).d_vis_weights)     CUDA_CHECK_RETURN(cudaFree((*device).d_vis_weights));
    if((*device).d_vis_uvw_coords)  CUDA_CHECK_RETURN(cudaFree((*device).d_vis_uvw_coords));
    if((*device).d_weight_map)      CUDA_CHECK_RETURN(cudaFree((*device).d_weight_map));
    if((*device).d_visibilities)    CUDA_CHECK_RETURN(cudaFree((*device).d_visibilities));
    if((*device).d_prolate)         CUDA_CHECK_RETURN(cudaFree((*device).d_prolate));
    if((*device).d_sources)         CUDA_CHECK_RETURN(cudaFree((*device).d_sources));
    if((*device).d_psf)             CUDA_CHECK_RETURN(cudaFree((*device).d_psf));
    if((*device).d_max_locals)      CUDA_CHECK_RETURN(cudaFree((*device).d_max_locals));
    if((*device).d_measured_vis)    CUDA_CHECK_RETURN(cudaFree((*device).d_measured_vis));
    if((*device).d_receiver_pairs)  CUDA_CHECK_RETURN(cudaFree((*device).d_receiver_pairs));
    if((*device).d_w_grid_stack)  	CUDA_CHECK_RETURN(cudaFree((*device).d_w_grid_stack));
    if((*device).fft_plan)          free((*device).fft_plan);

    (*device).d_gains           = NULL;
    (*device).d_kernels         = NULL;
    (*device).d_kernel_supports = NULL;
    (*device).d_image           = NULL;
	(*device).d_uv_grid         = NULL;
	(*device).d_weight_map      = NULL;
    (*device).d_vis_uvw_coords  = NULL;
    (*device).d_vis_weights     = NULL;
    (*device).d_visibilities    = NULL;
    (*device).d_prolate         = NULL;
    (*device).d_sources         = NULL;
    (*device).d_psf             = NULL;
    (*device).d_max_locals      = NULL;
    (*device).d_measured_vis    = NULL;
    (*device).d_receiver_pairs  = NULL;
    (*device).fft_plan          = NULL;
    (*device).d_w_grid_stack	= NULL;

}

void init_timers(Timing *timers)
{
	init_timer(&(timers->gridder));
	init_timer(&(timers->nifty_solve_stack));
	init_timer(&(timers->ifft));
	init_timer(&(timers->solve_correction));

	init_timer(&(timers->degridder));
	init_timer(&(timers->nifty_predict_stack));
	init_timer(&(timers->fft));
	init_timer(&(timers->predict_correction));

	init_timer(&(timers->deconvolution));

	init_timer(&(timers->dft));

	init_timer(&(timers->gain_subtraction));
	init_timer(&(timers->gain_calibration));

	init_timer(&(timers->solver));
	init_timer(&(timers->solver_data_ingress));

	init_timer(&(timers->predict));
	init_timer(&(timers->predict_data_ingress));
	init_timer(&(timers->predict_data_egress));
}

void report_timings(Timing *timers)
{
	printf(">>> BULLSIT...\n");
	print_timer(&(timers->gain_subtraction), "Gain Subtraction");
	print_timer(&(timers->gain_calibration), "Gain Calibration");
	print_timer(&(timers->deconvolution), "Deconvolution");
	print_timer(&(timers->dft), "Direct Fourier Transform");

	printf(">>> SOLVER TIMING...\n");
	print_timer(&(timers->solver), "Solver");
	print_timer(&(timers->solver_data_ingress), "Solver Data Ingress");
	print_timer(&(timers->gridder), "Gridding");
	print_timer(&(timers->nifty_solve_stack), "NIFTy Solver Stacking");
	print_timer(&(timers->ifft), "FFT (inverse)");
	print_timer(&(timers->solve_correction), "Solver Convolution Correction");

	printf(">>> PREDICT TIMING...\n");
	print_timer(&(timers->predict), "Predict");
	print_timer(&(timers->predict_data_ingress), "Predict Data Ingress");
	print_timer(&(timers->degridder), "Degridding");
	print_timer(&(timers->nifty_predict_stack), "NIFTy Predict Stacking");
	print_timer(&(timers->fft), "FFT");
	print_timer(&(timers->predict_correction), "Predict Convolution Correction");
	print_timer(&(timers->predict_data_egress), "Predict Data Egress");

	#if SOLVER == NIFTY_GRIDDING
		printf("UPDATE: Where solver=nifty+fft+cc \n");
	#else
		printf("UPDATE: Where solver=wprojection+fft+cc \n");
	#endif
	
	#if PREDICT == NIFTY_GRIDDING
		printf("UPDATE: Where predict=cc+fft+nifty \n");
	#elif PREDICT == W_PROJECTION_GRIDDING
		printf("UPDATE: Where predict=cc+fft+wprojection \n");
	#else
		printf("UPDATE: Where predict=dft \n");
	#endif
}

void extract_pipeline_image(PRECISION *host_image, PRECISION *device_image, const int grid_size)
{
	if(device_image != NULL)
	{	printf("UPDATE >>>>>>> ALLOCATING DEVICE IMAGE FOR EXTRACTION \n\n");
		int grid_square = grid_size * grid_size;
  		CUDA_CHECK_RETURN(cudaMemcpy(host_image, device_image, grid_square * sizeof(PRECISION),cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
	}
}

void extract_predicted_visibilities(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	if(device->d_visibilities != NULL)
	{
	    CUDA_CHECK_RETURN(cudaMemcpy(host->visibilities, device->d_visibilities, 
        	config->num_host_visibilities * sizeof(VIS_PRECISION2), cudaMemcpyDeviceToHost));
			
		
		
		
    	cudaDeviceSynchronize();
	}
}

void extract_extracted_sources(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	if(device->d_sources != NULL)
	{
		if(host->h_sources)
		{
			free(host->h_sources);
			host->h_sources = NULL;
		}

		host->h_sources = (Source*) calloc(config->num_sources, sizeof(Source));

	    CUDA_CHECK_RETURN(cudaMemcpy(host->h_sources, device->d_sources, config->num_sources * sizeof(Source),
	        cudaMemcpyDeviceToHost));
	    cudaDeviceSynchronize();
	}
}

void extract_pipeline_gains(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	if(device->d_gains != NULL)
	{
	    CUDA_CHECK_RETURN(cudaMemcpy(host->h_gains, device->d_gains, 
        	config->num_recievers * sizeof(PRECISION2), cudaMemcpyDeviceToHost));
    	cudaDeviceSynchronize();
	}
}
