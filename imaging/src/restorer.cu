
// Copyright 2021 Anthony Griffin, Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2021 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

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
 	
#include "../restorer.h" 
#include "../controller.h"
#include "../fits_image_writer.h"


//#include "/home/seth/Desktop/test/cfitsio-3.49/include/fitsio.h"
// #include "/home/seth/Desktop/test/cfitsio-3.49/include/fitsio2.h"
// #include "/home/seth/Desktop/test/cfitsio-3.49/include/drvrsmem.h"
// #include "/home/seth/Desktop/test/cfitsio-3.49/include/longnam.h"

void show_matrix(PRECISION *matrix, int N);

void show_matrix(PRECISION *matrix, int N)
{
	for (int r = 0; r < N; r++)
	{
		printf("\t");			
		for (int c = 0; c < N; c++)
		{
			int i = r*N + c;
			if (matrix[i] == 0)
				printf(".");
			else
				printf("%i", int(matrix[i]));			
		}
		printf("\n");			
	}
}



void find_psf_max()
{
}


void do_image_restoration(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	printf("\n");
	printf("***********************************************************\n");
	printf("******************** IMAGE RESTORATION ********************\n");
	printf("***********************************************************\n\n");
	
	const char *dataset_name = "Gaussian";
	
	char full_file_name[MAX_LEN_CHAR_BUFF * 4];
	
	/// save residual image
	extract_pipeline_image(host->dirty_image, device->d_image, config->image_size);
	//normalize_image_for_weighting(config->image_size, host->dirty_image, config->visibility_scaled_weights_sum);
	snprintf(full_file_name, MAX_LEN_CHAR_BUFF*4, "%s%s_residual_image.fits", config->data_output_path, dataset_name);
	printf("UPDATE >>> Attempting to save image to %s... \n\n", full_file_name);
	save_image_to_fits_file(config, host->dirty_image, full_file_name);

	/// clip PSF
	
	PRECISION psfClipLevel = 97.5/100.0;
	
	// find psf max
	//PRECISION psfMax = 0;
	//PRECISION totalEnergy = 0;
	double psfMax = 0;
	double totalEnergy = 0;
	
	int R = 0;
	int C = 0;

	for (int r = 0; r < config->image_size; r++)
	{
		for (int c = 0; c < config->image_size; c++)
		{
			//PRECISION thisVal = host->h_psf[r*config->image_size + c];
			double thisVal = host->h_psf[r*config->image_size + c];
			if (thisVal >= psfMax)
			{
				psfMax = thisVal;
				C = c;
				R = r;
			}
			
			totalEnergy += thisVal*thisVal;
		}
	}
	
	printf("Max of PSF is %f at (%i, %i)\n", psfMax, C, R);	
	printf("Total energy in the PSF is %f\n", totalEnergy);	
	printf("PSF is %i x %i\n", config->image_size, config->image_size);
	
	int N = 1000;
	//PRECISION energy[N];
	double energy[N];
	int clipIndex = -1;
	
	for (int n = 0; n < N; n++)
	{
		//PRECISION thisSum = 0;
		double thisSum = 0;
		
		for (int r = -n; r <= n; r++)
		{
			for (int c = -n; c <= n; c++)
			{
				//PRECISION thisVal = host->h_psf[(r+R)*config->image_size + (c+C)];
				double thisVal = host->h_psf[(r+R)*config->image_size + (c+C)];
				
				thisSum += thisVal*thisVal;
			}
		}
		
		energy[n] = thisSum/totalEnergy;
		
		if ((clipIndex == -1) && (PRECISION(energy[n]) >= psfClipLevel))
		{
			clipIndex = n;
			printf("Clipping PSF at %.2f%%, clipped energy is %f, index is %i.\n", psfClipLevel*100.0, energy[n], clipIndex);
		}
		
		//printf("With (%3i x %3i), clipped energy is %f\n", n, n, energy[n]);	
	}

	int B = clipIndex;
	int beam_size = 2*B + 1;
	int image_size = config->image_size;
	
	// save beam
	PRECISION* beam = (PRECISION*) calloc(beam_size * beam_size, sizeof(PRECISION));
	
	for (int r = -B; r <= B; r++)
	{
		for (int c = -B; c <= B; c++)
		{
			int beam_index = (r + B) * beam_size  + (c + B);
			int  psf_index = (r + R) * image_size + (c + C);
			
			beam[beam_index] = host->h_psf[psf_index];
		}
	}
	
	/// make clean image
	extract_extracted_sources(config, host, device);
	// re-use dirty_image as clean image
	PRECISION* model_image = host->dirty_image;
	Source* sources = host->h_sources;
	//memset(model_image, 0, config->image_size * config->image_size * sizeof(PRECISION));

	
	printf("Number of pixel sources found is %d \n", config->num_sources);
	for (int s = 0; s < config->num_sources; s++)
	{
		int x = ROUND(sources[s].l / config->cell_size_rad) + config->image_size/2;
		int y = ROUND(sources[s].m / config->cell_size_rad) + config->image_size/2;
		PRECISION v = sources[s].intensity;
		printf("Source %3i is at (%f, %f) with value %f\n", s, sources[s].l, sources[s].m, v);
		printf("Source %3i is at (%4i, %4i) with value %f\n", s, x, y, v);
		
		//int index = config->image_size*y + x;
		//model_image[index] = v;
			
		const int2 min_image_point = make_int2(x - B, y - B);
		const int2 max_image_point = make_int2(x + B, y + B);

		int beam_index = 0;

		for(int image_y = min_image_point.y; image_y <= max_image_point.y; ++image_y)
		{	
			for(int image_x = min_image_point.x; image_x <= max_image_point.x; ++image_x)
			{
				int pixel_index = (image_y) * config->image_size + image_x;
				
				PRECISION beam_sample = beam[beam_index++];
				
				if (   (image_x >= 0) && (image_x < config->image_size) 
				    && (image_y >= 0) && (image_y < config->image_size) )
				{
					model_image[pixel_index] += v*beam_sample;
				}
			}
		}
	}

	/// save restored image
	snprintf(full_file_name, MAX_LEN_CHAR_BUFF*4, "%s%s_restored_image.fits", config->data_output_path, dataset_name);
	printf("UPDATE >>> Attempting to save image to %s... \n\n", full_file_name);
	save_image_to_fits_file(config, host->dirty_image, full_file_name);

	printf("\n");
}

#if 0
void restoring_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{
	printf("\n");
	printf("***********************************************************\n");
	printf("******************** IMAGE RESTORATION ********************\n");
	printf("***********************************************************\n\n");
	
	// hack
	//save_residual_image_as_FITS(config, host, device, "FullImage.fits");

	// Set up - allocate device mem, transfer host mem to device
	restoring_set_up(config, host, device);

	// Perform restoring
	start_timer(&timings->restoration);
	restoring_run(config, device);
	stop_timer(&timings->restoration);	
	
	///// check test image
    int total_pixels = config->image_size * config->image_size;
	CUDA_CHECK_RETURN(cudaMemcpy(
		host->residual_image, 
		device->d_image, 
		total_pixels * sizeof(PRECISION),
		cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	
	printf("Output image is: \n\n");
	show_matrix(host->residual_image, config->image_size);
	printf("\n");

	// Nifty test code!!!!
	printf("\nTesting on host...\n\n");
	exp_semicircle_test_run_h(config, host, device);
	printf("\nTesting on device...\n\n");
	exp_semicircle_test_run_d(config, host, device);

	if(!config->retain_device_mem)
	{
		// Transfer device mem back to host (only required data, and for retain data flag)
		//gridding_memory_transfer(config, host, device);

		// Clean up device
		//gridding_clean_up(device);
	}
}

void restoring_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
// AG cf gridding_set_up()
{
	// Testing
	if(device->d_sources == NULL)
	{
		printf("Source list not available!!!!!\n\n");

		return;
	}
	else
	{
		config->num_sources = 2;
		config->image_size = 10;
		
		printf("Source list seems to be available, with %i sources.\n\n", config->num_sources);
		
		//PRECISION3 temp = MAKE_PRECISION3(3, 3, 2);
		PRECISION3 temp[] = {MAKE_PRECISION3(3, 3, 2), MAKE_PRECISION3(3, 5, 3)};
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_sources, temp, sizeof(PRECISION3) * 2,
			cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		// zero image
 	    int total_pixels = config->image_size * config->image_size;
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_image), sizeof(PRECISION) * total_pixels));
    	CUDA_CHECK_RETURN(cudaMemset(device->d_image, 0, total_pixels * sizeof(PRECISION)));
    	cudaDeviceSynchronize();

		// initialise beam image
		config->beam_support = make_int2(1,1);
 	    int total_beam_elements = (2*(config->beam_support.x) + 1) * (2*(config->beam_support.y) + 1);
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_beam), sizeof(PRECISION) * total_beam_elements));
    	//CUDA_CHECK_RETURN(cudaMemset(device->d_beam, 0, total_beam_elements * sizeof(PRECISION)));

		// generate test beam
		PRECISION beam[] = {0, 1, 0, 1, 2, 1, 0, 1, 0};

		printf("Test beam is: \n\n");
		show_matrix(beam, 3);
		printf("\n");
		
		CUDA_CHECK_RETURN(cudaMemcpy(
			device->d_beam, 
			&beam, 
			sizeof(PRECISION) * total_beam_elements,
			cudaMemcpyHostToDevice));
    	cudaDeviceSynchronize();

		// check test image
  		CUDA_CHECK_RETURN(cudaMemcpy(
			host->residual_image, 
			device->d_image, 
			total_pixels * sizeof(PRECISION),
			cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

		printf("Input image is: \n\n");
		show_matrix(host->residual_image, config->image_size);
		printf("\n");
		
	}

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

	// Image
	if(!config->enable_psf && device->d_image == NULL)
	{
	    int grid_square = config->grid_size * config->grid_size;
    	CUDA_CHECK_RETURN(cudaMalloc(&(device->d_image), sizeof(PRECISION) * grid_square));
    	CUDA_CHECK_RETURN(cudaMemset(device->d_image, 0, grid_square * sizeof(PRECISION)));
    	cudaDeviceSynchronize();
	}
}
	
void restoring_run(Config *config, Device_Mem_Handles *device)
{
	int max_threads_per_block = min(config->gpu_max_threads_per_block, config->num_sources);
	int num_blocks = (int) ceil((double) config->num_sources / max_threads_per_block);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);

	printf("UPDATE >>> Restoring using %d blocks, %d threads, for %d sources...\n\n",
			num_blocks, max_threads_per_block, config->num_sources);

	// Execute convolving kernel
	convolve_source_with_beam<<<kernel_blocks, kernel_threads>>>(
		device->d_image, 
		device->d_beam, 
		config->beam_support,
		device->d_sources, 
		config->num_sources, 
		config->image_size);
		
	cudaDeviceSynchronize();

	printf("UPDATE >>> Restoring complete... BitB!!!\n\n");
}


// performs image restoration
__global__ void convolve_source_with_beam(PRECISION *image, const PRECISION *beam, const int2 beam_support,
	const PRECISION3 *source_list_lmv, const int num_sources, 
	const int image_size)
{
	const unsigned int source_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(source_index >= num_sources)
		return;
	
	//printf("Source %2i is at (%3i, %3i) with intensity %f.\n", source_index, int(source_list_lmv[source_index].x), int(source_list_lmv[source_index].y), source_list_lmv[source_index].z);

	const int half_support = beam_support.x;  // Why not /2??

	const int2 source_lm = make_int2(
		FLOOR(source_list_lmv[source_index].x),
		FLOOR(source_list_lmv[source_index].y)
	);
	
	const int2 min_image_point = make_int2(
		CEIL(source_list_lmv[source_index].x - half_support),
		CEIL(source_list_lmv[source_index].y - half_support)
	);

	const int2 max_image_point = make_int2(
		FLOOR(source_list_lmv[source_index].x + half_support),
		FLOOR(source_list_lmv[source_index].y + half_support)
	);
	
	PRECISION convolved   = PRECISION(0.0);
	PRECISION beam_sample = PRECISION(0.0);

	int image_index = 0;
	int beam_index = 0;
	
	PRECISION source_value = source_list_lmv[source_index].z;

	//printf("Source %2i: will affect [%3i to %3i, %3i to %3i] in the image.\n", source_index, min_image_point.x, max_image_point.x, min_image_point.y, max_image_point.y);
	
	for(int image_y = min_image_point.y; image_y <= max_image_point.y; ++image_y)
	{	
		for(int image_x = min_image_point.x; image_x <= max_image_point.x; ++image_x)
		{
			image_index = (image_y) * image_size + image_x;

			//if (source_index == 0)	printf("Source %2i: Image at (%3i, %3i) is intensity %f.\n", source_index, image_x, image_y, image[image_index]);
			
			beam_sample = beam[beam_index++];

			convolved = source_value * beam_sample;

			atomicAdd(&(image[image_index]), convolved);
		}
	}
}

#endif