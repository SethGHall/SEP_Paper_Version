
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

#include "../system_test.h"

int evaluate_system_test_results(Config *config, PRECISION *dirty_image, Source *extracted_sources,
	Complex *predicted_visibilities, Visibility *uvw_coordinates, Complex *gains)
{
	// Dirty Image evaluation
	double relative_diff_threshold = 1E-5;
	printf("TEST >>> Evaluating dirty image against ideal image...\n\n");
	double relative_diff = evaluate_dirty_image(config, dirty_image);
	printf("TEST >>> dirty image relative difference measured as: %f\n\n", relative_diff);
	if(relative_diff > relative_diff_threshold)
	{
		printf("TEST >>> Relative difference is too high (threshold value: %f), system test failed...\n\n",
			relative_diff_threshold);
		return EXIT_FAILURE;
	}
	printf("TEST >>> Dirty image evaluation succeeded, continuing...\n\n");

	// Extracted sources evaluation
	printf("TEST >>> Evaluating extracted sources against test sources...\n\n");
	if(!evaluate_extracted_sources(config, extracted_sources))
	{
		printf("TEST >>> Extracted sources do not match test sources, system test failed...\n\n");
		return EXIT_FAILURE;
	}
	printf("TEST >>> Extracted sources evaluation succeeded, continuing...\n\n");

	// Predicted visibilities evaluation
	printf("TEST >>> Evaluating predicted visibilities against test visibilities...\n\n");
	if(!evaluate_predicted_visiblities(config, predicted_visibilities, uvw_coordinates))
	{
		printf("TEST >>> Predicted visibilities do not match test visibilities, system test failed...\n\n");
		return EXIT_FAILURE;
	}
	printf("TEST >>> Predicted visibilities evaluation succeeded, continuing...\n\n");

	printf("TEST >>> Evaluating predicted gains against true gains...\n\n");
	double gains_difference = evaluate_estimated_gains(config, gains);
	printf("TEST >>> Predicted gains show a %f relative L2 difference...\n\n",gains_difference);

	return EXIT_SUCCESS;
}


double evaluate_estimated_gains(Config *config, Complex *gains)
{
	FILE *file = fopen(config->system_test_gains, "r");
	if(file == NULL)
	{
		printf("TEST >>> Unable to locate system test gains file, system test unable to proceed...\n\n");
		return DBL_MAX;
	}
	PRECISION trueGainReal = 0.0; 
	PRECISION trueGainImag = 0.0; 

	Complex rotationZ = (Complex){
		.real = gains[0].real/sqrt(gains[0].real*gains[0].real + gains[0].imaginary * gains[0].imaginary), 
		.imaginary = - gains[0].imaginary/sqrt(gains[0].real*gains[0].real + gains[0].imaginary * gains[0].imaginary)
	};

	PRECISION sum_square_diff = 0.0;
	for(int i = 0; i < config->num_recievers; ++i)
	{
	#if SINGLE_PRECISION
		fscanf(file, "%f %f", &trueGainReal, &trueGainImag);
	#else
		fscanf(file, "%lf %lf", &trueGainReal, &trueGainImag);
	#endif

		Complex rotatedGain = (Complex){
			.real = gains[i].real * rotationZ.real - gains[i].imaginary * rotationZ.imaginary,
			.imaginary =  gains[i].real * rotationZ.imaginary + gains[i].imaginary * rotationZ.real
		};

		sum_square_diff += ((rotatedGain.real - trueGainReal) * (rotatedGain.real - trueGainReal) +
						   (rotatedGain.imaginary- trueGainImag) * (rotatedGain.imaginary - trueGainImag));

	}
	return sum_square_diff / config->num_recievers;
}



double evaluate_dirty_image(Config *config, PRECISION *dirty_image)
{
	// Locate comparison dirty image
	FILE *file = fopen(config->system_test_image, "r");
	if(file == NULL)
	{
		printf("TEST >>> Unable to locate system test dirty image file, system test unable to proceed...\n\n");
		return DBL_MAX;
	}

	// Measure relative root mean square error (rrmse) on the fly
	double diff_sum = 0.0;
	double dirty_sum = 0.0;
	for(int i = 0; i < config->grid_size * config->grid_size; ++i)
	{
		PRECISION sample = 0.0;
		#if SINGLE_PRECISION
			fscanf(file, "%f ", &sample);
		#else
			fscanf(file, "%lf ", &sample);
		#endif

		diff_sum += POW(dirty_image[i] - sample, 2.0);
		dirty_sum += POW(dirty_image[i], 2.0);
	}

	fclose(file);
	return (SQRT(diff_sum) / SQRT(dirty_sum)) * 100.0;
}

bool evaluate_extracted_sources(Config *config, Source *extracted_sources)
{
	// Locate comparison sources
	FILE *file = fopen(config->system_test_sources, "r");
	if(file == NULL)
	{
		printf("TEST >>> Unable to locate system test source file, system test unable to proceed...\n\n");
		return false;
	}

	int num_sources = 0;
	fscanf(file, "%d ", &num_sources);
	bool is_accurate = true;

	for(int i = 0; i < num_sources; ++i)
	{
		PRECISION sample_l = 0.0;
		PRECISION sample_m = 0.0;
		PRECISION sample_intensity = 0.0;

		#if SINGLE_PRECISION
			fscanf(file, "%f %f %f", &sample_l, &sample_m, &sample_intensity);
		#else
			fscanf(file, "%lf %lf %lf", &sample_l, &sample_m, &sample_intensity);
		#endif

		if(!approx_equal(sample_l, extracted_sources[i].l) || 
		   !approx_equal(sample_m, extracted_sources[i].m) ||
		   !approx_equal(sample_intensity, extracted_sources[i].intensity))
		{
			is_accurate = false;
			break;
		}
	}

	fclose(file);
	return is_accurate;
}

bool evaluate_predicted_visiblities(Config *config, Complex *predicted_visibilities, Visibility *uvw_coordinates)
{
	// Locate comparison visibilities
	FILE *file = fopen(config->system_test_visibilities, "r");
	if(file == NULL)
	{
		printf("TEST >>> Unable to locate system test visibility file, system test unable to proceed...\n\n");
		return false;
	}

	int num_visibilities = 0;
	fscanf(file, "%d ", &num_visibilities);
	bool is_accurate = true;

	for(int i = 0; i < num_visibilities; ++i)
	{
		PRECISION u = 0.0;
		PRECISION v = 0.0;
		PRECISION w = 0.0;
		PRECISION real = 0.0;
		PRECISION imag = 0.0;
		PRECISION weight = 0.0;

		#if SINGLE_PRECISION
			fscanf(file, "%f %f %f %f %f %f", &u, &v, &w, &real, &imag, &weight);
		#else
			fscanf(file, "%lf %lf %lf %lf %lf %lf", &u, &v, &w, &real, &imag, &weight);
		#endif

		if(config->right_ascension)
		{
			uvw_coordinates[i].u *= -1.0;
			uvw_coordinates[i].w *= -1.0;
 		}

		// Measure for discrepancy between uvw coordinates and complex visibilities
		if(!approx_equal(u, uvw_coordinates[i].u) ||
		   !approx_equal(v, uvw_coordinates[i].v) ||
		   !approx_equal(w, uvw_coordinates[i].w) ||
		   !approx_equal(real, predicted_visibilities[i].real) ||
		   !approx_equal(imag, predicted_visibilities[i].imaginary))
		{
			is_accurate = false;
			break;
		}
	}

	fclose(file);
	return is_accurate;
}

bool approx_equal(PRECISION a, PRECISION b)
{	
	return ABS(a - b) < 1E-5;
}

void init_system_test_config(Config *config)
{
	// general
	config->num_major_cycles                    = 1;
	config->num_recievers                       = 512;
	config->num_baselines						= (config->num_recievers*(config->num_recievers-1))/2;
	config->grid_size                           = 2048;
	config->cell_size                           = 8.52211548825356E-06; 
	config->frequency_hz                        = SPEED_OF_LIGHT; // data pre-scaled by 1.40e+08
	config->num_visibilities                    = 3924480;
	config->gpu_max_threads_per_block           = 1024;
	config->gpu_max_threads_per_block_dimension = 32;
	config->dirty_image_output                    = "";
	config->right_ascension						= true;
	config->visibility_source_file				= "../system_test_data/input/visibilities.csv";
	config->output_path							= "../system_test_data/output/";

	config->predicted_vis_output 				= "../system_test_data/sample_predicted_visibilities.csv";

	config->save_dirty_image					= false;
	config->save_residual_image					= false;
	config->save_extracted_sources			    = false;
	config->save_predicted_visibilities			= false;

	// Testing
	config->perform_system_test                 = true;
	config->system_test_image                   = "../system_test_data/model/dirty_image.csv";
	config->system_test_sources                 = "../system_test_data/model/sources.csv";
	config->system_test_visibilities            = "../system_test_data/model/visibilities.csv";
	config->system_test_gains					= "../system_test_data/model/true_gains_rotated.csv";
	// Gains 
	config->default_gains_file = "";
	config->max_calibration_cycles = 10;
	//if set to true use gains of 1.0+0j otherwise values given in above gains file.
	config->use_default_gains	= true;
	config->perform_gain_calibration = true;

	// Gridding
	config->max_w               = 1895.410847844;
	config->num_kernels         = 17;
	config->w_scale             = pow(config->num_kernels - 1, 2.0) / config->max_w;
	config->oversampling        = 16;
	config->uv_scale            = config->cell_size * config->grid_size;
	// config->kernel_real_file    = "../system_test_data/input/w-proj_kernels_real_x16.csv";
	// config->kernel_imag_file    = "../system_test_data/input/w-proj_kernels_imag_x16.csv";
	// config->kernel_support_file = "../system_test_data/input/w-proj_supports_x16.csv";
	config->force_weight_to_one	= true;

	// Deconvolution
	config->number_minor_cycles_cal = 5000;
	config->number_minor_cycles_img = 200;
	config->loop_gain           = 0.1;  // 0.1 is typical
	config->weak_source_percent_gc = 0.001;//0.00005; // example: 0.05 = 5%
	config->weak_source_percent_img = 0.05;//0.00005; // example: 0.05 = 5%
	config->psf_max_value       = 0.0;  // customize as needed, or allow override by reading psf.
	/*
		Used to determine if we are extracting noise, based on the assumption
		that located source < noise_detection_factor * running_average
	*/ 
	config->noise_factor          = 1.5;
	config->model_sources_output  = "";
	config->residual_image_output = "";
	// config->psf_input_file        = "../system_test_data/input/point_spread_function.csv";
	config->num_sources           = 0;

	// Direct Fourier Transform
	config->predicted_vis_output = "";
}