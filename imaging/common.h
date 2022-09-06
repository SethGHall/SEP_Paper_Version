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

#ifndef COMMON_H_
#define COMMON_H_  

	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>
	#include <float.h>
	#include <string.h>
	#include <limits.h>
	#include <stdbool.h>

	#include <cuda.h>
	#include <cuda_runtime_api.h>
	#include <device_launch_parameters.h>
	#include <cufft.h>
	#include <cuda_fp16.h>
	
#ifdef __cplusplus
extern "C" {
#endif

	#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__,__LINE__, #value, value)
	#define CUFFT_SAFE_CALL(err) cufft_safe_call(err, __FILE__, __LINE__)

	#define snprintf(buffer, length, format, ...) validate_snprintf(MAX_LEN_CHAR_BUFF, __LINE__, __FILE__, snprintf(buffer, length, format, ##__VA_ARGS__))

/**
 *	The number of threads to active per thread block (GPU hardware dependant, typically 1024)
 */
#ifndef MAX_THREADS_PER_BLOCK
	#define MAX_THREADS_PER_BLOCK 1024
#endif

/**
 * Maximum length for intermediate buffers and file names/paths specified via config
 */
#ifndef MAX_LEN_CHAR_BUFF
	#define MAX_LEN_CHAR_BUFF 256
#endif

/**
 * Floating-point precision representation of the speed of light constant
 */
#ifndef SPEED_OF_LIGHT
	#define SPEED_OF_LIGHT 299792458.0
#endif

/**
 * Determines whether an approximately accurate visibility is calculated
 * during the DFT process. Use of approximation saves on square root ops.
 */
#ifndef APPROXIMATE_DFT
	#define APPROXIMATE_DFT 1
#endif


#define KERNEL_SUPPORT_BOUND 16

// bound for maximum possible samples for storing precalculated gauss-legendre quadrature nodes, weights, kernel
// value should ideally be nextPowerOf2(p), where p = (1.5*config->support+2)
// For more info on value of p, see first paragraph under equation 3.10, page 8 of paper:
// A parallel non-uniform fast Fourier transform library based on an "exponential of semicircle" kernel
#define QUADRATURE_SUPPORT_BOUND 32

// Number of iterations to perform for more precise approximation of legendre root
#define MAX_NEWTON_RAPHSON_ITERATIONS 100


/**
 * A flag which determines if the pipeline will execute using single or double precision. This includes
 * buffers used to store required data (ie: visibilities, kernels, etc.) and interim results, and also
 * sets whether computations will be performed using single or double precision calculations.
 */
#ifndef SINGLE_PRECISION
	#define SINGLE_PRECISION 1
#endif

// Psuedo-enums to determine which algorithms to use for solving/prediction
#ifndef W_PROJECTION_GRIDDING
    #define W_PROJECTION_GRIDDING 0 // TODO
#endif

#ifndef NIFTY_GRIDDING
    #define NIFTY_GRIDDING 1
#endif  

#ifndef DFT_PREDICTION   
    #define DFT_PREDICTION 2
#endif


#ifndef SOLVER
    #define SOLVER NIFTY_GRIDDING
#endif

#ifndef PREDICT
    #define PREDICT NIFTY_GRIDDING
#endif

#if SINGLE_PRECISION
	#define PRECISION float
	#define PRECISION2 float2
	#define PRECISION3 float3
	#define PRECISION4 float4
	#define PRECISION_MAX FLT_MAX
	#define PI ((float) 3.141592654)
	#define CUFFT_C2C_PLAN CUFFT_C2C
	#define CUFFT_C2P_PLAN CUFFT_C2R

#else
	#define PRECISION double
	#define PRECISION2 double2
	#define PRECISION3 double3
	#define PRECISION4 double4
	#define PRECISION_MAX DBL_MAX
	#define PI ((double) 3.1415926535897931)
	#define CUFFT_C2C_PLAN CUFFT_Z2Z
	#define CUFFT_C2P_PLAN CUFFT_Z2D
#endif

#if SINGLE_PRECISION
	#define SIN(x) sinf(x)
	#define COS(x) cosf(x)
	#define SINCOS(x, y, z) sincosf(x, y, z)
	#define ABS(x) fabsf(x)
	#define SQRT(x) sqrtf(x)
	#define ROUND(x) roundf(x)
	#define CEIL(x) ceilf(x)
	#define LOG(x) logf(x)
	#define POW(x, y) powf(x, y)
	#define FLOOR(x) floorf(x)
	#define VEXP(x)  expf(x)
    #define VSQRT(x) sqrtf(x)
	#define MAKE_PRECISION2(x,y) make_float2(x,y)
	#define MAKE_PRECISION3(x,y,z) make_float3(x,y,z)
	#define MAKE_PRECISION4(x,y,z,w) make_float4(x,y,z,w)
	#define CUFFT_EXECUTE_C2P(a,b,c) cufftExecC2R(a,b,c)
	#define CUFFT_EXECUTE_C2C(a,b,c,d) cufftExecC2C(a,b,c,d)
	#define SVD_BUFFER_SIZE(a,b,c,d) cusolverDnSgesvd_bufferSize(a,b,c,d)
	#define SVD(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) cusolverDnSgesvd(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)
#else
	#define SIN(x) sin(x)
	#define COS(x) cos(x)
	#define SINCOS(x, y, z) sincos(x, y, z)
	#define ABS(x) fabs(x)
	#define SQRT(x) sqrt(x)
	#define ROUND(x) round(x)
	#define CEIL(x) ceil(x)
	#define LOG(x) log(x)
	#define POW(x, y) pow(x, y)
	#define FLOOR(x) floor(x)
	#define VEXP(x)  exp(x)
    #define VSQRT(x) sqrt(x)
	#define MAKE_PRECISION2(x,y) make_double2(x,y)
	#define MAKE_PRECISION3(x,y,z) make_double3(x,y,z)
	#define MAKE_PRECISION4(x,y,z,w) make_double4(x,y,z,w)
	#define CUFFT_EXECUTE_C2P(a,b,c) cufftExecZ2D(a,b,c)
	#define CUFFT_EXECUTE_C2C(a,b,c,d) cufftExecZ2Z(a,b,c,d)
	#define SVD_BUFFER_SIZE(a,b,c,d) cusolverDnDgesvd_bufferSize(a,b,c,d)
	#define SVD(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) cusolverDnDgesvd(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)
#endif

/**
 * Switches between 16-bit or 32-bit visibilities/wproj kernels
 */
#ifndef ENABLE_16BIT_VISIBILITIES
	#define ENABLE_16BIT_VISIBILITIES 0
#endif

/**
 * Defines the memory type for visibility/wproj kernel precision (16 bit or 32 bit)
 */
#if ENABLE_16BIT_VISIBILITIES
	#define VIS_PRECISION __half
	#define VIS_PRECISION2 __half2
	#define MAKE_VIS_PRECISION2(x,y) make_half2(x,y)
#else
	#define VIS_PRECISION float
	#define VIS_PRECISION2 float2
	#define MAKE_VIS_PRECISION2(x,y) make_float2(x,y)
#endif

	/**
	 * A standard two-element structure representing a floating-point precision complex number
	 */
	typedef struct Complex {
		PRECISION real;			/**< Real component of the complex number (either single or double) */
		PRECISION imaginary;	/**< Imaginary component of the complex number (either single or double) */
	} Complex;

	/**
	 * A three-element structure representing uvw coordinates for a visibility
	 */
	typedef struct VisCoord {
		PRECISION u;			/**< U coordinate of a visibility */
		PRECISION v;			/**< V coordinate of a visibility */
		PRECISION w;			/**< W coordinate of a visibility */
	} VisCoord;

	/**
	 * A three-element structure representing a standard point source
	 */
	typedef struct Source {
		PRECISION l;			/**< L coordinate of a point source */
		PRECISION m;            /**< M coordinate of a point source */
		PRECISION intensity;	/**< The brightness of a point source */
	} Source;

	/**
	 * Possible options for visibility weighting schemes
	 */
	enum weighting_scheme { NATURAL, UNIFORM, ROBUST };
	/**
	   Possible options for multiscale clean shape. 
	   default 0 for delta, 1 for tapered truncated parabolas, 2 for truncated gaussians
	 */
	enum multiscale_clean_shape { DELTA, PARABOLA, GAUSSIAN };

	/**
	 * Stores the values for various configurable parameters used in the execution of the imaging pipeline
	 */

	typedef struct Nifty_Config {
        uint32_t num_w_grids_batched; // How many w grids to grid at once on the GPU
        bool perform_shift_fft;  // flag to (equivalently) rearrange each grid so origin is at lower-left corner for FFT
        double min_plane_w; // w coordinate of minimum w plane
        double max_plane_w; // w coordinate of maximum w plane
        double alpha;
        double beta;  // NOTE this 2.307 is only for when upsampling = 2
        double upsampling; // sigma, the scaling factor for padding each w grid (image_size * upsampling)
        uint32_t num_total_w_grids; // Total number of w grids 
        uint32_t support; // full support for semicircle gridding kernel
		double conv_corr_norm_factor;
		double epsilon;
	} Nifty_Config;

	typedef struct Config {

		/* For now we have a subconfig for Nifty, in future we will adapt this for other modules */
		//nifty config
		Nifty_Config nifty_config;

		// general
		int num_host_visibilities;
		int num_visibilities;          //REDUNDANT - just in for now
		int num_host_uvws;
		int vis_batch_size;
		int uvw_batch_size;
		
		char data_input_path[MAX_LEN_CHAR_BUFF];
		char data_output_path[MAX_LEN_CHAR_BUFF];
		char imaging_output_id[MAX_LEN_CHAR_BUFF]; //used to uniquely identify output files for a particular imaging cycle

		bool retain_device_mem;
		int num_major_cycles;
		int num_recievers;
		int num_baselines;
		int num_timesteps;
		int grid_size;
		double cell_size_rad;
        double fov_deg;
		int gpu_max_threads_per_block;				// TODO: load from cuda device query
		int gpu_max_threads_per_block_dimension;	// TODO: load from cuda device query
		bool right_ascension;
		double grid_size_padding_scalar;
		enum weighting_scheme weighting;
		double robustness;
		PRECISION visibility_scaled_weights_sum;

		//Frequencies
		double frequency_hz_start;        
		double frequency_bandwidth;
		double frequency_hz_increment;           //calculated from bandwidth/(numchannels-1)
		unsigned int number_frequency_channels;
		unsigned int timesteps_per_batch;      //number of visibility batches to be gridded per frequency
		
		
		//multiscale clean
		enum multiscale_clean_shape ms_clean_shape;     //enum to chose shape of multiscale cleaning
		double ms_scale_bias;                //lower scale bias will favour smaller scales during cleaning
		unsigned int num_ms_supports;
		double* ms_supports;
		bool mf_use;                         //flag to use multifrequency scale in gridding
		double mf_reference_hz;              
		double mf_spectral_index_overall;    //default spectral index alpha0
		unsigned int mf_num_moments;         // number of Taylor terms
		

		bool save_dirty_image;				/**< Flag to enable the saving of the dirty image to file (per major cycle) */
		bool save_residual_image;			/**< Flag to enable the saving of the residual image to file (per major cycle) */
		bool save_extracted_sources;		/**< Flag to enable the saving of extracted sources to file (after completing all minor cycles) */
		bool save_predicted_visibilities;	/**< Flag to enable the saving of predicted visibilities to file (per major cycle) */
		bool save_estimated_gains;			/**< Flag to enable the saving of estimated gains (after completing all calibration cycles) */

		// Testing
		bool perform_system_test;
		const char *system_test_image;
		const char *system_test_sources;
		const char *system_test_visibilities;
		const char *system_test_gains;

		// Gains
		char default_gains_file[MAX_LEN_CHAR_BUFF];
		bool perform_gain_calibration;
		bool use_default_gains;
		int max_calibration_cycles;
		int number_cal_major_cycles;

		// gridding
		bool enable_psf;
		double uv_scale;
		int total_kernel_samples;
		double w_scale;
		char visibility_source_file[MAX_LEN_CHAR_BUFF];
		char visibility_source_UVW_file[MAX_LEN_CHAR_BUFF];

		int min_half_support;
		int max_half_support;
		int num_kernels;
		int oversampling;
		int image_size;
		int psf_size;
        double min_abs_w; // minimum absolute w coordinate across all vis (non negative)
        double max_abs_w; // maximum absolute w coordinate across all vis
		bool load_kernels_from_file;
		bool save_kernels_to_file;
		char wproj_real_file[MAX_LEN_CHAR_BUFF * 2];
		char wproj_imag_file[MAX_LEN_CHAR_BUFF * 2];
		char wproj_supp_file[MAX_LEN_CHAR_BUFF * 2];

		// fft

		// deconvolution
		unsigned int number_minor_cycles_cal;
		unsigned int number_minor_cycles_img;
		double loop_gain;
		double weak_source_percent_gc;
		double weak_source_percent_img;
		double noise_factor;
		double psf_max_value;
		int num_sources;
		double search_region_percent;

		// Direct Fourier Transform
	} Config;

	typedef struct Host_Mem_Handles {

		// General
		PRECISION *dirty_image;
		PRECISION *residual_image;
		PRECISION *weight_map;

		// Gains
		Complex *h_gains;
		int2 *receiver_pairs;

		// Gridding
		VisCoord *vis_uvw_coords;
		VIS_PRECISION *vis_weights;
		VIS_PRECISION2 *visibilities;
		VIS_PRECISION2 *kernels;
		int2 *kernel_supports;

		// FFT

		// Correction
		PRECISION *prolate;

		// Deconvolution
		PRECISION *h_psf;
		Source *h_sources;

		// DFT
		VIS_PRECISION2 *measured_vis;

		//nifty gridding specific
		PRECISION2 	*w_grid_stack;
		PRECISION   *quadrature_nodes;
        PRECISION   *quadrature_weights;
        PRECISION   *quadrature_kernel;
		

	} Host_Mem_Handles;

	typedef struct Device_Mem_Handles {

		PRECISION2 *d_uv_grid;
		PRECISION *d_image;
		PRECISION *d_weight_map;

		// Gains
		PRECISION2 *d_gains;
		int2 *d_receiver_pairs;

		// Gridding
		VIS_PRECISION2 *d_kernels;
		int2 *d_kernel_supports;
		PRECISION3 *d_vis_uvw_coords;
		VIS_PRECISION2 *d_visibilities;
		VIS_PRECISION *d_vis_weights;

		// FFT
		cufftHandle *fft_plan;
		// Correction
		PRECISION *d_prolate;

		// Deconvolution
		PRECISION *d_psf;
		PRECISION3 *d_max_locals;
		PRECISION3 *d_sources;

		// DFT
		VIS_PRECISION2 *d_measured_vis;


		//nifty gridding specific
		PRECISION2  *d_w_grid_stack;


	} Device_Mem_Handles;

	void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err);

	void cufft_safe_call(cufftResult err, const char *file, const int line);

	const char* cuda_get_error_enum(cufftResult error);

	void image_free(Device_Mem_Handles *device);

	bool allocate_new_image_memory(Config *config, Device_Mem_Handles *device);

	void allocate_grid_device_mem(Config *config, Device_Mem_Handles *device);

	bool copy_gpu_image_to_host(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void allocate_device_measured_vis(Config *config, Device_Mem_Handles *device);

	void allocate_device_vis_coords(Config *config, Device_Mem_Handles *device);

	void free_device_vis_coords(Device_Mem_Handles *device);

	void free_device_measured_vis(Device_Mem_Handles *device);

	void free_device_predicted_vis(Device_Mem_Handles *device);

	void copy_predicted_vis_to_host(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void copy_measured_vis_to_device(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void copy_predicted_vis_to_device(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void copy_vis_coords_to_device(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void save_predicted_vis_to_file(Config *config, Host_Mem_Handles *host);

	void free_device_sources(Device_Mem_Handles *device);

	void allocate_device_sources(Config *config, Device_Mem_Handles *device);

	void validate_snprintf(int buffer_size, int line_num, const char* file_name, int result);

#ifdef __cplusplus
}
#endif 

#endif /* COMMON_H_ */
