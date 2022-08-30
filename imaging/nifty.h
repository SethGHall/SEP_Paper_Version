
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

#ifndef NIFTY_H_
#define NIFTY_H_  

    #include "controller.h"

    // Use of constant buffers instead of cudaMalloc'd memory allows for
    // more efficient caching, and "broadcasting" across threads
    extern __constant__ PRECISION quadrature_nodes[QUADRATURE_SUPPORT_BOUND];
    extern __constant__ PRECISION quadrature_weights[QUADRATURE_SUPPORT_BOUND];
    extern __constant__ PRECISION quadrature_kernel[QUADRATURE_SUPPORT_BOUND];

#ifdef __cplusplus
extern "C" {
#endif

    __device__ PRECISION conv_correction(PRECISION support, PRECISION k);

    __device__ PRECISION exp_semicircle(PRECISION beta, PRECISION x);

    __device__ PRECISION2 phase_shift(PRECISION w, PRECISION l, PRECISION m, PRECISION signage);
	
	void nifty_gridding_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings);
	
	void nifty_degridding_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings);
	
	void nifty_gridding_run(Config *config, Device_Mem_Handles *device);
	
	void nifty_degridding_run(Config *config, Device_Mem_Handles *device);
	
	void nifty_psf_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timers);
	
	void nifty_host_side_setup(Config *config, Host_Mem_Handles *host);
	
	void nifty_gridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void nifty_degridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void nifty_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void nifty_visibility_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void generate_gauss_legendre_conv_kernel(Host_Mem_Handles *host, Config *config);

    void psf_normalization_nifty(Config *config, Device_Mem_Handles *device);
	
	void nifty_clean_up(Device_Mem_Handles *device);

    void execute_source_list_to_image(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

    __global__ void scale_for_FFT(
        PRECISION2 *w_grid_stack, 
        const int num_w_planes, 
        const int grid_size, 
        const PRECISION scalar
    );

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
    );

    __global__ void apply_w_screen_and_sum(
        PRECISION *dirty_image, // INPUT & OUTPUT: real plane for accumulating phase corrected w layers across batches
        const uint32_t image_size, // one dimensional size of image plane (grid_size / sigma), assumed square
        const PRECISION pixel_size, // converts pixel index (x, y) to normalised image coordinate (l, m) where l, m between -0.5 and 0.5
        const PRECISION2 *w_grid_stack, // INPUT: flat array containing 2D computed w layers (w layer = iFFT(w grid))
        const uint32_t grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
        const int32_t grid_start_w, // signed index of first w grid in current subset stack
        const uint32_t num_w_grids_subset, // number of w grids bound in current subset stack
        const PRECISION inv_w_scale, // scaling factor for converting w coord to signed w grid index
        const PRECISION min_plane_w, // w coordinate of smallest w plane
        const bool perform_shift_fft  // flag to (equivalently) rearrange each grid so origin is at lower-left corner for FFT
    );

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
    );

    __global__ void conv_corr_and_scaling(
        PRECISION *dirty_image, // OUTPUT: final dirty image of all processed w layers
        const uint32_t image_size, // one dimensional size of image plane (grid_size / sigma), assumed square
        const PRECISION pixel_size, // converts pixel index (x, y) to normalised image coordinate (l, m) where l, m between -0.5 and 0.5
        const uint32_t support, // full support for gridding kernel
        const PRECISION conv_corr_norm_factor, // Strongest sample (the peak) in conv corr kernel
        const PRECISION *conv_corr_kernel, // Precalculated one side of convolutional correction kernel
        const PRECISION inv_w_range, // Inverse of range between max_plane_w and min_plane_w
        const PRECISION weight_channel_product, // Product of visibility sum of weights and num channels
        const PRECISION inv_w_scale, // Inverse of calculated w_scale term
        const bool solving
    );

     __global__ void source_list_to_image(
        const PRECISION3 *sources, 
        const int num_sources,
        PRECISION *image, 
        const int image_size, 
        const PRECISION cell_size
    );

    __global__ void find_psf_nifty_max(PRECISION *max_psf, const PRECISION *psf, const int grid_size);

    __global__ void psf_normalization_nifty_kernel(PRECISION max_psf, PRECISION *psf, const int image_size);

    double get_legendre(double x, int n, double *derivative);

    double get_approx_legendre_root(int32_t i, int32_t n);

    double calculate_legendre_root(int32_t i, int32_t n, double accuracy, double *weight);

#ifdef __cplusplus
}
#endif 

#endif /* NIFTY_H_ */
