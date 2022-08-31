
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

#ifndef MSMFS_NIFTY_H_
#define MSMFS_NIFTY_H_ 

	#include "controller.h"
	
#ifdef __cplusplus
extern "C" {
#endif

	void msmfs_nifty_gridding_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings);
	
	void msmfs_nifty_gridding_run(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);
	
	void msmfs_nifty_gridding_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void msmfs_nifty_clean_up(Device_Mem_Handles *device);
	 
	__global__ void nifty_accumulate_taylor_term_plane(PRECISION *plane, PRECISION *image_cube, const int t_plane, const int image_dim);
	
	__device__ PRECISION conv_corr(PRECISION support, PRECISION k); 
	
	
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
	);


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
	);

#ifdef __cplusplus
}
#endif 

#endif /* MSMFS_NIFTY_H_ */