
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
 
#ifndef WPROJECTION_H_
#define WPROJECTION_H_ 

#include "common.h"
#include "complex.h"
#include "fft.h"

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Macro function for compressed ternary operation of a min function, expands to "return" minimum of the two inputs
     * @param a The first input
     * @param b The second input 
     */
	#define MIN(a,b) (((a)<(b))?(a):(b))
    
    /**
     * Macro function for compressed ternary operation of a max function, expands to "return" maximum of the two inputs
     * @param a The first input
     * @param b The second input 
     */
    #define MAX(a,b) (((a)>(b))?(a):(b))

	bool generate_w_projection_kernels(Config *config, Host_Mem_Handles *host);
	
    void generate_phase_screen(int iw, int conv_size, int inner, PRECISION sampling, PRECISION w_scale, PRECISION* taper, Complex *screen);

    void normalize_kernels_by_maximum(Complex *kernels, PRECISION *maximums, int number_w_planes, int conv_half_size);

    void normalize_kernels_sum_of_one(Complex *kernels, int number_w_planes, int conv_half_size, int oversample);

    /**
     * Calculates a floating-point precision representation of the half support for the supplied w term
     * 
     * @param w The w term to be evaluated for support
     * @param min_support The minimum half supported needed for the current observation
     * @param w_to_max_support_ratio Ratio of max support - min support, and the maximum uvw of the observation
     */
    PRECISION calculate_support(PRECISION w, int min_support, PRECISION w_max_support_ratio);

    /**
	 * Calculates the next power of 2 integer from the supplied argument. It is possible for the same integer to be the next power of 2.
	 * 
	 * @param x The integer we want to calculate the next power of 2 integer from.
     * @return Returns the next calculated power of 2 integer (possibly itself).
	 */
	unsigned int get_next_pow_2(unsigned int x);

    void populate_ps_window(PRECISION *window, int size);

    PRECISION calculate_window_stride(int index, int size);

    double prolate_spheroidal(double nu);

    /**
	 * Loads a set of pre-calculated W-Projection convolution kernels from file (assuming required files are found).
	 * 
	 * @param config Global config struct which defines pipeline configuration
     * @param host Host memory handles (needed to store kernels into suitable buffer handle)
     * @return Returns true if file was found, kernels loaded, and file closed, otherwise false
	 */
    bool load_kernels_from_file(Config *config, Host_Mem_Handles *host);

    /**
     * Checks to see if pre-calculated W-Projection convolution kernel files can be located in users file system
     * @param config Global config struct which defines pipeline configuration
     * @return Returns true if kernel files were found, otherwise false
     */
    bool are_kernel_files_available(Config *config);

    /**
	 * Saves a set of calculated W-Projection convolution kernels to file (assuming required files are able to be saved).
	 * 
	 * @param config Global config struct which defines pipeline configuration
     * @param kernels The flattened set of W-Projection kernels
     * @param w_scale Ratio of number of w-planes - 1 squared, and the maximum uvw of the observation
     * @param w_to_max_support_ratio Ratio of max support - min support, and the maximum uvw of the observation
     * @param conv_half_size The half number of elements of a convolution kernel buffer used in W-Projection kernel creation 
     * @return Returns true if kernels were successfully saved to file, otherwise false
	 */
    bool save_kernels_to_file(Config *config, Complex *kernels, PRECISION w_scale, 
        PRECISION w_to_max_support_ratio, int conv_half_size);

    /**
	 * Copies a set of calculated W-Projection convolution kernels from generator code into to a flattened buffer, for use in the SEP Imaging Pipeline.
     * Performs trimming of redundant padding needed during kernel creation, and only copies over bottom-right quadrant of each kernel
     * into the flattened buffer. How much of each kernel is copied over depends on each kernels half support size, and the oversampling factor
     * specified during pipeline configuration.
	 * 
	 * @param config Global config struct which defines pipeline configuration
     * @param host Host memory handles (needed to store kernels into suitable buffer handle)
     * @param kernels The flattened set of W-Projection kernels
     * @param w_scale Ratio of number of w-planes - 1 squared, and the maximum uvw of the observation
     * @param w_to_max_support_ratio Ratio of max support - min support, and the maximum uvw of the observation
     * @param conv_half_size The half number of elements of a convolution kernel buffer used in W-Projection kernel creation 
     * @return Returns true if kernels were successfully saved to file, otherwise false
	 */
    bool bind_kernels_to_host(Config *config, Host_Mem_Handles *host, Complex* kernels,
        PRECISION w_scale, PRECISION w_to_max_support_ratio, int conv_half_size);

#ifdef __cplusplus
}
#endif 
 
#endif /* WPROJECTION_H_ */
