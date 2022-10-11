
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
 
#ifndef DIRECT_FOURIER_TRANSFORM_H_
#define DIRECT_FOURIER_TRANSFORM_H_ 

	#include "common.h"
	#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

	void dft_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings);

	void dft_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void dft_run(Config *config, Device_Mem_Handles *device, Timing *timings);

	void dft_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void dft_clean_up(Device_Mem_Handles *device);

	__global__ void direct_fourier_transform(const PRECISION3 *vis_uvw, VIS_PRECISION2 *predicted_vis,
		const int vis_count, const PRECISION3 *sources, const int source_count,
		const int num_channels, const int num_baselines, const PRECISION freq, const PRECISION freqInc);

#ifdef __cplusplus
}
#endif 
 
#endif /* DIRECT_FOURIER_TRANSFORM_H_ */
