
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
 
#ifndef GAINS_H_
#define GAINS_H_ 

#include <cusolverDn.h>
#include "common.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

	#define CUDA_SOLVER_CHECK_RETURN(value) check_cuda_solver_error_aux(__FILE__,__LINE__, #value, value)

	void gains_apply_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings);

	void gain_calibration_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings);

	void gains_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void gains_apply_run(Config *config, Device_Mem_Handles *device);

	void gains_apply_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void gain_calibration_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void gains_clean_up(Device_Mem_Handles *device);

	__global__ void apply_gains_subtraction(VIS_PRECISION2 *measured_vis, VIS_PRECISION2 *predicted_vis, const int num_vis,
		const PRECISION2 *gains_recip, const int2 *receiver_pairs, const int num_recievers, const int num_baselines);

	void gain_calibration_run(Config *config, Device_Mem_Handles *device);

	__global__ void reset_gains(PRECISION2 *gains, const int num_recievers);

	__global__ void update_gain_calibration(const VIS_PRECISION2 *vis_measured_array, const VIS_PRECISION2 *vis_predicted_array, 
	const PRECISION2 *gains_array, const int2 *receiver_pairs, PRECISION *A_array, PRECISION *Q_array, const int num_recievers, const int num_baselines);

	bool execute_calibration_SVD(Config *config, cusolverDnHandle_t solver_handle, PRECISION *d_A, PRECISION *d_U, PRECISION *d_V, PRECISION *d_S, PRECISION *work, int work_size);

	__global__ void gains_product(PRECISION2 *d_gains, PRECISION2 d_gains_product, const int num_recievers);

	__global__ void calculate_suq_product(const PRECISION *d_S, const PRECISION *d_U, const PRECISION *d_Q, PRECISION *d_SUQ, const int num_entries);

	__global__ void calculate_delta_update_gains(const PRECISION *d_V, const PRECISION *d_SUQ, PRECISION2 *d_gains, const int num_recievers, const int num_cols);

	void check_cuda_solver_error_aux(const char *file, unsigned line, const char *statement, cusolverStatus_t err);

	void execute_gains(Config *config, Device_Mem_Handles *device, Timer *timer);

	void rotateAndOutputGains(Config *config, Host_Mem_Handles *host, int cycle);

	void update_gains(Config *config, Device_Mem_Handles *device);

	void free_device_gains(Device_Mem_Handles *device);

	void free_device_receiver_pairs(Device_Mem_Handles *device);

	void copy_gains_to_host(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void copy_gains_to_device(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	void copy_receiver_pairs_to_device(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device);

	__global__ void apply_gains(VIS_PRECISION2 *measured_vis, VIS_PRECISION2 *predicted_vis, const int num_vis,
				const PRECISION2 *gains, const int2 *receiver_pairs, const int num_recievers, const int num_baselines);

	__global__ void calibrate_gains(PRECISION2 *gains, const int num_recievers);

	__global__ void reciprocal_transform(PRECISION2 *gains, const int num_recievers);

	__device__ PRECISION2 complex_multiply(const PRECISION2 z1, const PRECISION2 z2);

	__device__ PRECISION2 complex_divide(const PRECISION2 z1, const PRECISION2 z2);

	__device__ PRECISION2 complex_subtract(const PRECISION2 z1, const PRECISION2 z2);

	__device__ PRECISION2 complex_reciprocal(const PRECISION2 z);

	__device__ PRECISION2 flip_for_i(const PRECISION2 z);

	__device__ PRECISION2 flip_for_neg_i(const PRECISION2 z);

	__device__ PRECISION2 complex_conjugate(const PRECISION2 z1);

	__device__ VIS_PRECISION2 complex_multiply_16bit(const VIS_PRECISION2 z1, const VIS_PRECISION2 z2);

	__device__ VIS_PRECISION2 complex_subtract_16bit(const VIS_PRECISION2 z1, const VIS_PRECISION2 z2);

#ifdef __cplusplus
}
#endif

#endif /* GAINS_H_ */