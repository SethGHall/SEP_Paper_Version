
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

#ifndef TIMER_H_
#define TIMER_H_  

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef struct Timer {
		cudaEvent_t start;
		cudaEvent_t end;
		double accumulated_time_millis;
    	double current_avg_time_millis;
    	double sum_of_square_diff_millis;
    	uint32_t iterations;
	} Timer;

	typedef struct Timing {

		Timer gridder;
		Timer nifty_solve_stack;
		Timer ifft;
		Timer solve_correction;

		Timer degridder;
		Timer nifty_predict_stack;
		Timer fft;
		Timer predict_correction;

		Timer deconvolution;

		Timer dft;

		Timer gain_subtraction;
		Timer gain_calibration;

		Timer solver;
		Timer solver_data_ingress;

		Timer predict;
		Timer predict_data_ingress;
		Timer predict_data_egress;

	} Timing;

	void init_timer(Timer *timer);

	void start_timer(Timer *timer, bool ignore);

	void stop_timer(Timer *timer, bool ignore);

	void print_timer(Timer *timer, const char* stage_string);

#ifdef __cplusplus
}
#endif 

#endif /* TIMER_H_ */
