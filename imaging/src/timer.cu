
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

#include "../timer.h"

void init_timer(Timer *timer)
{
	CUDA_CHECK_RETURN(cudaEventCreate(&(timer->start)));
	CUDA_CHECK_RETURN(cudaEventCreate(&(timer->end)));
	timer->accumulated_time_millis = 0.0;
    timer->current_avg_time_millis = 0.0;
    timer->sum_of_square_diff_millis = 0.0;
    timer->iterations = 0;
}

void start_timer(Timer *timer, bool ignore)
{
	if(!ignore)
	{
		cudaEventRecord(timer->start);
		cudaEventSynchronize(timer->start);	
	}
}

void stop_timer(Timer *timer, bool ignore)
{
	if(!ignore)
	{
		cudaEventRecord(timer->end);
		cudaEventSynchronize(timer->end);
		float elapsed = 0.0;
		cudaEventElapsedTime(&elapsed, timer->start, timer->end);
		timer->accumulated_time_millis += elapsed;
	    timer->iterations++;
	    timer->current_avg_time_millis = timer->accumulated_time_millis / timer->iterations;
	    timer->sum_of_square_diff_millis += (elapsed - timer->current_avg_time_millis) * (elapsed - timer->current_avg_time_millis);
	}
}

void print_timer(Timer *timer, const char* stage_string)
{
	if(timer->iterations > 0)
	{
		double average = timer->current_avg_time_millis;
		double std_dev = sqrt(timer->sum_of_square_diff_millis / (timer->iterations));
		printf(">>> TIMING: %s average time %f milliseconds (std dev: ±%f, over %d iterations), accumulated time %f...\n\n",
			stage_string, average, std_dev, timer->iterations, timer->accumulated_time_millis);
	}
}