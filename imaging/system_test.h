
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

#ifndef SYSTEM_TEST_H_
#define SYSTEM_TEST_H_ 

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

	int evaluate_system_test_results(Config *config, PRECISION *dirty_image, Source *extracted_sources,
		Complex *predicted_visibilities, Visibility *uvw_coordinates, Complex *gains);

	double evaluate_dirty_image(Config *config, PRECISION *dirty_image);

	double evaluate_estimated_gains(Config *config, Complex *gains);

	bool evaluate_extracted_sources(Config *config, Source *extracted_sources);

	bool evaluate_predicted_visiblities(Config *config, Complex *predicted_visibilities, Visibility *uvw_coordinates);

	bool approx_equal(PRECISION a, PRECISION b);

	void init_system_test_config(Config *system_test_config);

#ifdef __cplusplus
}
#endif

#endif /* SYSTEM_TEST_H_ */
