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

#include "imaging/controller.h"

int main(int argc, char **argv)
{
	Config config;

	// argv[0] is program executable name
	// argv[1] is a default pipeline config yaml doc
	// argv[2] is a custom pipeline config yaml doc
	if(argc == 3)
	{
		char* default_config = argv[1];
		printf("INFO >>> Default pipeline configuration file supplied from path %s...\n\n", default_config);
		char* custom_config  = argv[2];
		printf("INFO >>> Custom pipeline configuration file supplied from path %s...\n\n", custom_config);

		if(!init_config_from_file(&config, default_config, true))
		{
			printf("ERROR >>> Unable to parse default configuration file, terminating...\n\n");
			return EXIT_FAILURE;
		}

		if(!init_config_from_file(&config, custom_config, false))
			printf("ERROR >>> Unable to parse custom configuration file, resorting to default configuration...\n\n");

		 update_calculated_configs(&config);
	}
	else
		printf("INFO >>> No default and custom config files supplied...\n\n");

	 return execute_controller(&config);
} 
