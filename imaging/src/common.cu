
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

#include "../common.h"

void image_free(Device_Mem_Handles *device)
{
    printf("UPDATE >>> Freeing Image Memory memory...\n\n");
    if(device->d_image != NULL) 
        CUDA_CHECK_RETURN(cudaFree(device->d_image));
    device->d_image = NULL;
}

void allocate_device_measured_vis(Config *config, Device_Mem_Handles *device)
{
    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_measured_vis), sizeof(VIS_PRECISION2) * config->num_visibilities));
}

void allocate_device_vis_coords(Config *config, Device_Mem_Handles *device)
{
    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_uvw_coords), sizeof(PRECISION3) * config->num_visibilities));
}

void free_device_vis_coords(Device_Mem_Handles *device)
{   
    if((*device).d_vis_uvw_coords)   
        CUDA_CHECK_RETURN(cudaFree(device->d_vis_uvw_coords));
    device->d_vis_uvw_coords = NULL;
}

void free_device_measured_vis(Device_Mem_Handles *device)
{
    if((*device).d_measured_vis)   
        CUDA_CHECK_RETURN(cudaFree(device->d_measured_vis));
    device->d_measured_vis = NULL;
}

void free_device_predicted_vis(Device_Mem_Handles *device)
{
    if((*device).d_visibilities)   
        CUDA_CHECK_RETURN(cudaFree(device->d_visibilities));
    device->d_visibilities = NULL;
}

void copy_measured_vis_to_device(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
    printf("UPDATE >>> Copying measured visibilities to device, number of visibilities: %d...\n\n",config->num_visibilities);
    CUDA_CHECK_RETURN(cudaMemcpy(device->d_measured_vis, host->measured_vis, sizeof(VIS_PRECISION2) * config->num_visibilities,
        cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();    
}

void free_device_sources(Device_Mem_Handles *device)
{   
    if(device->d_sources != NULL) 
        CUDA_CHECK_RETURN(cudaFree(device->d_sources));
    device->d_sources = NULL;
}

void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);

	exit(EXIT_FAILURE);
}

void cufft_safe_call(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
		printf("CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n",
			file, line, err, cuda_get_error_enum(err));
		cudaDeviceReset();
    }
}

const char* cuda_get_error_enum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
        	return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    	case CUFFT_INVALID_DEVICE:
    		return "CUFFT_INVALID_DEVICE";

        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";

        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";

        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";

        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";

        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
        }

        return "<unknown>";
}

void validate_snprintf(int buffer_size, int line_num, const char* file_name, int result)
{
    if(result < 0 || result > buffer_size)
    {
        printf("WARNING >>> Potential error or truncation when using snprintf on line %d of file %s\n\n", line_num, file_name);
    }
}