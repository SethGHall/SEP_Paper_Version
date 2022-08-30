
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

#include "../gains.h"

void gain_calibration_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{
	printf("UPDATE >>> Performing gain calibration...\n\n");
	gains_set_up(config, host, device);

	start_timer(&timings->gain_calibration);
	gain_calibration_run(config, device);
	stop_timer(&timings->gain_calibration);

	if(!config->retain_device_mem)
	{
		// Transfer device mem back to host (only required data, and for retain data flag)
		gain_calibration_memory_transfer(config, host, device);

		// Clean up device
		gains_clean_up(device);
	}
}

void gains_apply_execute(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device, Timing *timings)
{
	// Set up - allocate device mem, transfer host mem to device
	printf("UPDATE >>> Applying gains... \n\n");
	gains_set_up(config, host, device);

	start_timer(&timings->gain_subtraction);
	gains_apply_run(config, device);
	stop_timer(&timings->gain_subtraction);

	if(!config->retain_device_mem)
	{
		// Transfer device mem back to host (only required data, and for retain data flag)
		gains_apply_memory_transfer(config, host, device);

		// Clean up device
		gains_clean_up(device);
	}
}


void gains_set_up(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	// Calibrated or default gains
	if(device->d_gains == NULL)
	{
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_gains), sizeof(PRECISION2) * config->num_recievers));
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_gains, host->h_gains, sizeof(PRECISION2) * config->num_recievers,
			cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	}

	// Reciever Pairs
	if(device->d_receiver_pairs == NULL)
	{
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_receiver_pairs),  sizeof(int2) * config->num_baselines));
		CUDA_CHECK_RETURN(cudaMemcpy(device->d_receiver_pairs, host->receiver_pairs, config->num_baselines * sizeof(int2), cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	}

	// Predicted Vis
	if(device->d_visibilities == NULL)
	{
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_visibilities), sizeof(VIS_PRECISION2) * config->num_visibilities));
	    CUDA_CHECK_RETURN(cudaMemcpy(device->d_visibilities, host->visibilities, sizeof(VIS_PRECISION2) * config->num_visibilities,
	        cudaMemcpyHostToDevice));
	    cudaDeviceSynchronize();
	}

	// Measured Vis
	if(device->d_measured_vis == NULL)
	{
		CUDA_CHECK_RETURN(cudaMalloc(&(device->d_measured_vis), sizeof(VIS_PRECISION2) * config->num_visibilities));
	    CUDA_CHECK_RETURN(cudaMemcpy(device->d_measured_vis, host->measured_vis, sizeof(VIS_PRECISION2) * config->num_visibilities,
	        cudaMemcpyHostToDevice));
	    cudaDeviceSynchronize();
	}
}

void gains_apply_run(Config *config, Device_Mem_Handles *device)
{
	int max_threads_per_block = min(config->gpu_max_threads_per_block, config->num_recievers);
	int num_blocks = (int) ceil((double) config->num_recievers / max_threads_per_block);

	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);

	//performs a reciprocal transform on the gains before applying to save on divisions
	reciprocal_transform<<<kernel_blocks, kernel_threads>>>(device->d_gains, config->num_recievers);
	cudaDeviceSynchronize();

	max_threads_per_block = min(config->gpu_max_threads_per_block, config->num_visibilities);
	num_blocks = (int) ceil((double) config->num_visibilities / max_threads_per_block);
	dim3 gains_blocks(num_blocks, 1, 1);
	dim3 gains_threads(max_threads_per_block, 1, 1);

	if(!config->perform_gain_calibration)
	{
		//apply gain calibration to update gains between measured and predicted visibilities
		apply_gains_subtraction<<<gains_blocks, gains_threads>>>(
				device->d_measured_vis,
				device->d_visibilities,
				config->num_visibilities,
				device->d_gains,
				device->d_receiver_pairs,
				config->num_recievers,
				config->num_baselines
		);
	}
	else
	{
		apply_gains<<<gains_blocks, gains_threads>>>(
				device->d_measured_vis,
				device->d_visibilities,
				config->num_visibilities,
				device->d_gains,
				device->d_receiver_pairs,
				config->num_recievers,
				config->num_baselines
		);
	}
	cudaDeviceSynchronize();
	//transform back from reciprocal of gains
	
	reciprocal_transform<<<kernel_blocks, kernel_threads>>>(device->d_gains,config->num_recievers);
	cudaDeviceSynchronize();
}

void gain_calibration_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{	// Gains
	printf("UPDATE >>> Copying gains array back to host, number of num_recievers: %d...\n\n",config->num_recievers);
	CUDA_CHECK_RETURN(cudaMemcpy(host->h_gains, device->d_gains, config->num_recievers * sizeof(PRECISION2), 
		cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}


void gains_apply_memory_transfer(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
{
	// Gains
	printf("UPDATE >>> Copying gains array back to host, number of num_recievers: %d...\n\n",config->num_recievers);
	CUDA_CHECK_RETURN(cudaMemcpy(host->h_gains, device->d_gains, config->num_recievers * sizeof(PRECISION2), 
		cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

	// Predicted vis
    CUDA_CHECK_RETURN(cudaMemcpy(host->visibilities, device->d_visibilities, 
        config->num_visibilities * sizeof(VIS_PRECISION2), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

void gains_clean_up(Device_Mem_Handles *device)
{
    if(device->d_gains != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_gains));
    device->d_gains = NULL;

    if(device->d_receiver_pairs != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_receiver_pairs));
    device->d_receiver_pairs = NULL;    

    if(device->d_visibilities != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_visibilities));
    device->d_visibilities = NULL;

    if(device->d_measured_vis != NULL)
        CUDA_CHECK_RETURN(cudaFree(device->d_measured_vis));
    device->d_measured_vis = NULL;
}



//USED FOR TESTING: before checking gains, need to "rotate" them relative to each other using the gain at 0
void rotateAndOutputGains(Config *config, Host_Mem_Handles *host, int cycle)
{
	Complex rotationZ = (Complex)
	{
		.real      = host->h_gains[0].real/SQRT(host->h_gains[0].real*host->h_gains[0].real + 
			         host->h_gains[0].imaginary * host->h_gains[0].imaginary), 
		.imaginary = -host->h_gains[0].imaginary/SQRT(host->h_gains[0].real*host->h_gains[0].real + 
                      host->h_gains[0].imaginary * host->h_gains[0].imaginary)
	};

	printf("UPDATE >>> ROTATED GAINS ...\n\n");

	char buffer[MAX_LEN_CHAR_BUFF * 2];
	snprintf(buffer, MAX_LEN_CHAR_BUFF, "%scycle_%d_output_gains.bin", config->data_output_path, cycle);
	printf("UPDATE >>> Attempting to save gains to %s... \n\n", buffer);

	FILE *f = fopen(buffer, "wb");

	if(f == NULL)
	{	
		printf(">>> ERROR: Unable to save image to file %s, check file/folder structure exists...\n\n", buffer);
		return;
	}

	for(int i=0;i<config->num_recievers;i++)
	{
		Complex rotatedGain = (Complex)
		{
			.real      = host->h_gains[i].real * rotationZ.real - host->h_gains[i].imaginary * rotationZ.imaginary,
			.imaginary = host->h_gains[i].real * rotationZ.imaginary + host->h_gains[i].imaginary * rotationZ.real
		};

		fwrite(&rotatedGain, sizeof(Complex), 1, f);
	}

	fclose(f);
}

__global__ void gains_product(PRECISION2 *d_gains, PRECISION2 *d_gains_product, const int num_recievers)
{
	const unsigned int receiver = blockIdx.x * blockDim.x + threadIdx.x;
	if(receiver >= num_recievers)
		return;

	d_gains_product[receiver] = complex_multiply(d_gains_product[receiver], d_gains[receiver]);
}


__global__ void reciprocal_transform(PRECISION2 *gains, const int num_recievers)
{
	const unsigned int receiver = blockIdx.x * blockDim.x + threadIdx.x;
	if(receiver >= num_recievers)
		return;

	gains[receiver] = complex_reciprocal(gains[receiver]);
}

__global__ void reset_gains(PRECISION2 *gains, const int num_recievers)
{
	const unsigned int receiver = blockIdx.x * blockDim.x + threadIdx.x;
	if(receiver >= num_recievers)
		return;

	gains[receiver] = MAKE_PRECISION2(1.0,0.0);
}

//gain calibration module
void gain_calibration_run(Config *config, Device_Mem_Handles *device)
{
	printf("UPDATE >>> Performing Gain Calibration using %d calibration cycles...\n\n", config->max_calibration_cycles);

	// int max_threads_per_block = min(config->gpu_max_threads_per_block, config->num_recievers);
	// int num_blocks = (int) ceil((double) config->num_recievers / max_threads_per_block);

	// dim3 reset_blocks(num_blocks, 1, 1);
	// dim3 reset_threads(max_threads_per_block, 1, 1);

	// reset_gains<<<reset_blocks, reset_threads>>>(
	// 	device->d_gains,
	// 	config->num_recievers
	// );

	PRECISION *Q_array;
	PRECISION *A_array;
	//init Q and A Array for reuse;
	CUDA_CHECK_RETURN(cudaMalloc(&Q_array,  sizeof(PRECISION) * 2 * config->num_recievers));
	CUDA_CHECK_RETURN(cudaMalloc(&A_array,  sizeof(PRECISION) * 2 * config->num_recievers * 2 * config->num_recievers));

	//SET CUDA WORK PLAN:
 	int max_threads_per_block = min(config->gpu_max_threads_per_block, config->num_baselines);
	int num_blocks = (int) ceil((double) config->num_baselines / max_threads_per_block);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);

	//Set SVD plans and matrices for reuse
	const int num_rows = 2 * config->num_recievers;
	const int num_cols = 2 * config->num_recievers;

	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	PRECISION *d_U = NULL;
	PRECISION *d_V = NULL;
	PRECISION *d_S = NULL;

	CUDA_CHECK_RETURN(cudaMalloc(&d_U, sizeof(PRECISION) * num_rows * num_cols));
	CUDA_CHECK_RETURN(cudaMalloc(&d_V, sizeof(PRECISION) * num_rows * num_cols));
	CUDA_CHECK_RETURN(cudaMalloc(&d_S, sizeof(PRECISION) * num_rows));

	int work_size = 0;
	CUDA_SOLVER_CHECK_RETURN(SVD_BUFFER_SIZE(solver_handle, num_rows, num_cols, &work_size));

	cudaDeviceSynchronize();

	PRECISION *work;	
	CUDA_CHECK_RETURN(cudaMalloc(&work, work_size * sizeof(PRECISION)));

	PRECISION *d_SUQ = NULL;
	CUDA_CHECK_RETURN(cudaMalloc(&d_SUQ, sizeof(PRECISION) * num_cols));


	for(int i=0;i<config->max_calibration_cycles;++i)
	{
		CUDA_CHECK_RETURN(cudaMemset(Q_array, 0, sizeof(PRECISION) * 2 * config->num_recievers));
		CUDA_CHECK_RETURN(cudaMemset(A_array, 0, sizeof(PRECISION) * 2 * config->num_recievers * 2 * config->num_recievers));
		CUDA_CHECK_RETURN(cudaMemset(d_V, 0, sizeof(PRECISION) * num_rows * num_cols));
		CUDA_CHECK_RETURN(cudaMemset(d_U, 0, sizeof(PRECISION)* num_rows * num_cols));
		CUDA_CHECK_RETURN(cudaMemset(d_S, 0, sizeof(PRECISION) * num_rows));
		CUDA_CHECK_RETURN(cudaMemset(work, 0, sizeof(PRECISION) * work_size));
		CUDA_CHECK_RETURN(cudaMemset(d_SUQ, 0, sizeof(PRECISION) * num_cols));


		printf("UPDATE >>> Performing Gain Calibration cycle: %d ...\n", i);
 		//EXECUTE CUDA KERNEL UPDATING Q AND A array (NEED ATOMIC ACCESS!)
		update_gain_calibration<<<kernel_blocks, kernel_threads>>>(
				device->d_measured_vis,
				device->d_visibilities,
				device->d_gains,
				device->d_receiver_pairs,
				A_array,
				Q_array,
				config->num_recievers,
				config->num_baselines
		);
		cudaDeviceSynchronize();	

		//NOW DO THE SVD
		if(!execute_calibration_SVD(config, solver_handle, A_array, d_U, d_V, d_S, work, work_size))
		{	printf("UPDATE >>> Unable to calibrate further, S array in SVD found a 0, exiting calibration cycle...\n");
			break;
		}	
		else
		{
			
			int max_threads_per_block = min(config->gpu_max_threads_per_block, num_cols);
			int num_blocks = (int) ceil((double)num_cols / max_threads_per_block);
			dim3 kernel_blocks(num_blocks, 1, 1);
			dim3 kernel_threads(max_threads_per_block, 1, 1);
			//Calculate product of S inverse,U and Q
			calculate_suq_product<<<kernel_blocks, kernel_threads>>>(
				d_S,
				d_U,
				Q_array, 
				d_SUQ, 
				num_cols);
			cudaDeviceSynchronize();

			max_threads_per_block = min(config->gpu_max_threads_per_block, config->num_recievers);
			num_blocks = (int) ceil((double) config->num_recievers / max_threads_per_block);
			dim3 blocks(num_blocks, 1, 1);
			dim3 threads(max_threads_per_block, 1, 1);
			//Caluclate product of V and SUQ from above, and update gains array
			calculate_delta_update_gains<<<blocks, threads>>>(
					d_V,
					d_SUQ,
					device->d_gains,
					config->num_recievers,
					num_cols
				);
			cudaDeviceSynchronize();
		}

		//RESET ALL MATRICES FOR NEXT CYCLE
	}

	if (d_SUQ   ) CUDA_CHECK_RETURN(cudaFree(d_SUQ));
	if (d_S     ) CUDA_CHECK_RETURN(cudaFree(d_S));
    if (d_U     ) CUDA_CHECK_RETURN(cudaFree(d_U));
    if (d_V     ) CUDA_CHECK_RETURN(cudaFree(d_V));
    if (work    ) CUDA_CHECK_RETURN(cudaFree(work));
	CUDA_SOLVER_CHECK_RETURN(cusolverDnDestroy(solver_handle));
	CUDA_CHECK_RETURN(cudaFree(Q_array));
	CUDA_CHECK_RETURN(cudaFree(A_array));

	printf("UPDATE >>>  Gain Calibration Complete ...\n");
}

//Apply gains on GPU between predicted and measured visibility values - USED FOR GAIN CALIBRATION
__global__ void apply_gains(VIS_PRECISION2 *measured_vis, VIS_PRECISION2 *predicted_vis, const int num_vis,
	const PRECISION2 *gains_recip, const int2 *receiver_pairs, const int num_recievers, const int num_baselines)
{
	const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;
	if(vis_index >= num_vis)
		return;

	int baselineNumber =  vis_index % num_baselines;
	//THIS ASSUMES THAT WE HAVE GAINS AS THE RECIPROCAL
	PRECISION2 gains_a_recip = gains_recip[receiver_pairs[baselineNumber].x];
	PRECISION2 gains_b_recip_conj = complex_conjugate(gains_recip[receiver_pairs[baselineNumber].y]);
	PRECISION2 gains_product_recip = complex_multiply(gains_a_recip, gains_b_recip_conj);

	#if ENABLE_16BIT_VISIBILITIES
		VIS_PRECISION2 gains_product_recip_16bit = MAKE_VIS_PRECISION2((VIS_PRECISION) gains_product_recip.x, (VIS_PRECISION) gains_product_recip.y);
		VIS_PRECISION2 measured_with_gains = complex_multiply_16bit(measured_vis[vis_index], gains_product_recip_16bit);
	#else
		PRECISION2 promoted = MAKE_PRECISION2(measured_vis[vis_index].x, measured_vis[vis_index].y);
		PRECISION2 measured_with_gains = complex_multiply(promoted, gains_product_recip);
	#endif

	predicted_vis[vis_index] = MAKE_VIS_PRECISION2((VIS_PRECISION) measured_with_gains.x, (VIS_PRECISION) measured_with_gains.y); 
}

//Apply gains on GPU between predicted and measured visibility values - USED FOR IMAGING
__global__ void apply_gains_subtraction(VIS_PRECISION2 *measured_vis, VIS_PRECISION2 *predicted_vis, const int num_vis,
	const PRECISION2 *gains_recip, const int2 *receiver_pairs, const int num_recievers, const int num_baselines)
{
	const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;
	if(vis_index >= num_vis)
		return;

	int baselineNumber =  vis_index % num_baselines;

	//THIS ASSUMES THAT WE HAVE GAINS AS THE RECIPROCAL
	PRECISION2 gains_a_recip = gains_recip[receiver_pairs[baselineNumber].x];
	PRECISION2 gains_b_recip_conj = complex_conjugate(gains_recip[receiver_pairs[baselineNumber].y]);
	PRECISION2 gains_product_recip = complex_multiply(gains_a_recip, gains_b_recip_conj);

	#if ENABLE_16BIT_VISIBILITIES
		VIS_PRECISION2 gains_product_recip_16bit = MAKE_VIS_PRECISION2((VIS_PRECISION) gains_product_recip.x, (VIS_PRECISION) gains_product_recip.y);
		VIS_PRECISION2 measured_with_gains = complex_multiply_16bit(measured_vis[vis_index], gains_product_recip_16bit);
		VIS_PRECISION2 subtracted_gain_vis = complex_subtract_16bit(measured_with_gains, predicted_vis[vis_index]);
	#else
		PRECISION2 promoted_measure = MAKE_PRECISION2(measured_vis[vis_index].x, measured_vis[vis_index].y);
		PRECISION2 promoted_predicted = MAKE_PRECISION2(predicted_vis[vis_index].x, predicted_vis[vis_index].y);
		PRECISION2 measured_with_gains = complex_multiply(promoted_measure, gains_product_recip);
		PRECISION2 subtracted_gain_vis = complex_subtract(measured_with_gains, promoted_predicted);
	#endif

	predicted_vis[vis_index] = MAKE_VIS_PRECISION2((VIS_PRECISION) subtracted_gain_vis.x, (VIS_PRECISION) subtracted_gain_vis.y); 
}


//Execute SVD - Note we must have CUDA 10 for this library to work
bool execute_calibration_SVD(Config *config, cusolverDnHandle_t solver_handle, PRECISION *d_A, 
								PRECISION *d_U, PRECISION *d_V, PRECISION *d_S, 
								PRECISION *work, int work_size)
{
    const int num_rows = 2 * config->num_recievers;
	const int num_cols = 2 * config->num_recievers;
	
	int *devInfo;
	CUDA_CHECK_RETURN(cudaMalloc(&devInfo, sizeof(int)));

	//ALL OUR MATRICES ARE "ROW" MAJOR HOWEVER AS A IS SYMMETRIC DOES NOT NEED TO BE TRANSPOSED FOR FOR SVD ROUTINE
	//SO NEED TO NOT TRANSPOSE U AND TRANSPOSE VSTAR
	CUDA_SOLVER_CHECK_RETURN(SVD(solver_handle, 'A', 'A', num_rows, num_cols, d_A,
		 num_rows, d_S, d_U, num_rows, d_V, num_cols, work, work_size, NULL, devInfo));
	cudaDeviceSynchronize();

	int devInfo_h = 0;	
	CUDA_CHECK_RETURN(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	//CHECKING S PRODUCT!!
    bool success = (devInfo_h == 0);
	//printf("UPDATE >>> SVD complete...\n\n");
	if (devInfo ) CUDA_CHECK_RETURN(cudaFree(devInfo));
	return success;
}

//delta+= transpose(V)*product of S U and Q (calculated in previous kernel)
__global__ void calculate_delta_update_gains(const PRECISION *d_V, const PRECISION *d_SUQ, PRECISION2 *d_gains, 
													const int num_recievers, const int num_cols)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x ;

	if(index >= num_recievers)
		return;

	PRECISION delta_top = 0;
	PRECISION delta_bottom = 0;

	int vindex = index * 2;
	for(int i=0;i<num_cols; ++i)
	{
		delta_top += d_SUQ[i] * d_V[vindex*num_cols + i];//[i*num_cols + vindex];
		delta_bottom += d_SUQ[i] * d_V[(vindex+1)*num_cols + i];//[i*num_cols + vindex+1];
	}

	d_gains[index].x += delta_top;
	d_gains[index].y += delta_bottom; 
}

__global__ void calculate_suq_product(const PRECISION *d_S, const PRECISION *d_U, const PRECISION *d_Q, 
	PRECISION *d_SUQ, const int num_entries)
{
	//qus 2Nx1, q = 2Nx1 , s=2Nx1, u=2N*2N
	const int index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index >= num_entries)
		return;

	PRECISION sinv = (abs(d_S[index]) > 1E-6) ? 1.0/d_S[index] : 0.0;

	PRECISION product = 0; 
	for(int i=0;i<num_entries;++i)
	{
		product += d_Q[i] * d_U[index*num_entries + i];
	}

	d_SUQ[index] = product * sinv;
}

__global__ void update_gain_calibration(const VIS_PRECISION2 *vis_measured_array, const VIS_PRECISION2 *vis_predicted_array, 
	const PRECISION2 *gains_array, const int2 *receiver_pairs, PRECISION *A_array, PRECISION *Q_array, 
	const int num_recievers, const int num_baselines)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index >= num_baselines)
		return;

	PRECISION2 vis_measured = MAKE_PRECISION2(vis_measured_array[index].x, vis_measured_array[index].y);
	PRECISION2 vis_predicted = MAKE_PRECISION2(vis_predicted_array[index].x, vis_predicted_array[index].y);

	int2 antennae = receiver_pairs[index];

	PRECISION2 gainA = gains_array[antennae.x];
	PRECISION2 gainB_conjugate = complex_conjugate(gains_array[antennae.y]);

	//NOTE do not treat residual as a COMPLEX!!!!! (2 reals)
	PRECISION2 residual = complex_subtract(vis_measured, complex_multiply(vis_predicted,complex_multiply(gainA, gainB_conjugate)));

	//CALCULATE Partial Derivatives
	PRECISION2 part_respect_to_real_gain_a = complex_multiply(vis_predicted, gainB_conjugate);

	PRECISION2 part_respect_to_imag_gain_a = flip_for_i(complex_multiply(vis_predicted, gainB_conjugate));

	PRECISION2 part_respect_to_real_gain_b = complex_multiply(vis_predicted,gainA);

	PRECISION2 part_respect_to_imag_gain_b = flip_for_neg_i(complex_multiply(vis_predicted, gainA));

	//Calculate Q[2a],Q[2a+1],Q[2b],Q[2b+1] arrays - In this order... NEED ATOMIC UPDATE 
	double qValue = part_respect_to_real_gain_a.x * residual.x 
					+ part_respect_to_real_gain_a.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.x]), qValue);

	qValue = part_respect_to_imag_gain_a.x * residual.x 
					+ part_respect_to_imag_gain_a.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.x+1]), qValue);

	qValue = part_respect_to_real_gain_b.x * residual.x 
					+ part_respect_to_real_gain_b.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.y]), qValue);

	qValue = part_respect_to_imag_gain_b.x * residual.x 
					+ part_respect_to_imag_gain_b.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.y+1]), qValue);


	int num_cols = 2 * num_recievers;
	//CALCULATE JAcobian product on A matrix... 2a2a, 2a2a+1, 2a2b, 2a2b+1
	//2a2a
	double aValue = part_respect_to_real_gain_a.x * part_respect_to_real_gain_a.x + 
					part_respect_to_real_gain_a.y * part_respect_to_real_gain_a.y; 
	
	int aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a2a+1,
	aValue = part_respect_to_real_gain_a.x * part_respect_to_imag_gain_a.x + 
					part_respect_to_real_gain_a.y * part_respect_to_imag_gain_a.y; 

	aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.x + 1);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a2b
	aValue = part_respect_to_real_gain_a.x * part_respect_to_real_gain_b.x + 
					part_respect_to_real_gain_a.y * part_respect_to_real_gain_b.y;

	aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a2b+1
	aValue = part_respect_to_real_gain_a.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_real_gain_a.y * part_respect_to_imag_gain_b.y;

	aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.y+1);

	atomicAdd(&(A_array[aIndex]), aValue);
	//CACLUATE JAcobian product on A matrix... [2a+1,2a], [2a+1,2a+1], [2a+1,2b], [2a+1,2b+1]
	//2a+1,2a
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_real_gain_a.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_real_gain_a.y; 

	aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a+1,2a+1
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_imag_gain_a.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_imag_gain_a.y; 

	aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.x+1);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a+1,2b
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_real_gain_b.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_real_gain_b.y;

	aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a+1,2b+1
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_imag_gain_b.y;

	aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.y+1);

	atomicAdd(&(A_array[aIndex]), aValue);


	//CACLUATE JAcobian product on A matrix... [2b,2a], [2b,2a+1], [2b,2b], [2b,2b+1]
	//2b,2a
	aValue = part_respect_to_real_gain_b.x * part_respect_to_real_gain_a.x + 
					part_respect_to_real_gain_b.y * part_respect_to_real_gain_a.y; 
	
	aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2b,2a+1
	aValue = part_respect_to_real_gain_b.x * part_respect_to_imag_gain_a.x + 
			 		part_respect_to_real_gain_b.y * part_respect_to_imag_gain_a.y;

	aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.x+1);

	atomicAdd(&(A_array[aIndex]), aValue);
	
	//2b,2b
	aValue = part_respect_to_real_gain_b.x * part_respect_to_real_gain_b.x + 
					part_respect_to_real_gain_b.y * part_respect_to_real_gain_b.y;
	
	aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);


	//2b, 2b+1
	aValue = part_respect_to_real_gain_b.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_real_gain_b.y * part_respect_to_imag_gain_b.y;

	aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.y + 1);

	atomicAdd(&(A_array[aIndex]), aValue);

	//CALCULATE JAcobian product on A matrix... [2b+1,2a], [2b+1,2a+1], [2b+1,2b], [2b+1,2b+1]
	//2b+1,2a
	aValue =  part_respect_to_imag_gain_b.x * part_respect_to_real_gain_a.x + 
					part_respect_to_imag_gain_b.y * part_respect_to_real_gain_a.y; 

	aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);


	//2b+1,2a+1
	aValue =  part_respect_to_imag_gain_b.x * part_respect_to_imag_gain_a.x+ 
					 part_respect_to_imag_gain_b.y * part_respect_to_imag_gain_a.y;

	aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.x+1);

	atomicAdd(&(A_array[aIndex]), aValue);


	//2b+1,2b
	aValue =  part_respect_to_imag_gain_b.x * part_respect_to_real_gain_b.x+ 
					part_respect_to_imag_gain_b.y * part_respect_to_real_gain_b.y;

	aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2b+1, 2b+1
	aValue = part_respect_to_imag_gain_b.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_imag_gain_b.y * part_respect_to_imag_gain_b.y; 

	aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.y+1);

	atomicAdd(&(A_array[aIndex]), aValue);

}

__device__ PRECISION2 complex_multiply(const PRECISION2 z1, const PRECISION2 z2)
{	
	return MAKE_PRECISION2(z1.x * z2.x - z1.y * z2.y, z1.x * z2.y + z1.y * z2.x);
}

__device__ VIS_PRECISION2 complex_multiply_16bit(const VIS_PRECISION2 z1, const VIS_PRECISION2 z2)
{
    return MAKE_VIS_PRECISION2(z1.x * z2.x - z1.y * z2.y, z1.y * z2.x + z1.x * z2.y);
}

// http://mathworld.wolfram.com/ComplexDivision.html
__device__ PRECISION2 complex_divide(const PRECISION2 z1, const PRECISION2 z2)
{
    PRECISION a = z1.x * z2.x + z1.y * z2.y;
    PRECISION b = z1.y * z2.x - z1.x * z2.y;
    PRECISION c = z2.x * z2.x + z2.y * z2.y;
    return MAKE_PRECISION2(a / c, b / c);
}

__device__ PRECISION2 flip_for_i(const PRECISION2 z)
{
	return MAKE_PRECISION2(-z.y, z.x);
}

__device__ PRECISION2 flip_for_neg_i(const PRECISION2 z)
{
	return MAKE_PRECISION2(z.y, -z.x);
}

__device__ PRECISION2 complex_conjugate(const PRECISION2 z1)
{
    return MAKE_PRECISION2(z1.x, -z1.y);
}

__device__ PRECISION2 complex_subtract(const PRECISION2 z1, const PRECISION2 z2)
{
    return MAKE_PRECISION2(z1.x - z2.x, z1.y - z2.y);
}

__device__ VIS_PRECISION2 complex_subtract_16bit(const VIS_PRECISION2 z1, const VIS_PRECISION2 z2)
{
    return MAKE_VIS_PRECISION2(z1.x - z2.x, z1.y - z2.y);
}

__device__ PRECISION2 complex_reciprocal(const PRECISION2 z)
{   
    PRECISION real = z.x / (z.x * z.x + z.y * z.y); 
    PRECISION imag = z.y / (z.x * z.x + z.y * z.y); 
    return MAKE_PRECISION2(real, -imag); 
}

void check_cuda_solver_error_aux(const char *file, unsigned line, const char *statement, cusolverStatus_t err)
{
	if (err == CUSOLVER_STATUS_SUCCESS)
		return;

	printf(">>> CUDA ERROR: %s returned %s: %u ",statement, file, line);
	exit(EXIT_FAILURE);
}
