
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

#include "../controller.h"

bool memory_region_ready[NUM_MEM_REGIONS];
char pthread_filename[MAX_LEN_CHAR_BUFF]; // = "GLEAM_small_subset_intensity.bin";
char pthread_data_path[MAX_LEN_CHAR_BUFF];

int execute_controller(Config *config)
{
	printf("=========================================================================\n");
	printf(">>> AUT HPC Research Laboratory - SDP Evolutionary Pipeline (Imaging) <<<\n");
	printf("=========================================================================\n\n");

	#if SINGLE_PRECISION
		printf("UPDATE >>> Pipeline will execute COMPUTE CALCULATIONS in SINGLE precision...\n\n");

	#else
		printf("UPDATE >>> Pipeline will execute COMPUTE CALCULATIONS in DOUBLE precision...\n\n");
	#endif

	#if ENABLE_16BIT_VISIBILITIES
		printf("UPDATE >>> Pipeline will TRANSFER VISIBILITIES in HALF (16bit) precision...\n\n");
	#else 
		printf("UPDATE >>> Pipeline will TRANSFER VISIBILITIES SINGLE precision...\n\n");
	#endif

	if(config->retain_device_mem)
		printf("UPDATE >>> Device memory will be retained across all stages of the imaging pipeline...\n\n");
	else
		printf("UPDATE >>> Device memory will be not be retained across all stages of the imaging pipeline...\n\n");

	Host_Mem_Handles host_mem;

	// Zero out all memory handles
	init_mem(&host_mem);

	// Gain Calibration
	if(!gains_host_set_up(config, &host_mem))
		return exit_and_host_clean("Unable to allocate host based gains array memory", &host_mem);

	if(!visibility_UVW_host_setup(config, &host_mem))
		return exit_and_host_clean("Unable to set up visibilities UVW information", &host_mem);

	if(!visibility_intensity_host_setup(config, &host_mem))
		return exit_and_host_clean("Unable to set up visibilities INTENSITY information", &host_mem);
	
#if SOLVER == NIFTY_GRIDDING		
		printf("UPDATE >>> Setting up host side memory for NIFTY...\n\n");
		nifty_host_side_setup(config, &host_mem);
		
#else 
		// W Projection Gridding
		printf("UPDATE >>> Setting up host side memory for W_PROJECTION...\n\n");
		if(!kernel_host_setup(config, &host_mem))
			return exit_and_host_clean("Unable to set up convolution kernels", &host_mem);
		if(!correction_set_up(config, &host_mem))
			return exit_and_host_clean("Unable to set up Convolution Correction", &host_mem); 
#endif
	

	// Convolution correction
	// Deconvolution
	int psf_size_square = config->image_size * config->image_size;
	host_mem.h_psf = (PRECISION*) calloc(psf_size_square, sizeof(PRECISION));
	// SETUP HOST PSF.... NEED BETTER WAY TO DO THIS???

	if(!allocate_host_images(config, &host_mem))
		return exit_and_host_clean("Unable to allocate host based memory for output images", &host_mem);

	// HACK: required for pthread argument bypass
	snprintf(pthread_filename, MAX_LEN_CHAR_BUFF, "%s", config->visibility_source_file);
	snprintf(pthread_data_path, MAX_LEN_CHAR_BUFF, "%s", config->data_input_path);

	MemoryRegions regions;
    regions.num_visibilities = config->num_visibilities;

	size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);
	if(PAGE_SIZE <= 0)
		PAGE_SIZE = 4096;

	for(int i=0;i<NUM_MEM_REGIONS;++i)
	{	
		regions.memory_region[i] = (VIS_PRECISION2*) memalign(PAGE_SIZE,config->num_visibilities * sizeof(VIS_PRECISION2));
		memory_region_ready[i] = false;
	}

	//SETUP POSIX THREADS TO READ VISIBLITY FILES
	pthread_t vis_reader_thread;
    int result_code = pthread_create(&vis_reader_thread, NULL, load_vis_into_memory_regions, &regions);
	if(result_code != 0)
    {   printf("ERROR >>> CANNOT RUN POSIX THREADS .... \n");
        return EXIT_FAILURE;
    }	
	
	
	
	


	for(int imagingCycle=0; imagingCycle < NUM_IMAGING_ITERATIONS; ++imagingCycle)
	{
		int region_num = imagingCycle%NUM_MEM_REGIONS;	
		printf("PIPELINE >>> WAITING FOR %d TO BE READY... \n\n ",region_num); 
		while(!memory_region_ready[region_num])
        {
            usleep(100);
        }
        printf("PIPELINE >>> MEMORY REGION %d READY... \n\n ",region_num); 
		
		host_mem.measured_vis = regions.memory_region[region_num];

		snprintf(config->imaging_output_id,MAX_LEN_CHAR_BUFF, "imaging_cycle_%d_",imagingCycle);
		//bind memory region to Complex vis;
		execute_imaging_pipeline(config, &host_mem);
		//pretend_imaging_pipeline(&host_mem,imagingCycle,config->num_visibilities);
		//free Complex vis
		printf("UPDATE >>> IMAGING CYCLE %d COMPLETE!!!!\n", imagingCycle);
		memory_region_ready[region_num] = false;
		host_mem.measured_vis = NULL;
        usleep(SLEEP_PIPELINE_MICRO);
	}

	printf("UPDATE >>> CLEANING UP MEMORY REGIONS ... \n\n");
	for(int i=0;i<NUM_MEM_REGIONS;++i)
	{	
		if(regions.memory_region[i] != NULL)  
			free(regions.memory_region[i]);
		regions.memory_region[i] = NULL;
	}
	
	clean_up(&host_mem);
	printf("UPDATE >>> Imaging Pipeline Complete...\n\n");
	return EXIT_SUCCESS;
}

void* load_vis_into_memory_regions(void *args)
{
    MemoryRegions *regions = (MemoryRegions*) args;
    
  
    for(int i=0; i<NUM_IMAGING_ITERATIONS;i++)
    {
        int region_num = i%NUM_MEM_REGIONS;
        
        printf(">>>LOADER thread waiting for memory REGION %d...\n\n", region_num);
        while(memory_region_ready[region_num])
        {
            usleep(100);
        }
        
        printf(">>>LOADER populating REGION %d...\n\n", region_num);
         
		char buffer[MAX_LEN_CHAR_BUFF * 2];
		// snprintf(buffer, MAX_LEN_CHAR_BUFF, "%s%d_%s", pthread_data_path, i, pthread_filename);
		snprintf(buffer,MAX_LEN_CHAR_BUFF * 2, "%s%s", pthread_data_path, pthread_filename);
		 
        populate_memory_megion(regions->memory_region[region_num], regions->num_visibilities, buffer);
        
        printf(">>>LOADER setting REGION %d READY...\n\n", region_num);
        memory_region_ready[region_num] = true;
    }
    return 0;
}

void populate_memory_megion(VIS_PRECISION2 *region, int num_vis,  char* filename)
{
	printf("UPDATE LOADER >>> Attempting to read VIS from %s... \n\n", filename);
    FILE *handle = fopen(filename, "rb");
	if(handle == NULL)
    {   printf("ERROR INSIDE LOADER >>> Cannot read file from %s \n",filename); 
        return;
    }
	int num_vis_read;
	fread(&num_vis_read, sizeof(int), 1, handle);
	printf("LOADER READING %d vis\n",num_vis_read);
    if(num_vis_read != num_vis)
    {
        printf("ERROR INSIDE LOADER >>> VISIBILITY NUMBERS DO NOT MATCH FILE %d vs CONFIG %d \n",num_vis_read,num_vis); 
        return;
    }  

	// Used to load in single/double vis before convert to half precision
	PRECISION2* temp_loader = (PRECISION2*) calloc(num_vis, sizeof(PRECISION2));
    int num_elements_read = fread(temp_loader, sizeof(PRECISION2), num_vis, handle);
	
	if(num_vis_read != num_elements_read)
    {
        printf("ERROR INSIDE LOADER >>> VISIBILITY NUMBERS %d DO NOT MATCH AMOUNT READ FROM FILE %d \n",num_vis_read,num_elements_read); 
		free(temp_loader);
		temp_loader = NULL;
        return;
    }  
	printf("NUM ELEMENTS READ IS %d \n",num_elements_read);
	// Copy over from single/double to half/single precision
	for(int i = 0; i < num_vis; i++)
		region[i] = MAKE_VIS_PRECISION2((VIS_PRECISION) temp_loader[i].x, (VIS_PRECISION) temp_loader[i].y);

	free(temp_loader);
	temp_loader = NULL;

	fclose(handle);	
}


    // return true;

void init_mem(Host_Mem_Handles *host)
{
    (*host).kernels             = NULL;
    (*host).kernel_supports     = NULL;
    (*host).dirty_image         = NULL;
	(*host).residual_image      = NULL;
	(*host).weight_map          = NULL;
    (*host).vis_uvw_coords      = NULL;
    (*host).vis_weights         = NULL;
    (*host).visibilities        = NULL;
    (*host).h_psf               = NULL;
    (*host).h_sources           = NULL;
    (*host).measured_vis        = NULL;
    (*host).prolate             = NULL;
    (*host).receiver_pairs      = NULL;
    (*host).h_gains             = NULL;
	(*host).quadrature_nodes	= NULL;
	(*host).quadrature_weights	= NULL;
	(*host).quadrature_kernel	= NULL;
}

//Saves output image to file  - used for testing
void save_image_to_file(Config *config, PRECISION *image, const char *file_name, int cycle)
{
	char buffer[MAX_LEN_CHAR_BUFF * 4];
	snprintf(buffer, MAX_LEN_CHAR_BUFF*4, "%s%smajor_cycle_%d_%s", 
		config->data_output_path, config->imaging_output_id,cycle, file_name);
	printf("UPDATE >>> Attempting to save image to %s... \n\n", buffer);

	FILE *f = fopen(buffer, "wb");

	if(f == NULL)
	{	
		printf(">>> ERROR: Unable to save image to file %s, check file/folder structure exists...\n\n", buffer);
		return;
	}
	
	int saved = fwrite(image, sizeof(PRECISION), config->image_size * config->image_size, f);
	printf(">>> GRID DIMS IS : %d\n", config->image_size);
	printf(">>> SAVED TO FILE: %d\n", saved);
    fclose(f);
}

bool parse_config_attribute(struct fy_document *fyd, const char *attrib_name, const char *format, void* data, bool required)
{
	char buffer[MAX_LEN_CHAR_BUFF * 2];
	snprintf(buffer, MAX_LEN_CHAR_BUFF, "/%s %s", attrib_name, format);

	if(fy_document_scanf(fyd, buffer, data))
	{
		printf("UPDATE >>> Successfully parsed attribute %s...\n\n", attrib_name);
		return true;
	}
	else
	{
		if(required)
			fy_document_destroy(fyd);
		printf("ERROR >>> Unable to find attribute %s in YAML config file (required: %s)...\n\n", attrib_name, (required) ? "true" : "false");
		return false;
	}
}

bool parse_config_attribute_bool(struct fy_document *fyd, const char *attrib_name, const char *format, bool* data, bool required)
{
	int obtained = 0;
	if(parse_config_attribute(fyd, attrib_name, format, &obtained, required))
	{	
		*data = (obtained == 1) ? true : false;
		return true;
	}

	return false;
}

void update_calculated_configs(Config *config)
{
	// Calculated assuming we have values set from previous sweeps
	config->num_visibilities = 0;
	config->total_kernel_samples = 0;
	config->psf_max_value = 0.0;
	config->num_sources = 0;
	config->enable_psf = false;
	config->num_baselines = (config->num_recievers * (config->num_recievers- 1 )) / 2;
	config->visibility_scaled_weights_sum = 0.0;
	config->gpu_max_threads_per_block = MAX_THREADS_PER_BLOCK;
	config->gpu_max_threads_per_block_dimension = 32;
	config->cell_size_rad = asin(2.0 * sin(0.5 * config->fov_deg*PI/(180.0)) / PRECISION(config->image_size));

	//calculate depending on what griddding operation is available.	
#if SOLVER == NIFTY_GRIDDING
		//Overide some w-projection and other value if NIFTY GRIDDING
		config->nifty_config.beta = (config->nifty_config.beta * ((PRECISION)config->nifty_config.support)); // NOTE this 2.307 is only for when upsampling = 2
		config->grid_size = ceil(config->image_size * config->nifty_config.upsampling);   // upscaled one dimension size of grid for each w grid
		
		/****************************************************************************************
		 * DO NOT MODIFY BELOW - DO NOT MODIFY BELOW - DO NOT MODIFY BELOW - DO NOT MODIFY BELOW 
		 ***************************************************************************************/
		config->min_abs_w *= config->frequency_hz / PRECISION(SPEED_OF_LIGHT); // scaled from meters to wavelengths
		config->max_abs_w *= config->frequency_hz / PRECISION(SPEED_OF_LIGHT); // scaled from meters to wavelengths
		double x0 = -0.5 * PRECISION(config->image_size) * config->cell_size_rad;
		double y0 = -0.5 * PRECISION(config->image_size) * config->cell_size_rad;
		double nmin = sqrt(MAX(1.0 - x0*x0 - y0*y0, 0.0)) - 1.0;

		if (x0*x0 + y0*y0 > 1.0)
			nmin = -sqrt(abs(1.0 - x0*x0 - y0*y0)) - 1.0;

		config->w_scale = 0.25 / abs(nmin); // scaling factor for converting w coord to signed w grid index
		config->nifty_config.num_total_w_grids = (uint32_t) ((config->max_abs_w - config->min_abs_w) / config->w_scale + 2); //  number of w grids required
		config->w_scale = 1.0 / ((1.0 + 1e-13) * (config->max_abs_w - config->min_abs_w) / (config->nifty_config.num_total_w_grids - 1));
		config->nifty_config.min_plane_w = config->min_abs_w - (0.5*(PRECISION)config->nifty_config.support-1.0) / config->w_scale;
		config->nifty_config.max_plane_w = config->max_abs_w + (0.5*(PRECISION)config->nifty_config.support-1.0) / config->w_scale;
		config->nifty_config.num_total_w_grids += config->nifty_config.support - 2;
		/****************************************************************************************
		 * DO NOT MODIFY ABOVE - DO NOT MODIFY ABOVE - DO NOT MODIFY ABOVE - DO NOT MODIFY ABOVE 
		 ***************************************************************************************/
#else
		config->grid_size = (int) ceil(config->image_size * config->grid_size_padding_scalar); 
		config->w_scale  = pow(config->num_kernels - 1, 2.0) / config->max_abs_w;
# endif

	//To ensure gridsize is divisible by two because also need to have a "half grid size" integer
	if(config->grid_size % 2 != 0)
		config->grid_size -= 1;

	config->uv_scale = config->cell_size_rad * (PRECISION)config->grid_size * config->frequency_hz / SPEED_OF_LIGHT;
	
	const char* prec_format = (SINGLE_PRECISION) ? "flt" : "dbl";
	// file name convention: real/imag/supp minsupp maxsupp os maxw gridsize
	snprintf(config->wproj_real_file, MAX_LEN_CHAR_BUFF * 2, "%swproj_real_minsupp-%d_maxsupp-%d_os-%d_maxw-%.15f_gridsize-%d_prec-%s.bin",
		config->data_input_path, config->min_half_support, config->max_half_support, config->oversampling,
		config->max_abs_w, config->grid_size, prec_format);

	snprintf(config->wproj_imag_file, MAX_LEN_CHAR_BUFF * 2, "%swproj_imag_minsupp-%d_maxsupp-%d_os-%d_maxw-%.15f_gridsize-%d_prec-%s.bin",
		config->data_input_path, config->min_half_support, config->max_half_support, config->oversampling,
		config->max_abs_w, config->grid_size, prec_format);

	snprintf(config->wproj_supp_file, MAX_LEN_CHAR_BUFF * 2, "%swproj_supp_minsupp-%d_maxsupp-%d_os-%d_maxw-%.15f_gridsize-%d_prec-%s.bin",
		config->data_input_path, config->min_half_support, config->max_half_support, config->oversampling,
		config->max_abs_w, config->grid_size, prec_format);

#if SOLVER == NIFTY_GRIDDING
    printf("Min w plane: %f\n", config->nifty_config.min_plane_w);
    printf("Max w plane: %f\n", config->nifty_config.max_plane_w);
    printf("Num planes: %d\n", config->nifty_config.num_total_w_grids);
    printf("Beta: %f\n", config->nifty_config.beta);
#else
	printf(">>> Min half support: %d\n", config->min_half_support);
	printf(">>> Max half support: %d\n", config->max_half_support);
	printf(">>> Num w planes: %d\n", config->num_kernels);
	printf(">>> Oversampling: %d\n", config->oversampling);
	printf(">>> Kernels from file?: %d\n", config->load_kernels_from_file);
	printf(">>> Save kernels to file?: %d\n", config->save_kernels_to_file);
	printf(">>> WProj real file: %s\n", config->wproj_real_file);
	printf(">>> WProj imag file: %s\n", config->wproj_imag_file);
	printf(">>> WProj supp file: %s\n\n", config->wproj_supp_file);
#endif
    printf("Field of view (deg): %f\n", config->fov_deg);
    printf("Cell size (rad): %f\n", config->cell_size_rad);
    printf("Grid size: %d\n", config->grid_size);
    printf("UV scale: %f\n", config->uv_scale);
    printf("W scale: %f\n", config->w_scale);
	printf("Min w: %f\n", config->min_abs_w);
    printf("Max w: %f\n", config->max_abs_w);
}

bool init_config_from_file(Config *config, char* config_file, bool required)
{
	// Parse YAML document into config struct
	struct fy_document *fyd = NULL;
	fyd = fy_document_build_from_file(NULL, config_file);

	if(!fyd)
	{
		printf("ERROR >>> Unable to locate YAML based configuration file. Specified file location: %s...\n\n", config_file);
		return false;
	}
	else
	{
		printf("UPDATE >>> Successfully located YAML based configuration file, parsing...\n\n");

		if(!parse_config_attribute(fyd, "NUM_RECEIVERS", "%d", &(config->num_recievers), required) && required)
			return false;
		
		if(!parse_config_attribute(fyd, "IMAGE_SIZE", "%d", &(config->image_size), required) && required) 
			return false;
		
		if(!parse_config_attribute(fyd, "FOV_DEG", "%lf", &(config->fov_deg), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "VIS_INTENSITY_FILE", "%s", config->visibility_source_file, required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "VIS_UVW_FILE", "%s", config->visibility_source_UVW_file, required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "MIN_HALF_SUPPORT", "%d", &(config->min_half_support), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "MAX_HALF_SUPPORT", "%d", &(config->max_half_support), required) && required)
			return false;

		if(!parse_config_attribute(fyd, "NUM_W_PLANES", "%d", &(config->num_kernels), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "OVERSAMPLING", "%d", &(config->oversampling), required) && required)
			return false;

		if(!parse_config_attribute(fyd, "MIN_ABS_W", "%lf", &(config->min_abs_w), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "MAX_ABS_W", "%lf", &(config->max_abs_w), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "INPUT_DATA_PATH", "%s", config->data_input_path, required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "OUTPUT_DATA_PATH", "%s", config->data_output_path, required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "NUM_MAJOR_CYCLES", "%d", &(config->num_major_cycles), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "GRID_PADDING_SCALAR", "%lf", &(config->grid_size_padding_scalar), required) && required)
			return false;

		if(!parse_config_attribute(fyd, "DEFAULT_GAINS_FILE", "%s", config->default_gains_file, required) && required)
			return false;

		if(!parse_config_attribute(fyd, "MAX_CALIBRATION_CYCLES", "%d", &(config->max_calibration_cycles), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "NUM_OF_CALIBRATION_CYCLES", "%d", &(config->number_cal_major_cycles), required) && required) 
			return false;
			
		if(!parse_config_attribute(fyd, "FREQUENCY_HZ", "%lf", &(config->frequency_hz), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "NUM_MINOR_CYCLES_CALIBRATION", "%d", &(config->number_minor_cycles_cal), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "NUM_MINOR_CYCLES_IMAGING", "%d", &(config->number_minor_cycles_img), required) && required) 
			return false;
		
		if(!parse_config_attribute(fyd, "LOOP_GAIN", "%lf", &(config->loop_gain), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "WEAK_SOURCE_THRESHOLD_CALIBRATION", "%lf", &(config->weak_source_percent_gc), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "WEAK_SOURCE_THRESHOLD_IMAGING", "%lf", &(config->weak_source_percent_img), required) && required) 
			return false;
		
		if(!parse_config_attribute(fyd, "NOISE_FACTOR", "%lf", &(config->noise_factor), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "SEARCH_REGION_PERCENT", "%lf", &(config->search_region_percent), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "WEIGHTING_SCHEME", "%d", &(config->weighting), required) && required) 
			return false;

		if(!parse_config_attribute(fyd, "ROBUSTNESS", "%lf", &(config->robustness), required) && required) 
			return false;

		//NEW NIFTY SPECIFIC CONFIGS
		if(!parse_config_attribute_bool(fyd, "NIFTY_PERFORM_SHIFT_FFT", "%d", &(config->nifty_config.perform_shift_fft), required) && required)
			return false;
		if(!parse_config_attribute(fyd, "NIFTY_UPSAMPLING", "%lf", &(config->nifty_config.upsampling), required) && required) 
			return false;
		if(!parse_config_attribute(fyd, "NIFTY_SUPPORT", "%d", &(config->nifty_config.support), required) && required) 
			return false;
		if(!parse_config_attribute(fyd, "NIFTY_BETA", "%lf", &(config->nifty_config.beta), required) && required) 
			return false;
		if(!parse_config_attribute(fyd, "NIFTY_NUM_W_GRIDS_TO_BATCH", "%d", &(config->nifty_config.num_w_grids_batched), required) && required) 
			return false;

		// Optional - requires bool hack (no bool string modifier in C)
		if(!parse_config_attribute_bool(fyd, "RETAIN_DEVICE_MEM", "%d", &(config->retain_device_mem), required) && required)
			return false;

		if(!parse_config_attribute_bool(fyd, "RIGHT_ASCENSION", "%d", &(config->right_ascension), required) && required)
			return false;

		if(!parse_config_attribute_bool(fyd, "SAVE_DIRTY_IMAGES", "%d", &(config->save_dirty_image), required) && required)
			return false;
			
		if(!parse_config_attribute_bool(fyd, "SAVE_RESIDUAL_IMAGES", "%d", &(config->save_residual_image), required) && required)
			return false;

		if(!parse_config_attribute_bool(fyd, "SAVE_EXTRACTED_SOURCES", "%d", &(config->save_extracted_sources), required) && required)
			return false;

		if(!parse_config_attribute_bool(fyd, "SAVE_PREDICTED_VIS", "%d", &(config->save_predicted_visibilities), required) && required)
			return false;

		if(!parse_config_attribute_bool(fyd, "SAVE_ESTIMATED_GAINS", "%d", &(config->save_estimated_gains), required) && required)
			return false;

		if(!parse_config_attribute_bool(fyd, "USE_DEFAULT_GAINS", "%d", &(config->use_default_gains), required) && required)
			return false;

		if(!parse_config_attribute_bool(fyd, "PERFORM_GAIN_CALIBRATION", "%d", &(config->perform_gain_calibration), required) && required)
			return false;

		if(!parse_config_attribute_bool(fyd, "LOAD_KERNELS_FROM_FILE", "%d", &(config->load_kernels_from_file), required) && required)
			return false;

		if(!parse_config_attribute_bool(fyd, "SAVE_KERNELS_TO_FILE", "%d", &(config->save_kernels_to_file), required) && required)
			return false;
	}

	fy_document_destroy(fyd);
	return true;
}

void clean_up(Host_Mem_Handles *host)
{
    printf("UPDATE >>> Cleaning up all allocated host memory...\n");
    if((*host).kernels)         free((*host).kernels);
    if((*host).kernel_supports) free((*host).kernel_supports);
    if((*host).dirty_image)     free((*host).dirty_image);
	if((*host).residual_image)  free((*host).residual_image);
	if((*host).weight_map)      free((*host).weight_map);
    if((*host).vis_uvw_coords)  free((*host).vis_uvw_coords);
    if((*host).vis_weights)     free((*host).vis_weights);
    if((*host).visibilities)    free((*host).visibilities);
    if((*host).measured_vis)    free((*host).measured_vis);
    if((*host).h_psf)           free((*host).h_psf);
    if((*host).h_sources)       free((*host).h_sources);
    if((*host).prolate)         free((*host).prolate);
    if((*host).receiver_pairs)  free((*host).receiver_pairs);
    if((*host).h_gains)         free((*host).h_gains);
	
	if((*host).quadrature_nodes)      free((*host).quadrature_nodes);
	if((*host).quadrature_weights)    free((*host).quadrature_weights);
	if((*host).quadrature_kernel)     free((*host).quadrature_kernel);
	

    printf("UPDATE >>> DEALLOCATED THE MEMORY...\n");
    (*host).kernels         = NULL;
    (*host).kernel_supports = NULL;
    (*host).dirty_image     = NULL;
	(*host).residual_image  = NULL;
	(*host).weight_map      = NULL;
    (*host).vis_uvw_coords  = NULL;
    (*host).vis_weights     = NULL;
    (*host).visibilities    = NULL;
    (*host).h_psf           = NULL;
    (*host).h_sources       = NULL;
    (*host).measured_vis    = NULL;
    (*host).prolate         = NULL;
    (*host).receiver_pairs  = NULL;
    (*host).h_gains         = NULL;
	(*host).quadrature_nodes = NULL;
	(*host).quadrature_weights = NULL;
	(*host).quadrature_kernel  = NULL;
}

int exit_and_host_clean(const char *message, Host_Mem_Handles *host_mem)
{
    printf("ERROR >>> %s, exiting...\n\n", message);
    clean_up(host_mem);
    return EXIT_FAILURE;
}

bool kernel_host_setup(Config *config, Host_Mem_Handles *host)
{	
	bool success = false;

	// Assumes w-projection kernels have been pre-generated and are readily available
	if(config->load_kernels_from_file && are_kernel_files_available(config))
	{	
		printf("UPDATE >>> Loading W-Projection convolutional gridding kernels from file...\n\n");
		success = load_kernels_from_file(config, host);
	}
	// Either we are producing a fresh set of kernels, or the provided kernel files were not located
	else
	{
		printf("UPDATE >>> Generating a fresh set of W-Projection convolutional gridding kernels...\n\n");
		success = generate_w_projection_kernels(config, host);
	}

	return success;
}


bool visibility_UVW_host_setup(Config *config, Host_Mem_Handles *host)
{
	char buffer[MAX_LEN_CHAR_BUFF * 2];
	snprintf(buffer, MAX_LEN_CHAR_BUFF, "%s%s", config->data_input_path, config->visibility_source_UVW_file);
	printf("UPDATE >>> Loading visibility UVWs from file %s...\n\n", buffer);
	
	FILE *vis_file = fopen(buffer, "rb");
    if(vis_file == NULL)
    {   
		printf("ERROR >>> Unable to open visibility file %s...\n\n", buffer);
        return false; // unsuccessfully loaded data
    }
	int num_vis = 0;
    fread(&num_vis, sizeof(int), 1, vis_file);
	printf("UPDATE >>> Reading in %d visibility UVW coordinates \n\n",num_vis);
    config->num_visibilities = num_vis;
	
	host->vis_uvw_coords = (VisCoord*) calloc(num_vis, sizeof(VisCoord));
	if(host->vis_uvw_coords == NULL)
    {	printf("ERROR >> Unable to allocate memory for visibility information...\n\n");
        fclose(vis_file);
        return false;
    }
	
	int num_elements_read = fread(host->vis_uvw_coords, sizeof(VisCoord), num_vis, vis_file);
	fclose(vis_file);
	
	if(num_elements_read != config->num_visibilities)
	{	printf("ERROR >>> Reading UVW data %d did not match expected visiblity count %d.. \n", num_elements_read, config->num_visibilities);
		return false;
	}
	
	
	PRECISION maxW_detected = 0.0;
	PRECISION minW_detected = 0.0;
	
	//NEED TO CONVERT UVW to single channel?
	double meters_to_wavelengths = config->frequency_hz / SPEED_OF_LIGHT;
	for(int vis_index = 0; vis_index < config->num_visibilities; ++vis_index)
	{	if(config->right_ascension)  
        {
			host->vis_uvw_coords[vis_index].u *= (PRECISION) -1.0;
			host->vis_uvw_coords[vis_index].w *= (PRECISION) -1.0;
		}
		host->vis_uvw_coords[vis_index].u *= (PRECISION) meters_to_wavelengths;
		host->vis_uvw_coords[vis_index].v *= (PRECISION) meters_to_wavelengths;	
		host->vis_uvw_coords[vis_index].w *= (PRECISION) meters_to_wavelengths;	

		if(host->vis_uvw_coords[vis_index].w > maxW_detected)
			maxW_detected = host->vis_uvw_coords[vis_index].w;
		if(host->vis_uvw_coords[vis_index].w < minW_detected)
			minW_detected = host->vis_uvw_coords[vis_index].w;
	}
	printf("UPDATE >>> MIN W DETECTED AS %f and MAX W DETECTED AS %f ..... \n\n",minW_detected,maxW_detected);
	return true;
}

bool visibility_intensity_host_setup(Config *config, Host_Mem_Handles *host)
{
	char buffer[MAX_LEN_CHAR_BUFF * 2];
	snprintf(buffer, MAX_LEN_CHAR_BUFF, "%s%s", config->data_input_path, config->visibility_source_file);
	printf("UPDATE >>> Loading visibility INTENSITY from file %s...\n\n", buffer);

	FILE *vis_file = fopen(buffer, "rb");
    if(vis_file == NULL)
    {   printf("ERROR >>> Unable to open visibility intensity file %s...\n\n", buffer);
        return false; // unsuccessfully loaded data
    }
    // Configure number of visibilities from file
    int num_vis = 0;
    fread(&num_vis, sizeof(int), 1, vis_file);
    
	//SHOULD CHECK IF ENOUGH ON UVW number??
	if(config->num_visibilities != num_vis)
	{	printf("ERROR >>> Reading Visbility data %d did not match expected visiblity UVW count %d.. \n",num_vis,config->num_visibilities);
		return false;
	}
	printf("UPDATE >>> Need to Read %d visibility intensities from file \n",config->num_visibilities);
    // Allocate memory for incoming visibilities
    host->measured_vis = (VIS_PRECISION2*) calloc(num_vis, sizeof(VIS_PRECISION2));
    host->visibilities = (VIS_PRECISION2*) calloc(num_vis, sizeof(VIS_PRECISION2));
    host->vis_weights  = (VIS_PRECISION*)  calloc(num_vis, sizeof(VIS_PRECISION));

    if(host->measured_vis == NULL || host->visibilities  == NULL || host->vis_weights == NULL)
    {
        printf("ERROR >> Unable to allocate memory for visibility information...\n\n");
        fclose(vis_file);
        return false;
    }

	// Used to load in single/double vis before convert to half precision
	// TODO: Refactor this to load in 3 attributes instead of 2 (real, imaginary, and weight)
	PRECISION2* temp_loader = (PRECISION2*) calloc(num_vis, sizeof(PRECISION2));
    int num_elements_read = fread(temp_loader, sizeof(PRECISION2), num_vis, vis_file);

    if(num_elements_read != config->num_visibilities)
	{	printf("ERROR >>> Reading Visibility data %d did not match expected visiblity count %d.. \n",num_elements_read,config->num_visibilities);
		free(temp_loader);
		temp_loader = NULL;
		return false;
	}
    printf("UPDATE >>> Successfully loaded %d visibilities from file...\n\n", config->num_visibilities);

	// Copy over from single/double to half/single precision
	for(int i = 0; i < config->num_visibilities; i++)
	{
		host->measured_vis[i] = MAKE_VIS_PRECISION2(((VIS_PRECISION) temp_loader[i].x), ((VIS_PRECISION) temp_loader[i].y));
		host->vis_weights[i] = (PRECISION) 1.0; // TODO: Refactor this to be read from file
	}

	free(temp_loader);
	temp_loader = NULL;
    return true;
}

bool allocate_host_images(Config *config, Host_Mem_Handles *host)
{
    host->dirty_image = (PRECISION*) calloc(config->image_size * config->image_size, sizeof(PRECISION));
    if(host->dirty_image == NULL)
        return false;

    host->residual_image = (PRECISION*) calloc(config->image_size * config->image_size, sizeof(PRECISION));
    if(host->residual_image == NULL)
		return false;
		
	if(config->weighting == UNIFORM || config->weighting == ROBUST)
	{
		host->weight_map = (PRECISION*) calloc(config->grid_size * config->grid_size, sizeof(PRECISION));
		if(host->weight_map == NULL)
			return false;
	}

    return true;
}

bool correction_set_up(Config *config, Host_Mem_Handles *host)
{
	// Allocate memory for half prolate spheroidal
	host->prolate = (PRECISION*) calloc(config->grid_size / 2, sizeof(PRECISION));
	if(host->prolate == NULL)
		return false;

	// Calculate prolate spheroidal
	create_1D_half_prolate(host->prolate, config->grid_size);

	return true;
}

void create_1D_half_prolate(PRECISION *prolate, int grid_size)
{
	int grid_half_size = grid_size / 2;

	for(int index = 0; index < grid_half_size; ++index)
	{
		double nu = (double) index / (double) grid_half_size;
		prolate[index] = (PRECISION) prolate_spheroidal(nu);
	}
}

//this function reads in gains either from file (NOT IMPLEMENTED YET)
//or uses default gains of 1.0+0i
bool gains_host_set_up(Config *config, Host_Mem_Handles *host)
{
	host->h_gains = (Complex*) calloc(config->num_recievers, sizeof(Complex));

	if(host->h_gains == NULL)
		return false;

	if(config->use_default_gains)
	{	
		for(int i = 0 ; i < config->num_recievers; ++i)
		{
			host->h_gains[i] = (Complex) {.real = (PRECISION)1.0, .imaginary = (PRECISION)0.0};
		}
	}
	else
	{	
		printf(">>> UPDATE: Loading default gains from file %s ",config->default_gains_file);
		FILE *file_gains = fopen(config->default_gains_file , "r");

		if(!file_gains)
		{	
			printf(">>> ERROR: Unable to LOAD GAINS FILE grid files %s , check file structure exists...\n\n", config->default_gains_file);
			return false;
		}
		double gainsReal = 0.0;
		double gainsImag	= 0.0;
		for(int i = 0 ; i < config->num_recievers; ++i)
		{
		// #if SINGLE_PRECISION
		// 	fscanf(file_gains, "%f %f ", &gainsReal, &gainsImag);
		// #else
			fscanf(file_gains, "%lf %lf ", &gainsReal, &gainsImag);
		//#endif

			host->h_gains[i] = (Complex) {.real = (PRECISION)gainsReal, .imaginary = (PRECISION)gainsImag};
		}
	}

	//allocate receiver pairs
	host->receiver_pairs = (int2*) calloc(config->num_baselines, sizeof(int2));
	if(host->receiver_pairs == NULL)
		return false;

	calculate_receiver_pairs(config, host->receiver_pairs);
	return true;
}

/*
	allocates receiver pairs for each baseline visibility. Assumes the following order
	for receiver indices (0,1), (0,2), (0,3), .... (0,N-1).... (1,2),(1,3) ..... (N-2,N-1)
	where N is number of receivers
*/
void calculate_receiver_pairs(Config *config, int2 *receiver_pairs)
{
	int a = 0;
	int b = 1;

	for(int i=0;i<config->num_baselines;++i)
	{
		//printf(">>>> CREATING RECEIVER PAIR (%d,%d) \n",a,b);
		receiver_pairs[i].x = a;
		receiver_pairs[i].y = b;

		b++;
		if(b>=config->num_recievers)
		{
			a++;
			b = a+1;
		}
	}
}

void save_predicted_visibilities(Config *config, Host_Mem_Handles *host, const int cycle)
{
	char buffer[MAX_LEN_CHAR_BUFF * 3];
	snprintf(buffer, MAX_LEN_CHAR_BUFF*3, "%s%smajor_cycle_%d_predicted_vis.bin", config->data_output_path,config->imaging_output_id, cycle);

    FILE *f = fopen(buffer, "wb");
    printf("UPDATE >>> TRYING to save here %s predicted visibilities...\n\n", buffer);
    if(f == NULL)
    {
        printf("ERROR >>> Unable to save predicted visibilities to file, skipping...\n\n");
        return;
    }

    printf("UPDATE >>> Writing predicted visibilities to file...\n\n");
    fwrite(&(config->num_visibilities), sizeof(int), 1, f);

    PRECISION meters_to_wavelengths = config->frequency_hz / SPEED_OF_LIGHT;
    PRECISION vis_weight = 1.0;
    // Record individual visibilities
    for(int v = 0; v < config->num_visibilities; ++v)
    {
        VisCoord current_uvw = host->vis_uvw_coords[v];
        VIS_PRECISION2 current_vis = host->visibilities[v];

        current_uvw.u /= meters_to_wavelengths;
        current_uvw.v /= meters_to_wavelengths;
        current_uvw.w /= meters_to_wavelengths;

        if(config->right_ascension)
        {
            current_uvw.u *= -1.0;
            current_uvw.w *= -1.0;
        }

        fwrite(&current_uvw, sizeof(VisCoord), 1, f);
        fwrite(&current_vis, sizeof(VIS_PRECISION2), 1, f);
        fwrite(&vis_weight, sizeof(int), 1, f);
    }

    fclose(f);
    printf("UPDATE >>> Predicted visibilities have been successfully written to file...\n\n");
}

void save_extracted_sources(Source *sources, int number_of_sources, const char *path, const char *identifier, const char *output_file, int cycle)
{
	char buffer[MAX_LEN_CHAR_BUFF * 3];
	snprintf(buffer, MAX_LEN_CHAR_BUFF*3, "%s%scycle_%d_%s", path, identifier, cycle, output_file);

	printf("UPDATE >>> Attempting to save sources to %s... \n\n", buffer);

	FILE *file = fopen(buffer, "wb");

	if(file == NULL)
	{
		printf(">>> ERROR: Unable to save sources to file, moving on...\n\n");
		return;
	}

	fwrite(&number_of_sources, sizeof(int), 1, file);
	fwrite(sources, sizeof(Source), number_of_sources, file);
	fclose(file);
}
