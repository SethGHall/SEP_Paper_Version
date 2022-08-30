
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

#include "../wprojection.h"

bool generate_w_projection_kernels(Config *config, Host_Mem_Handles *host)
{
    if(SINGLE_PRECISION)
        printf("INFO >>> Generating W-Projection kernels using single precision...\n\n");
    else
        printf("INFO >>> Generating W-Projection kernels using double precision...\n\n");

    config->total_kernel_samples = 0;
    host->kernel_supports = (int2*) calloc(config->num_kernels, sizeof(int2));
    if(host->kernel_supports == NULL)
        return false;

    int number_w_planes = config->num_kernels;
    int grid_size = (int) floor(config->grid_size / config->grid_size_padding_scalar);
    int image_size = grid_size;
    
    int oversample = config->oversampling;
    int min_support = config->min_half_support;
    int max_support = config->max_half_support;
    
    size_t max_bytes_per_plane = 10 * 1024 * 1024; // automate this based on max support
    
    PRECISION max_uvw  = config->max_abs_w;
    PRECISION w_scale = POW(config->num_kernels - 1, 2.0) / max_uvw;
    PRECISION cell_size =  config->cell_size_rad;
    PRECISION w_to_max_support_ratio = (max_support - min_support) / max_uvw;
    PRECISION fov = cell_size * image_size;

    // Calculate convolution kernel memory requirements
    size_t max_mem_bytes = max_bytes_per_plane * number_w_planes;
    PRECISION max_conv_size = SQRT(max_mem_bytes / (16.0 * number_w_planes));
    // printf("Max conv size: %f\n", max_conv_size);
    int nearest = get_next_pow_2((unsigned int) 2 * (int) (max_conv_size / 2.0));
    printf("Nearest: %d\n", nearest);
    int conv_size = nearest;
    // printf("Conv size: %d\n", conv_size);
    int conv_half_size = conv_size / 2;
    // printf("Conv half size: %d\n", conv_half_size);
    
    int inner = conv_size / oversample;
    PRECISION max_l = SIN(0.5 * fov);
    PRECISION sampling = ((2.0 * max_l * oversample) / image_size) * ((PRECISION) grid_size / (PRECISION) conv_size);

    // printf("Sampling: %f\n", sampling);
    // printf("FOV: %f\n", fov);

    // Intermediate memory needed for generating kernel set
    Complex *kernels = (Complex*) calloc(number_w_planes * conv_half_size * conv_half_size, sizeof(Complex));
    Complex *screen = (Complex*) calloc(conv_size * conv_size, sizeof(Complex));
    PRECISION *maximums = (PRECISION*) calloc(number_w_planes, sizeof(PRECISION));
    PRECISION* taper = (PRECISION*) calloc(inner, sizeof(PRECISION));

    if(kernels == NULL || screen == NULL || maximums == NULL || taper == NULL)
    {
        printf("ERROR >>> Unable to allocate intermediate memory blocks for generating W-Projection kernels...\n\n");
        if(kernels != NULL)  free(kernels);
        if(screen != NULL)   free(screen);
        if(maximums != NULL) free(maximums);
        if(taper != NULL)    free(taper);
        return false;
    }

    populate_ps_window(taper, inner);

    printf("UPDATE >>> Creating W-Projection kernels...\n\n");
    for(int iw = 0; iw < number_w_planes; ++iw)
    {
        printf("UPDATE >>> Generating kernel %d (out of %d)\n", iw, number_w_planes);
        // Zero out screen
        memset(screen, 0, conv_size * conv_size * sizeof(Complex));
        
        // Generate screen
        generate_phase_screen(iw, conv_size, inner, sampling, w_scale, taper, screen);
        
        // printf(">>> UPDATE: Executing Fourier Transform...\n");
        // FFT
        fft_2d(screen, conv_size);
        
        // store maximum
        maximums[iw] = cpu_complex_magnitude(screen[0]);
        // printf("Maximums: %f\n", maximums[iw]);
        
        // printf(">>> UPDATE: Clipping useful quadrant for further processing...\n\n");
        // Clip
        for(int row = 0; row < conv_half_size; ++row)
            for(int col = 0; col < conv_half_size; ++col)
            {
                int offset = iw * conv_half_size * conv_half_size;
                int k_index = offset + row * conv_half_size + col;
                kernels[k_index] = screen[row * conv_size + col];
            }

        // Set kernel support and starting index metadata, update total running samples
        PRECISION w = iw * iw / w_scale;
        PRECISION support = calculate_support(w, config->min_half_support, w_to_max_support_ratio);
        int oversampled_support = (ROUND(support) + 1) * config->oversampling;
        host->kernel_supports[iw].x = support;
        host->kernel_supports[iw].y = config->total_kernel_samples;
        config->total_kernel_samples += oversampled_support * oversampled_support;
    }

    free(taper);
    free(screen);
    
    printf("\nUPDATE >>> Normalizing quadrants by the global maximum (typically peak of plane where w == 0)...\n\n");
    normalize_kernels_by_maximum(kernels, maximums, number_w_planes, conv_half_size);
    free(maximums);
    
    printf("UPDATE >>> Normalizing kernel quadrants to the sum of one...\n\n");
    normalize_kernels_sum_of_one(kernels, number_w_planes, conv_half_size, oversample);
    
    if(config->save_kernels_to_file)
    {
        if(save_kernels_to_file(config, kernels, w_scale, w_to_max_support_ratio, conv_half_size))
            printf("UPDATE >>> W-Projection kernels successfully saved to file...\n\n");
        else
            printf("ERROR >>> Unable to save newly generated W-Projection kernels to file...\n\n");
    }

    printf("UPDATE >>> Binding W-Projection kernels to host memory for later use...\n\n");
    if(!bind_kernels_to_host(config, host, kernels, w_scale, w_to_max_support_ratio, conv_half_size))
    {
        printf("ERROR >>> Unable to bind newly generated kernels to host memory...\n\n");
        if(kernels != NULL) free(kernels);
    }
        printf("UPDATE >>> Kernels successfully bound to host memory...\n\n");

    if(kernels != NULL) free(kernels);
    printf("UPDATE >>> W-Projection kernels successfully generated...\n\n");
    return true;
}

bool save_kernels_to_file(Config *config, Complex *kernels, PRECISION w_scale, 
    PRECISION w_to_max_support_ratio, int conv_half_size)
{
    printf("UPDATE >>> Saving generated W-Projection kernels to file...\n\n");

    FILE *real = fopen(config->wproj_real_file, "wb");
    FILE *imag = fopen(config->wproj_imag_file, "wb");
    FILE *supp = fopen(config->wproj_supp_file, "wb");
    
    if(real == NULL || imag == NULL || supp == NULL)
    {
        if(real != NULL) fclose(real);
        if(imag != NULL) fclose(imag);
        if(supp != NULL) fclose(supp);
        return false;
    }

    for(int iw = 0; iw < config->num_kernels; ++iw)
    {
        PRECISION w = iw * iw / w_scale;
        int support = (int) ROUND(calculate_support(w, config->min_half_support, w_to_max_support_ratio));
        int oversampled_support = (support + 1) * config->oversampling;
        int kernel_offset = iw * conv_half_size * conv_half_size;
        fwrite(&support, sizeof(int), 1, supp);
        
        for(int row = 0; row < oversampled_support; ++row)
        {
            for(int col = 0; col < oversampled_support; ++col)
            {
                int plane_index = kernel_offset + row * conv_half_size + col;
                fwrite(&(kernels[plane_index].real), sizeof(PRECISION), 1, real);
                fwrite(&(kernels[plane_index].imaginary), sizeof(PRECISION), 1, imag);
            }
        }
    }
    
    fclose(supp); fclose(imag); fclose(real);
    return true;
}

void normalize_kernels_sum_of_one(Complex *kernels, int number_w_planes, int conv_half_size, int oversample)
{
    PRECISION sum = 0.0;
    
    for (int iy = -4; iy <= 4; ++iy)
        for (int ix = -4; ix <= 4; ++ix)
            sum += kernels[(abs(ix) * oversample + conv_half_size * (abs(iy) * oversample))].real;
    
    unsigned int number_of_samples = number_w_planes * conv_half_size * conv_half_size;
    for(unsigned int index = 0; index < number_of_samples; ++index)
        kernels[index] = cpu_complex_scale(kernels[index], 1.0 / sum);
}

void normalize_kernels_by_maximum(Complex *kernels, PRECISION *maximums, int number_w_planes, int conv_half_size)
{
    PRECISION maximum = -PRECISION_MAX;
    for (int iw = 0; iw < number_w_planes; ++iw)
        maximum = MAX(maximum, maximums[iw]);
    
    unsigned int number_of_samples = number_w_planes * conv_half_size * conv_half_size;
    for(unsigned int index = 0; index < number_of_samples; ++index)
        kernels[index] = cpu_complex_scale(kernels[index], 1.0 / maximum);
}

void generate_phase_screen(int iw, int conv_size, int inner, PRECISION sampling, PRECISION w_scale, PRECISION* taper, Complex *screen)
{
    PRECISION f = (2.0 * PI * iw * iw) / w_scale;
    int inner_half = inner / 2;
    
    for(int iy = -inner_half; iy < inner_half; ++iy)
    {
        PRECISION taper_y = taper[iy + inner_half];
        PRECISION m = sampling * (PRECISION) iy;
        PRECISION msq = m*m;
        int offset = (iy > -1 ? iy : (iy + conv_size)) * conv_size;
        
        for(int ix = -inner_half; ix < inner_half; ++ix)
        {
            PRECISION l = sampling * (PRECISION) ix;
            PRECISION rsq = l * l + msq;
            if (rsq < 1.0) {
                PRECISION taper_x = taper[ix + inner_half];
                PRECISION taper = taper_x * taper_y;
                int index = (offset + (ix > -1 ? ix : (ix + conv_size)));
                PRECISION phase = f * (SQRT(1.0 - rsq) - 1.0);
                screen[index] = (Complex) {
                    .real = taper * COS(phase),
                    .imaginary = taper * SIN(phase)
                };
            }
        }
    }
}

PRECISION calculate_support(PRECISION w, int min_support, PRECISION w_max_support_ratio)
{   
    return ABS(w_max_support_ratio * w) + min_support;
}

// Suitable for 32-bit unsigned integers
unsigned int get_next_pow_2(unsigned int x)
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    
    return x;
}

void populate_ps_window(PRECISION *window, int size)
{
    for(int index = 0; index < size; ++index)
    {
        PRECISION nu = ABS(calculate_window_stride(index, size));
        window[index] = prolate_spheroidal(nu);
    }
}

PRECISION calculate_window_stride(int index, int size)
{
    return (index - size / 2) / ((PRECISION) size / 2.0);
}

// Calculates a sample on across a prolate spheroidal
// Note: this is the Fred Schwabb approximation technique
double prolate_spheroidal(double nu)
{
    static double p[] = {0.08203343, -0.3644705, 0.627866, -0.5335581, 0.2312756,
        0.004028559, -0.03697768, 0.1021332, -0.1201436, 0.06412774};
    static double q[] = {1.0, 0.8212018, 0.2078043,
        1.0, 0.9599102, 0.2918724};

    int part = 0;
    int sp = 0;
    int sq = 0;
    double nuend = 0.0;
    double delta = 0.0;
    double top = 0.0;
    double bottom = 0.0;

    if(nu >= 0.0 && nu < 0.75)
    {
        part = 0;
        nuend = 0.75;
    }
    else if(nu >= 0.75 && nu < 1.0)
    {
        part = 1;
        nuend = 1.0;
    }
    else
        return 0.0;

    delta = nu * nu - nuend * nuend;
    sp = part * 5;
    sq = part * 3;
    top = p[sp];
    bottom = q[sq];

    for(int i = 1; i < 5; i++)
        top += p[sp+i] * pow(delta, i);
    for(int i = 1; i < 3; i++)
        bottom += q[sq+i] * pow(delta, i);
    return (bottom == 0.0) ? 0.0 : top/bottom;
}

bool bind_kernels_to_host(Config *config, Host_Mem_Handles *host, Complex* kernels,
    PRECISION w_scale, PRECISION w_to_max_support_ratio, int conv_half_size)
{
    // Allocate host mem for storing generated kernels
    host->kernels = (VIS_PRECISION2*) calloc(config->total_kernel_samples, sizeof(VIS_PRECISION2));
    if(host->kernels == NULL)
    {   
        printf("ERROR >>> Unable to allocate host memory for storing kernels...\n\n");
        return false;
    }

    // Copy over oversampled kernels, discarding padding from kernel creation process
    int flattened_kernels_index = 0;
    for(int iw = 0; iw < config->num_kernels; ++iw)
    {
        PRECISION w = iw * iw / w_scale;
        PRECISION support = calculate_support(w, config->min_half_support, w_to_max_support_ratio);
        int oversampled_support = (ROUND(support) + 1) * config->oversampling;
        int kernel_offset = iw * conv_half_size * conv_half_size;

        for(int row = 0; row < oversampled_support; ++row)
        {
            for(int col = 0; col < oversampled_support; ++col)
            {
                Complex sample = kernels[kernel_offset + row * conv_half_size + col];
                VIS_PRECISION2 downscaled = MAKE_VIS_PRECISION2((VIS_PRECISION) sample.real, (VIS_PRECISION) sample.imaginary);
                host->kernels[flattened_kernels_index++] = downscaled;
            }
        }
    }

    return true;
}

bool load_kernels_from_file(Config *config, Host_Mem_Handles *host)
{
    // Need to load kernel support file first, 
    host->kernel_supports = (int2*) calloc(config->num_kernels, sizeof(int2));
    if(host->kernel_supports == NULL)
        return false;

    printf("UPDATE >>> Loading kernel support file from %s...\n\n", config->wproj_supp_file);

    FILE *kernel_support_file = fopen(config->wproj_supp_file,"rb");

    if(kernel_support_file == NULL)
        return false;
    
    config->total_kernel_samples = 0;
    
    for(int plane_num = 0; plane_num < config->num_kernels; ++plane_num)
    {
        fread(&(host->kernel_supports[plane_num].x), sizeof(int), 1, kernel_support_file);
        host->kernel_supports[plane_num].y = config->total_kernel_samples;
        config->total_kernel_samples += (int)pow((host->kernel_supports[plane_num].x + 1) * config->oversampling, 2.0);
    }
    
    fclose(kernel_support_file);
    
    printf("UPDATE >>> Total number of samples needed to store kernels is %d...\n\n", config->total_kernel_samples);

    printf("UPDATE >>> Loading kernel files file from %s real and %s imaginary...\n\n",
        config->wproj_real_file, config->wproj_imag_file);

    // now load kernels into CPU memory
    FILE *kernel_real_file = fopen(config->wproj_real_file, "rb");
    FILE *kernel_imag_file = fopen(config->wproj_imag_file, "rb");
    
    if(!kernel_real_file || !kernel_imag_file)
    {
        if(kernel_real_file) fclose(kernel_real_file);
        if(kernel_imag_file) fclose(kernel_imag_file);
        return false; // unsuccessfully loaded data
    }

    host->kernels = (VIS_PRECISION2*) calloc(config->total_kernel_samples, sizeof(VIS_PRECISION2));
    if(host->kernels == NULL)
        return false;

    int kernel_index = 0;
    for(int plane_num = 0; plane_num < config->num_kernels; ++plane_num)
    {
        int number_samples_in_kernel = (int) pow((host->kernel_supports[plane_num].x + 1) * config->oversampling, 2.0);

        for(int sample_number = 0; sample_number < number_samples_in_kernel; ++sample_number)
        {	
            fread(&(host->kernels[kernel_index].x), sizeof(PRECISION), 1, kernel_real_file);
            fread(&(host->kernels[kernel_index].y), sizeof(PRECISION), 1, kernel_imag_file);
            kernel_index++;
        }
    }

    fclose(kernel_real_file);
    fclose(kernel_imag_file);	
    return true;
}

bool are_kernel_files_available(Config *config)
{
    FILE *real = fopen(config->wproj_real_file, "rb");
    FILE *imag = fopen(config->wproj_imag_file, "rb");
    FILE *supp = fopen(config->wproj_supp_file, "rb");

    bool available = (real != NULL) && (imag != NULL) && (supp != NULL);

    if(real != NULL) fclose(real);
    if(imag != NULL) fclose(imag);
    if(supp != NULL) fclose(supp);

    if(!available)
        printf("ERROR >>> Unable to locate specified W-Projection convolutional gridding kernels from file...\n\n");
    else
        printf("UPDATE >>> Successfully located pre-generated W-Projection convolutional gridding kernels...\n\n");

    return available;
}