
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

#include "../fft.h"

void fft_2d(Complex *matrix, int number_channels)
{
    // Calculate bit reversed indices
    int* bit_reverse_indices = (int*) calloc(number_channels, sizeof(int));
    calc_bit_reverse_indices(number_channels, bit_reverse_indices);
    Complex *reverse_buffer = (Complex*) calloc(number_channels * number_channels, sizeof(Complex));
    
    for(int row = 0; row < number_channels; ++row)
        for(int col = 0; col < number_channels; ++col)
        {
            int row_reverse = bit_reverse_indices[row];
            int col_reverse = bit_reverse_indices[col];
            int bit_reverse_index = row_reverse * number_channels + col_reverse;
            int matrix_index = row * number_channels + col;
            reverse_buffer[matrix_index] = matrix[bit_reverse_index];
            // printf("%d -> %d\n", matrix_index, bit_reverse_index);
        }
    
    memcpy(matrix, reverse_buffer, number_channels * number_channels * sizeof(Complex));
    free(reverse_buffer);
    free(bit_reverse_indices);
    
    for(int m = 2; m <= number_channels; m *= 2)
    {
        Complex omegaM = (Complex) {.real = COS(PI * 2.0 / m), .imaginary = SIN(PI * 2.0 / m)};
        
        for(int k = 0; k < number_channels; k += m)
        {
            for(int l = 0; l < number_channels; l += m)
            {
                Complex x = (Complex) {.real = 1.0, .imaginary = 0.0};
                
                for(int i = 0; i < m / 2; i++)
                {
                    Complex y = (Complex) {.real = 1.0, .imaginary = 0.0};
                    
                    for(int j = 0; j < m / 2; j++)
                    {   
                        // Perform 2D butterfly operation in-place at (k+j, l+j)
                        int in00Index = (k+i) * number_channels + (l+j);
                        Complex in00 = matrix[in00Index];
                        int in01Index = (k+i) * number_channels + (l+j+m/2);
                        Complex in01 = cpu_complex_multiply(matrix[in01Index], y);
                        int in10Index = (k+i+m/2) * number_channels + (l+j);
                        Complex in10 = cpu_complex_multiply(matrix[in10Index], x);
                        int in11Index = (k+i+m/2) * number_channels + (l+j+m/2);
                        Complex in11 = cpu_complex_multiply(cpu_complex_multiply(matrix[in11Index], x), y);
                        
                        Complex temp00 = cpu_complex_add(in00, in01);
                        Complex temp01 = cpu_complex_subtract(in00, in01);
                        Complex temp10 = cpu_complex_add(in10, in11);
                        Complex temp11 = cpu_complex_subtract(in10, in11);
                        
                        matrix[in00Index] = cpu_complex_add(temp00, temp10);
                        matrix[in01Index] = cpu_complex_add(temp01, temp11);
                        matrix[in10Index] = cpu_complex_subtract(temp00, temp10);
                        matrix[in11Index] = cpu_complex_subtract(temp01, temp11);
                        y = cpu_complex_multiply(y, omegaM);
                    }
                    x = cpu_complex_multiply(x, omegaM);
                }
            }
        }
    }
    
    for(int row = 0; row < number_channels; ++row)
        for(int col = 0; col < number_channels; ++col)
        {   
            int matrix_index = row * number_channels + col;
            PRECISION reciprocal = 1.0 / (number_channels * number_channels);
            matrix[matrix_index] = cpu_complex_scale(matrix[matrix_index], reciprocal);
        }
}

void calc_bit_reverse_indices(int n, int* indices)
{   
    for(int i = 0; i < n; ++i)
    {
        // Calculate index r to which i will be moved
        unsigned int i_prime = i;
        int r = 0;
        for(int j = 1; j < n; j *= 2)
        {
            int b = i_prime & 1;
            r = (r << 1) + b;
            i_prime = (i_prime >> 1);
        }
        indices[i] = r;
    }
}

void fft_shift_2d(Complex *matrix, int size)
{
    for(int row = 0; row < size; ++row)
        for(int col = 0; col < size; ++col)
        {
            int matrix_index = row * size + col;
            PRECISION scalar = 1 - 2 * ((row + col) & 1);
            matrix[matrix_index] = cpu_complex_scale(matrix[matrix_index], scalar);
        }
}
