
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

#include "../complex.h"

Complex cpu_complex_add(Complex z1, Complex z2)
{
    return (Complex) {
        .real = z1.real + z2.real,
        .imaginary = z1.imaginary + z2.imaginary
    };
}

Complex cpu_complex_subtract(Complex z1, Complex z2)
{
    return (Complex) {
        .real = z1.real - z2.real,
        .imaginary = z1.imaginary - z2.imaginary
    };
}

Complex cpu_complex_multiply(Complex z1, Complex z2)
{
    return (Complex) {
        .real = z1.real * z2.real - z1.imaginary * z2.imaginary,
        .imaginary = z1.imaginary * z2.real + z1.real * z2.imaginary
    };
}

Complex cpu_complex_scale(Complex z, PRECISION scalar)
{
    return (Complex) {
        .real = z.real * scalar,
        .imaginary = z.imaginary * scalar
    };
}

Complex cpu_complex_conj_exp(PRECISION phase)
{
    return (Complex) {
        .real = COS(2.0 * PI * phase),
        .imaginary = -SIN(2.0 * PI * phase)
    };
}

PRECISION cpu_complex_magnitude(Complex z)
{
    return SQRT(z.real * z.real + z.imaginary * z.imaginary);
}