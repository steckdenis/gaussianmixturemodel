/*
 * Copyright (c) 2016 Denis Steckelmacher
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "gaussianmixture.h"

#include <stdlib.h>
#include <iostream>

extern "C"
GaussianMixture *GaussianMixture_New(int input_dim,
                                     int output_dim,
                                     float initial_variance,
                                     float max_error)
{
    return new GaussianMixture(input_dim, output_dim, initial_variance, max_error);
}

extern "C"
void GaussianMixture_Del(GaussianMixture *m)
{
    delete m;
}

extern "C"
void GaussianMixture_SetValue(GaussianMixture *m,
                              float *in, int inS,
                              float *val, int valS)
{
    m->setValue(
        Eigen::Map<Eigen::VectorXf, Eigen::Aligned>(in, inS),
        Eigen::Map<Eigen::VectorXf, Eigen::Aligned>(val, valS)
    );
}

extern "C"
void GaussianMixture_GetValue(GaussianMixture *m,
                              float *in, int inS,
                              float *out)
{
    float weight;
    Eigen::VectorXf rs = m->value(
        Eigen::Map<Eigen::VectorXf, Eigen::Aligned>(in, inS),
        weight
    );
    
    memcpy((void *)out, (void *)rs.data(), rs.rows() * sizeof(float));
}

extern "C"
int GaussianMixture_NumClusters(GaussianMixture *m)
{
    return m->numClusters();
}