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