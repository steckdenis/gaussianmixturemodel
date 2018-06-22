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

#ifndef __GAUSSIANMIXTURE_H__
#define __GAUSSIANMIXTURE_H__

#include <Eigen/Dense>
#include <vector>

class NetworkSerializer;

/**
 * @brief Function approximator based on an incremental gaussian mixture model
 */
class GaussianMixture
{
    public:
        /**
         * @param initial_variance Initial variance of the input data (usually 0.01)
         * @param max_error Maximum error tolerated by the model, used to decide
         *                  when to increase its precision and when to simplify it.
         */
        GaussianMixture(unsigned int input_dim,
                unsigned int output_dim,
                float initial_variance,
                float max_error);
        ~GaussianMixture();

        /**
         * @brief Set the value of a point
         */
        void setValue(const Eigen::VectorXf &input, const Eigen::VectorXf &value);

        /**
         * @brief Get the value of a point
         *
         * See AbstractController::predict for a description of what @p weight
         * represents.
         */
        Eigen::VectorXf value(const Eigen::VectorXf &input, float &weight) const;
        
        /**
         * @brief Number of clusters in the model
         */
        int numClusters() const;

    private:
        /**
         * @brief Neuron activated in response to inputs and outputs
         */
        struct Neuron
        {
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign> Matrix;

            // Make a new neuron centered at an input/output tuple
            Neuron(const Eigen::VectorXf &input,
                   const Eigen::VectorXf &output,
                   float initial_variance);

            Eigen::MatrixXf _covariance_in_out;
            Eigen::MatrixXf _covariance_in;
            Matrix _inv_covariance_in;
            Eigen::VectorXf _mean_in;
            Eigen::VectorXf _mean_out;

            Eigen::LLT<Eigen::MatrixXf> _in_llt;

            float _score;                           // Score used to detect when a neuron becomes useless

            // Variable containing log(1/(sqrt((2*pi*|covariance|)^D))
            float _log_input_gaussian_normalization;

            // Variables holding temporary values computed for each input or output
            float _probability_of_in;           // p(input|neuron)
            float _square_mahalanobis_distance; // delta*inv_cov*delta
            float _probability_cond_in;         // p(neuron|input)

            // Methods used to update the temporary values
            void updateInGaussianNorm();
            void computeProbabilityOfIn(const Eigen::VectorXf &input);
            float inProba() const;
            float squaredMahalanobisDistance() const;
            void computeProbabilityCond(float sum_of_in_probas);    // Compute p(neuron|input), require sum(neuron){ neuron->inProba() }

            void contributeToOutput(Eigen::VectorXf &output,
                                   const Eigen::VectorXf &input);

            void update(const Eigen::VectorXf &input,
                        const Eigen::VectorXf &output);
        };

        std::vector<Neuron *> _neurons;

        // Keep track of the ranges of the input and output
        Eigen::VectorXf _min_in;
        Eigen::VectorXf _max_in;
        Eigen::VectorXf _min_out;
        Eigen::VectorXf _max_out;

        unsigned int _input_dim;
        unsigned int _output_dim;
        float _initial_variance;
        float _max_error;
};

#endif
