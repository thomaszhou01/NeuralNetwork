#ifndef NEURON_H_
#define NEURON_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

class Neuron {
    public:
        Neuron(unsigned int input_size, unsigned int activation_function);

        unsigned int get_input_size();

        float get_output();

        float get_weighted_sum();

        float get_weight_value(unsigned int weight);

        void reset_nodeVal();

        float propogate(const std::vector<float>& inputs);

        void update_nodeVal(float nodeVal);

        void update_weights(unsigned int weight, float input, float learning_rate);

        void update_bias(float learning_rate);

    
    private:
        float bias_;
        float weighted_sum_;
        float output_;
        unsigned int activation_function_;
        //need gradient for weights and biases
        float nodeVal_;

        unsigned int input_size_;
        std::vector<float> weights_;

        float activation_function(float input);

        float activation_function_derivative(float input);
};

#endif
