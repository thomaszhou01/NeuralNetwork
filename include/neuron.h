#ifndef NEURON_H_
#define NEURON_H_

#include <iostream>
#include <vector>

class Neuron {
    public:
        Neuron(unsigned int input_size);

        unsigned int get_input_size();

        float get_output();

        float get_weighted_sum();

        float get_weight_value(unsigned int weight);

        float propogate(const std::vector<float>& inputs);

        void update_gradient_weight(float gradient);

        void update_gradient_bias(float gradient);

        void update_weights(unsigned int weight, float inputs, float learning_rate);

        void update_bias(float learning_rate);

    
    private:
        float bias_;
        float weighted_sum_;
        float output_;
        //need gradient for weights and biases
        float gradient_weight_;
        float gradient_bias_;

        unsigned int input_size_;
        std::vector<float> weights_;

        float activation_function(float input);

        float activation_function_derivative(float input);
};

#endif
