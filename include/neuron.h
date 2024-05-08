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

        double get_output();

        double get_weighted_sum();

        double get_weight_value(unsigned int weight);

        void update_gradient_weights(unsigned int weight, double input);

        void update_gradient_bias();

        void reset_nodeVal();

        double propogate(const std::vector<double>& inputs);

        void update_nodeVal(double nodeVal);

        void update_weights(unsigned int weight, double learning_rate);

        void update_bias(double learning_rate);

    
    private:
        double bias_;
        double weighted_sum_;
        double output_;
        unsigned int activation_function_;
        //need gradient for weights and biases
        double nodeVal_;

        std::vector<double> gradient_weights_;
        double gradient_bias_;

        unsigned int input_size_;
        std::vector<double> weights_;

        double activation_function(double input);

        double activation_function_derivative(double input);
};

#endif
