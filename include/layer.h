#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include <cmath>
#include "neuron.h"

class Layer{
    public:
        Layer(unsigned int neuron_count, unsigned int input_size, unsigned int activation_function);

        unsigned int get_neuron_count();

        std::vector<double> propogate(const std::vector<double>& inputs);

        std::vector<double> back_propogate_output(std::vector<double>& inputs);

        std::vector<double> back_propogate_hidden(Layer &prev_layer, std::vector<double>& inputs);

        std::vector<double> update_gradients(std::vector<double>& inputs);

        std::vector<double> gradient_descent(double learning_rate);

        double get_cost(std::vector<double>& labels);
    
    private:
        unsigned int input_size_;
        unsigned int neuron_count_;
        unsigned int activation_function_;
        
        std::vector<Neuron> neurons_;
        std::vector<double> nodeValues_;

        double activation_function(double input);

        double activation_function_derivative(double input);

        double cost_function(double output, double expected_output);

        double cost_function_derivative(double output, double expected_output);
};


#endif
