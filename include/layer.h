#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include "neuron.h"

class Layer{
    public:
        Layer(unsigned int neuron_count, unsigned int input_size);

        unsigned int get_neuron_count();

        std::vector<float> propogate(const std::vector<float>& inputs);

        std::vector<float> back_propogate_output(std::vector<float>& inputs);

        std::vector<float> back_propogate_hidden(Layer &prev_layer, std::vector<float>& inputs);

        std::vector<float> gradient_descent(std::vector<float>& inputs, float learning_rate);

        float get_cost(std::vector<float>& labels);
    
    private:
        unsigned int input_size_;
        unsigned int neuron_count_;
        
        std::vector<Neuron> neurons_;

        void update_gradients(std::vector<float>& inputs);

        float activation_function(float input);

        float activation_function_derivative(float input);

        float cost_function(float output, float expected_output);

        float cost_function_derivative(float output, float expected_output);
};


#endif
