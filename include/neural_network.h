#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include "layer.h"
#include <vector>

class NeuralNetwork {
    public:
        NeuralNetwork(unsigned int input_size_);

        void add_layer(unsigned int neuron_count, unsigned int activation_function);

        void train(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& labels, unsigned int epochs, float learning_rate);

        
    private:
        unsigned int input_size_;
        unsigned int layer_count_;
        std::vector<Layer> layers_;

        std::vector<float> propogate(std::vector<float>& inputs);

        void back_propogate(std::vector<float>& inputs, std::vector<float>& labels, float learning_rate);
};


#endif
