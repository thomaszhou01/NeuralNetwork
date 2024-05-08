#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include "layer.h"
#include <vector>
//hello
class NeuralNetwork {
    public:
        NeuralNetwork(unsigned int input_size_);

        void add_layer(unsigned int neuron_count, unsigned int activation_function);

        void train(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& labels, unsigned int epochs, double learning_rate, unsigned int batch_size);

        int predict(std::vector<double>& inputs);
        
    private:
        unsigned int input_size_;
        unsigned int layer_count_;
        std::vector<Layer> layers_;

        std::vector<double> propogate(std::vector<double>& inputs);

        void back_propogate(std::vector<double>& labels, std::vector<double>& inputs);

        void optimize_weights(double learning_rate);
};


#endif
