#include "neural_network.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(unsigned int input_size_): input_size_(input_size_){
    std::cout << "Neural Network Created!" << std::endl;
    layer_count_ = 0;
}

void NeuralNetwork::add_layer(unsigned int neuron_count){
    unsigned int prev_layer_size = this->input_size_;

    if (this->layer_count_ != 0) {
        prev_layer_size = this->layers_[this->layer_count_ - 1].get_neuron_count();
    }

    this->layers_.push_back(Layer(neuron_count, prev_layer_size));
    this->layer_count_++;
    std::cout << "Layer Added!" << std::endl;
}  
