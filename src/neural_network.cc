#include "neural_network.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(unsigned int input_size_): input_size_(input_size_){
    std::cout << "Neural Network Created!" << std::endl;
    layer_count_ = 0;
}

void NeuralNetwork::add_layer(unsigned int neuron_count, unsigned int activation_function){
    unsigned int prev_layer_size = this->input_size_;

    if (this->layer_count_ != 0) {
        prev_layer_size = this->layers_[this->layer_count_ - 1].get_neuron_count();
    }

    this->layers_.push_back(Layer(neuron_count, prev_layer_size, activation_function));
    this->layer_count_++;
    std::cout << "Layer Added!" << std::endl;
}  

void NeuralNetwork::train(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& labels, unsigned int epochs, float learning_rate){
    std::cout << "Training Started" << std::endl;

    for(int i = 0; i < epochs; i++){

        std::cout << "Epoch: " << i << std::endl;

        float cost = 0.0f;
        for(int j = 0; j < inputs.size(); j++){
            std::vector<float> outputs = this->propogate(inputs[j]);
            
            float iteration_cost = this->layers_[this->layer_count_-1].get_cost(labels[j]);

            this->back_propogate(outputs, labels[j], learning_rate);

            cost += iteration_cost;
        }

        cost = cost / labels.size();
        std::cout << "Cost: " << cost << std::endl;

    }
}


std::vector<float> NeuralNetwork::propogate(std::vector<float>& inputs){
    std::vector<float> outputs = inputs;

    for(int i = 0; i < this->layer_count_; i++){
        outputs = this->layers_[i].propogate(outputs);
    }

    return outputs;
}

void NeuralNetwork::back_propogate(std::vector<float>& inputs, std::vector<float>& labels, float learning_rate){
    std::vector<float> outputs = labels;

    outputs = this->layers_[this->layer_count_-1].back_propogate_output(outputs);

    for(int i = this->layer_count_ - 2; i >= 0; i--){
        outputs = this->layers_[i].back_propogate_hidden(this->layers_[i+1], outputs);
    }

    for(int i = 0; i < this->layer_count_; i++){
        inputs = this->layers_[i].gradient_descent(inputs, learning_rate);
    }
}
