#include "neuron.h"
#include <stdlib.h>

Neuron::Neuron(unsigned int input_size): input_size_(input_size){
    this->output_ = 0.0f;
    this->bias_ = ((float)rand() / (float)RAND_MAX);
    this->weights_.reserve(this->input_size_);

    for(int i = 0; i < this->input_size_; i++){
        this->weights_.push_back(((float)rand() / (float)RAND_MAX));
    }
}


float Neuron::propogate(std::vector<float>& inputs){
    if(inputs.size() != this->input_size_){
        std::cerr << "Input size does not match neuron input size" << std::endl;
        return 0.0f;
    }

    float sum = this->bias_;

    for (int i = 0; i < this->input_size_; i++){
        sum += inputs[i] * this->weights_[i];
    }

    this->output_ = this->activation_function(sum);
    return this->output_;
}


float Neuron::activation_function(float input){
    //ReLU
    if(input < 0.0f){
        return 0.0f;
    }
    return input;
}
