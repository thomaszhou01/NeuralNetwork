#include "neuron.h"
#include <stdlib.h>

Neuron::Neuron(unsigned int input_size): input_size_(input_size){
    this->output_ = 0.0f;
    this->bias_ = ((float)rand() / (float)RAND_MAX);
    this->weights_.reserve(this->input_size_);
    this->weighted_sum_ = 0.0f;
    this->gradient_weight_ = 0.0f;
    this->gradient_bias_ = 0.0f;

    for(int i = 0; i < this->input_size_; i++){
        this->weights_.push_back(((float)rand() / (float)RAND_MAX));
    }
}

unsigned int Neuron::get_input_size(){
    return this->input_size_;
}

float Neuron::get_output(){
    return this->output_;
}

float Neuron::get_weighted_sum(){
    return this->weighted_sum_;
}

float Neuron::get_weight_value(unsigned int weight){
    return this->weights_[weight];
}

float Neuron::propogate(const std::vector<float>& inputs){
    if(inputs.size() != this->input_size_){
        std::cerr << "Input size does not match neuron input size" << std::endl;
        return 0.0f;
    }

    float sum = this->bias_;

    for (int i = 0; i < this->input_size_; i++){
        sum += inputs[i] * this->weights_[i];
    }
    this->weighted_sum_ = sum;
    this->output_ = this->activation_function(sum);
    return this->output_;
}

void Neuron::update_gradient_weight(float gradient){
    gradient_weight_ += gradient;
}

void Neuron::update_gradient_bias(float gradient){
    gradient_bias_ += gradient;
}

void Neuron::update_weights(unsigned int weight, float inputs, float learning_rate){
    this->weights_[weight] += learning_rate * gradient_weight_;
}

void Neuron::update_bias(float learning_rate){
    bias_ += learning_rate * gradient_bias_;
}


float Neuron::activation_function(float input){
    //ReLU
    if(input < 0.0f){
        return 0.0f;
    }
    return input;
}


float Neuron::activation_function_derivative(float input){
    //ReLU
    if(input < 0.0f){
        return 0.0f;
    }
    return 1.0f;
}

