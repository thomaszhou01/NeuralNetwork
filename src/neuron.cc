#include "neuron.h"
#include <stdlib.h>

Neuron::Neuron(unsigned int input_size, unsigned int activation_function): input_size_(input_size), activation_function_(activation_function){
    this->output_ = 0.0f;
    this->weighted_sum_ = 0.0f;
    this->nodeVal_ = 0.0f;
    
    this->bias_ = ((float)rand() / (float)RAND_MAX) * 2 - 1;
    this->weights_.reserve(this->input_size_);

    for(int i = 0; i < this->input_size_; i++){
        this->weights_.push_back(((float)rand() / (float)RAND_MAX) * 2 - 1);
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

void Neuron::reset_nodeVal(){
    this->nodeVal_ = 0.0f;
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

void Neuron::update_nodeVal(float nodeVal){
    nodeVal_ += nodeVal;
}

void Neuron::update_weights(unsigned int weight, float input, float learning_rate){
    this->weights_[weight] -= learning_rate * nodeVal_ * input;
}

void Neuron::update_bias(float learning_rate){
    bias_ -= learning_rate * nodeVal_;
}


float Neuron::activation_function(float input){
    if(activation_function_ == 0){
        //ReLU
        if(input < 0.0f){
            return 0.0f;
        }
        return input;
    }
    else {
        //Sigmoid
        return 1.0f / (1.0f + exp(-input));
    }
}


float Neuron::activation_function_derivative(float input){
    if(activation_function_ == 0){
        //ReLU
        if(input < 0.0f){
            return 0.0f;
        }
        return 1.0f;
    }
    else {
        //Sigmoid
        float activation = activation_function(input);
        return activation * (1.0f - activation);
    }
}

