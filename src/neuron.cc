#include "neuron.h"
#include <stdlib.h>

Neuron::Neuron(unsigned int input_size, unsigned int activation_function): input_size_(input_size), activation_function_(activation_function){
    this->output_ = 0.0f;
    this->weighted_sum_ = 0.0f;
    this->nodeVal_ = 0.0f;
    
    this->bias_ = ((double)rand() / (double)RAND_MAX) * 2 - 1;
    this->weights_.reserve(this->input_size_);

    for(int i = 0; i < this->input_size_; i++){
        this->weights_.push_back(((double)rand() / (double)RAND_MAX) * 2 - 1);
        this->gradient_weights_.emplace_back(0.0f);
    }
}

unsigned int Neuron::get_input_size(){
    return this->input_size_;
}

double Neuron::get_output(){
    return this->output_;
}

double Neuron::get_weighted_sum(){
    return this->weighted_sum_;
}

double Neuron::get_weight_value(unsigned int weight){
    return this->weights_[weight];
}

void Neuron::update_gradient_weights(unsigned int weight, double input){
    this->gradient_weights_[weight] += input * this->nodeVal_;
}

void Neuron::update_gradient_bias(){
    this->gradient_bias_ += this->nodeVal_;
}

void Neuron::reset_nodeVal(){
    this->nodeVal_ = 0.0f;
}

double Neuron::propogate(const std::vector<double>& inputs){
    if(inputs.size() != this->input_size_){
        std::cerr << "Input size does not match neuron input size" << std::endl;
        return 0.0f;
    }

    double sum = this->bias_;

    for (int i = 0; i < this->input_size_; i++){
        sum += inputs[i] * this->weights_[i];
    }
    this->weighted_sum_ = sum;
    this->output_ = this->activation_function(sum);
    return this->output_;
}

void Neuron::update_nodeVal(double nodeVal){
    nodeVal_ += nodeVal;
}

void Neuron::update_weights(unsigned int weight, double learning_rate){
    this->weights_[weight] -= learning_rate * this->gradient_weights_[weight];
    this->gradient_weights_[weight] = 0;
}

void Neuron::update_bias(double learning_rate){
    bias_ -= learning_rate * this->gradient_bias_;
    this->gradient_bias_ = 0;
}


double Neuron::activation_function(double input){
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


double Neuron::activation_function_derivative(double input){
    if(activation_function_ == 0){
        //ReLU
        if(input < 0.0f){
            return 0.0f;
        }
        return 1.0f;
    }
    else {
        //Sigmoid
        double activation = activation_function(input);
        return activation * (1.0f - activation);
    }
}

