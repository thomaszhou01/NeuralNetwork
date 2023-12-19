#include "layer.h"

Layer::Layer(unsigned int neuron_count, unsigned int input_size): input_size_(input_size), neuron_count_(neuron_count){
    this->neurons_.reserve(neuron_count);

    for(int i = 0; i < neuron_count; i++){
        this->neurons_.push_back(Neuron(input_size));
    }
}

unsigned int Layer::get_neuron_count(){
    return this->neuron_count_;
}

std::vector<float> Layer::propogate(const std::vector<float>& inputs){
    std::vector<float> outputs;
    outputs.reserve(this->neuron_count_);

    for(int i = 0; i < this->neuron_count_; i++){
        outputs.push_back(this->neurons_[i].propogate(inputs));
    }

    return outputs;
}

//takes expected values
std::vector<float> Layer::back_propogate_output(std::vector<float>& inputs){
    std::vector<float> outputs;
    outputs.reserve(this->neuron_count_);

    for (int i = 0; i < this->neuron_count_; i++){
        float output = this->cost_function_derivative(this->neurons_[i].get_output(), inputs[i]) * this->activation_function_derivative(this->neurons_[i].get_weighted_sum());
        outputs.push_back(output);
    }

    this->update_gradients(outputs);

    return outputs;
}

std::vector<float> Layer::back_propogate_hidden(Layer &prev_layer, std::vector<float>& inputs){
    std::vector<float> outputs;
    outputs.reserve(this->neuron_count_);

    for (int i = 0; i < this->neuron_count_; i++){
        float output = 0.0f;
        for (int j = 0; j < inputs.size(); j++){
            float weighted = prev_layer.neurons_[j].get_weight_value(j);
            output += inputs[j] * weighted;
        }
        output *= this->activation_function_derivative(this->neurons_[i].get_weighted_sum());
        outputs.push_back(output);
    }

    this->update_gradients(outputs);

    return outputs;
}

std::vector<float> Layer::gradient_descent(std::vector<float>& inputs, float learning_rate){
    std::vector<float> outputs;

    for (int i = 0; i < this->neuron_count_; i++){
        for(int j = 0; j < inputs.size(); j++){
            this->neurons_[i].update_weights(j, inputs[j], learning_rate);
        }

        this->neurons_[i].update_bias(learning_rate);
        outputs.push_back(this->neurons_[i].get_output());
    }

    return outputs;
}

float Layer::get_cost(std::vector<float>& labels){
    float cost = 0.0f;

    for(int i = 0; i < this->neuron_count_; i++){
        cost += (neurons_[i].get_output() - labels[i]) * (neurons_[i].get_output() - labels[i]);
    }

    return cost;
}

void Layer::update_gradients(std::vector<float>& inputs){
    for(int i = 0; i < this->neuron_count_; i++){
        for(int j = 0; j < this->neurons_[0].get_input_size(); j++){
            this->neurons_[i].update_gradient_weight(inputs[i] * this->neurons_[i].get_weight_value(j));
        }
        this->neurons_[i].update_gradient_bias(inputs[i]);
    }
}


float Layer::activation_function(float input){
    //ReLU
    if(input < 0.0f){
        return 0.0f;
    }
    return input;
}

float Layer::activation_function_derivative(float input){
    //ReLU
    if(input < 0.0f){
        return 0.0f;
    }
    return 1.0f;
}

float Layer::cost_function(float output, float expected_output){
    return (output-expected_output) * (output-expected_output);
}

float Layer::cost_function_derivative(float output, float expected_output){
    return 2 * (output - expected_output);
}

