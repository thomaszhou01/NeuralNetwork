#include "layer.h"

Layer::Layer(unsigned int neuron_count, unsigned int input_size, unsigned int activation_function): input_size_(input_size), neuron_count_(neuron_count), activation_function_(activation_function){
    this->neurons_.reserve(neuron_count);

    for(int i = 0; i < neuron_count; i++){
        this->neurons_.push_back(Neuron(input_size, activation_function));
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
        this->neurons_[i].update_nodeVal(output);
        // std::cout << "Weighted: " << this->neurons_[i].get_weighted_sum() << " " << this->neurons_[i].get_output() << " " << output << " " << outputs[i] << std::endl;
    }

    // outputs == nodeValues

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
        this->neurons_[i].update_nodeVal(output);
    }

    return outputs;
}

std::vector<float> Layer::gradient_descent(std::vector<float>& inputs, float learning_rate){
    std::vector<float> outputs;

    for (int i = 0; i < this->neuron_count_; i++){
        for(int j = 0; j < inputs.size(); j++){
            this->neurons_[i].update_weights(j, inputs[j], learning_rate);
        }

        this->neurons_[i].update_bias(learning_rate);
        this->neurons_[i].reset_nodeVal();
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

float Layer::activation_function(float input){
    //ReLU
    if(activation_function_ == 0){
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

float Layer::activation_function_derivative(float input){
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

float Layer::cost_function(float output, float expected_output){
    return (output-expected_output) * (output-expected_output);
}

float Layer::cost_function_derivative(float output, float expected_output){
    return 2 * (output - expected_output);
}

