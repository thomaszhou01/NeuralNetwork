#include "layer.h"

Layer::Layer(unsigned int neuron_count, unsigned int input_size, unsigned int activation_function): input_size_(input_size), neuron_count_(neuron_count), activation_function_(activation_function){
    this->neurons_.reserve(neuron_count);
    nodeValues_.reserve(neuron_count);

    for(int i = 0; i < neuron_count; i++){
        this->neurons_.push_back(Neuron(input_size, activation_function));
    }
}

unsigned int Layer::get_neuron_count(){
    return this->neuron_count_;
}

std::vector<double> Layer::propogate(const std::vector<double>& inputs){
    std::vector<double> outputs;
    outputs.reserve(this->neuron_count_);

    for(int i = 0; i < this->neuron_count_; i++){
        outputs.push_back(this->neurons_[i].propogate(inputs));
    }

    return outputs;
}

//takes expected values
std::vector<double> Layer::back_propogate_output(std::vector<double>& inputs){
    std::vector<double> outputs;
    outputs.reserve(this->neuron_count_);

    for (int i = 0; i < this->neuron_count_; i++){
        double output = this->cost_function_derivative(this->neurons_[i].get_output(), inputs[i]) * this->activation_function_derivative(this->neurons_[i].get_weighted_sum());
        outputs.push_back(output);
        this->neurons_[i].update_nodeVal(output);
        this->nodeValues_[i] = output;
        // std::cout << "Weighted: " << this->neurons_[i].get_weighted_sum() << " " << this->neurons_[i].get_output() << " " << output << " " << outputs[i] << std::endl;
    }

    // outputs == nodeValues

    return outputs;
}

std::vector<double> Layer::back_propogate_hidden(Layer &prev_layer, std::vector<double>& inputs){
    std::vector<double> outputs;
    outputs.reserve(this->neuron_count_);

    for (int i = 0; i < this->neuron_count_; i++){
        double output = 0.0f;
        for (int j = 0; j < inputs.size(); j++){
            double weighted = prev_layer.neurons_[j].get_weight_value(j);
            output += inputs[j] * weighted;
        }
        output *= this->activation_function_derivative(this->neurons_[i].get_weighted_sum());
        outputs.push_back(output);
        this->neurons_[i].update_nodeVal(output);
        this->nodeValues_[i] = output;
    }

    return outputs;
}

std::vector<double> Layer::update_gradients(std::vector<double>& inputs){
    std::vector<double> outputs;

    for (int i = 0; i < this->neuron_count_; i++){
        for(int j = 0; j < inputs.size(); j++){
            this->neurons_[i].update_gradient_weights(j, inputs[j]);
        }
        
        this->neurons_[i].update_gradient_bias();
        outputs.push_back(this->neurons_[i].get_output());
    }

    return outputs;
}

std::vector<double> Layer::gradient_descent(double learning_rate){
    std::vector<double> outputs;

    for (int i = 0; i < this->neuron_count_; i++){
        for(int j = 0; j < this->neurons_[i].get_input_size(); j++){
            this->neurons_[i].update_weights(j, learning_rate);
        }

        this->neurons_[i].update_bias(learning_rate);
        this->neurons_[i].reset_nodeVal();
        outputs.push_back(this->neurons_[i].get_output());
    }

    return outputs;
}

double Layer::get_cost(std::vector<double>& labels){
    double cost = 0.0f;

    for(int i = 0; i < this->neuron_count_; i++){
        cost += (neurons_[i].get_output() - labels[i]) * (neurons_[i].get_output() - labels[i]);
    }

    return cost;
}

double Layer::activation_function(double input){
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

double Layer::activation_function_derivative(double input){
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

double Layer::cost_function(double output, double expected_output){
    return (output-expected_output) * (output-expected_output);
}

double Layer::cost_function_derivative(double output, double expected_output){
    return 2 * (output - expected_output);
}

