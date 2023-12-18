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
