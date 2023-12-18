#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include "neuron.h"

class Layer{
    public:
        Layer(unsigned int neuron_count, unsigned int input_size);

        unsigned int get_neuron_count();

        std::vector<float> propogate(std::vector<float>& inputs);
    
    private:
        unsigned int input_size_;
        unsigned int neuron_count_;
        
        std::vector<Neuron> neurons_;
};


#endif
