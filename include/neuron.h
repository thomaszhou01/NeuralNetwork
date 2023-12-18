#ifndef NEURON_H_
#define NEURON_H_

#include <iostream>
#include <vector>

class Neuron {
    public:
        Neuron(unsigned int input_size);

        float propogate(std::vector<float>& inputs);

        float back_propogate(std::vector<float>& inputs);
    
    private:
        float bias_;
        float output_;

        unsigned int input_size_;
        std::vector<float> weights_;

        float activation_function(float input);
};

#endif
