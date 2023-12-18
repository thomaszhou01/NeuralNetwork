#include <iostream>
#include "neural_network.h"

int main(){

    std::cout << "Hello World!" << std::endl;

    NeuralNetwork nn{100};
    nn.add_layer(200);
    return 0;
}
