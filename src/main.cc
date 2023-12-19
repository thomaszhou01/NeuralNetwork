#include <iostream>
#include "neural_network.h"

int main(){

    std::cout << "Hello World!" << std::endl;

    NeuralNetwork nn{100};
    nn.add_layer(200);
    nn.add_layer(150);
    nn.add_layer(100);
    nn.add_layer(10);
    return 0;
}
