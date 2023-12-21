#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "neural_network.h"

int main(){
    std::ifstream training;
    training.open ("../data/mnist_train.csv");
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> labels;
    std::string line;
    float value;
    char comma;
    int count = 0;
    if (training.is_open())
    {
        while ( std::getline(training,line) )
        {
            std::istringstream iss(line);
            iss >> value;
            std::vector<float> label(10,0);
            label[value] = 1;
            labels.push_back(label);
            
            std::vector<float> input;
            while(iss >> comma >> value){
                input.push_back(value/255.0f);
            }
            inputs.push_back(input);

            count ++;
        }
        training.close();
    }


    srand((unsigned) time(NULL));
    NeuralNetwork nn{784};
    nn.add_layer(150, 0);
    nn.add_layer(150, 0);
    nn.add_layer(10, 1);
    nn.train(inputs, labels, 20, 0.0001);
    return 0;
}
