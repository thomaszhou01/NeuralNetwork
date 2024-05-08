#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "neural_network.h"

int main(){
    std::ifstream training;
    training.open ("../data/mnist_train.csv");
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> labels;
    std::string line;
    double value;
    char comma;
    int count = 0;
    if (training.is_open())
    {
        while ( std::getline(training,line) )
        {
            std::istringstream iss(line);
            iss >> value;
            std::vector<double> label(10,0);
            label[value] = 1;
            labels.push_back(label);
            
            std::vector<double> input;
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
    nn.add_layer(100, 0);
    nn.add_layer(10, 1);
    nn.train(inputs, labels, 6, 0.055, 1);

    std::ifstream test;
    test.open ("../data/mnist_test.csv");
    std::vector<std::vector<double>> inputsTest;
    std::vector<int> labelsTest;
    int labelTest;
    count = 0;
    if (test.is_open())
    {
        while ( std::getline(test,line) )
        {
            std::istringstream iss(line);
            iss >> labelTest;
            labelsTest.push_back(labelTest);
            
            std::vector<double> input;
            while(iss >> comma >> value){
                input.push_back(value/255.0f);
            }
            inputsTest.push_back(input);

            count ++;
        }
        test.close();
    }

    int correct = 0;
    int total = 0;
    for(int i = 0; i < labelsTest.size(); i++){
        int ans = nn.predict(inputsTest[i]);
        if (ans == labelsTest[i]){
            correct++;
        }
        total += 1;
    }

    std::cout << "Accuracy: " << ((double)correct)/((double)total) << " | Correct: " << correct << ", Total: " << total << std::endl;

    return 0;
}
