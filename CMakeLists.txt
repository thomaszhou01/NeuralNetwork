cmake_minimum_required(VERSION 3.12)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 14)

# Add your source files here
set(SOURCES
    src/main.cc
    src/neural_network.cc
    src/neuron.cc
    src/layer.cc
)

# Add your header files here
set(HEADERS
    include/neural_network.h
    include/neuron.h
    include/layer.h
)

# Add any additional dependencies here
set(DEPENDENCIES
    # Add your dependencies here
)

# Set the include directories
include_directories(include)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link any additional dependencies
target_link_libraries(${PROJECT_NAME} ${DEPENDENCIES})
