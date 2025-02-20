
#ifndef COMP_NEURAL_NETWORK_H_
#define COMP_NEURAL_NETWORK_H_

#include "mnist_file.h"

typedef struct comp_neural_network_t_ {
    float b[MNIST_LABELS];
    float W[MNIST_LABELS][MNIST_IMAGE_SIZE];
} comp_neural_network_t;

typedef struct comp_neural_network_gradient_t_ {
    float b_grad[MNIST_LABELS];
    float W_grad[MNIST_LABELS][MNIST_IMAGE_SIZE];
} comp_neural_network_gradient_t;

void comp_neural_network_random_weights(comp_neural_network_t * network);
void comp_neural_network_hypothesis(mnist_image_t * image, comp_neural_network_t * network, float activations[MNIST_LABELS]);
float comp_neural_network_gradient_update(mnist_image_t * image, comp_neural_network_t * network, comp_neural_network_gradient_t * gradient, uint8_t label);
float comp_neural_network_training_step(mnist_dataset_t * dataset, comp_neural_network_t * network, float learning_rate);
void fastTwoSum(float a, float b, float *s, float *t);


#endif
