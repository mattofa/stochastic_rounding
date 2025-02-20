#ifndef sr_neural_network_H_
#define sr_neural_network_H_

#include "mnist_file.h"

typedef struct sr_neural_network_t_ {
    float b[MNIST_LABELS];
    float W[MNIST_LABELS][MNIST_IMAGE_SIZE];
} sr_neural_network_t;

typedef struct sr_neural_network_gradient_t_ {
    float b_grad[MNIST_LABELS];
    float W_grad[MNIST_LABELS][MNIST_IMAGE_SIZE];
} sr_neural_network_gradient_t;

void sr_neural_network_random_weights(sr_neural_network_t * network);
void sr_neural_network_hypothesis(mnist_image_t * image, sr_neural_network_t * network, double activations[MNIST_LABELS]);
double sr_neural_network_gradient_update(mnist_image_t * image, sr_neural_network_t * network, sr_neural_network_gradient_t * gradient, uint8_t label);
float sr_neural_network_training_step(mnist_dataset_t * dataset, sr_neural_network_t * network, float learning_rate);
float SR(double x);

#endif