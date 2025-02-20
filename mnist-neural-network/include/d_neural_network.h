#ifndef d_neural_network_H_
#define d_neural_network_H_

#include "mnist_file.h"

typedef struct d_neural_network_t_ {
    double b[MNIST_LABELS];
    double W[MNIST_LABELS][MNIST_IMAGE_SIZE];
} d_neural_network_t;

typedef struct d_neural_network_gradient_t_ {
    double b_grad[MNIST_LABELS];
    double W_grad[MNIST_LABELS][MNIST_IMAGE_SIZE];
} d_neural_network_gradient_t;

void d_neural_network_random_weights(d_neural_network_t * network);
void d_neural_network_hypothesis(mnist_image_t * image, d_neural_network_t * network, double dactivations[MNIST_LABELS]);
double d_neural_network_gradient_update(mnist_image_t * image, d_neural_network_t * network, d_neural_network_gradient_t * gradient, uint8_t label);
double d_neural_network_training_step(mnist_dataset_t * dataset, d_neural_network_t * network, double learning_rate);


#endif