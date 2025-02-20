#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "include/mnist_file.h"
#include "include/d_neural_network.h"

// convert a pixel value from 0-255 to a double 0->1
#define PIXEL_SCALE(x) (((double) (x)) / 255.0f)

// Return a random double value between 0 and 1
#define RAND_DOUBLE() (((double) rand()) / ((double) RAND_MAX))

/**
 * Initialise weights and bias vectors between 0 and 1
 */
void d_neural_network_random_weights(d_neural_network_t * network)
{
    int i, j;

    // iterate through labels
    for (i = 0; i < MNIST_LABELS; i++) {
        network->b[i] = RAND_DOUBLE();

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            network->W[i][j] = RAND_DOUBLE();
        }
    }
}

/**
 * calculate softmax vector from the activations. Normalises the activations
 * to prevent large exponents
 */
void d_neural_network_softmax(double * activations, int length)
{
    int i;
    double sum, max;

    // iterate through each of the labels
    for (i = 1, max = activations[0]; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++) {
        activations[i] = exp(activations[i] - max);
        sum += activations[i];
    }

    for (i = 0; i < length; i++) {
        activations[i] /= sum;
    }
}

/**
 * Use the weights and bias vector to forward propagate through the neural
 * network and calculate the activations using double precision.
 */
void d_neural_network_hypothesis(mnist_image_t * image, d_neural_network_t * network, double activations[MNIST_LABELS])
{
    int i, j;

    for (i = 0; i < MNIST_LABELS; i++) {
        activations[i] = network->b[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            activations[i] += network->W[i][j] * PIXEL_SCALE(image->pixels[j]);
        }
    }

    d_neural_network_softmax(activations, MNIST_LABELS);
}

/**
 * Update the gradients for this step of gradient descent using the gradient
 * contributions from a single training example (image) in double precision.
 * 
 * This function returns the loss contribution from this training example.
 */
double d_neural_network_gradient_update(mnist_image_t * image, d_neural_network_t * network, d_neural_network_gradient_t * gradient, uint8_t label)
{
    double activations[MNIST_LABELS];
    double b_grad, W_grad;
    int i, j;

    // First forward propagate through the network to calculate activations
    d_neural_network_hypothesis(image, network, activations);

    for (i = 0; i < MNIST_LABELS; i++) {
        // This is the gradient for a softmax bias input
        b_grad = (i == label) ? activations[i] - 1.0 : activations[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            // The gradient for the neuron weight is the bias multiplied by the input weight
            W_grad = b_grad * PIXEL_SCALE(image->pixels[j]);

            // Update the weight gradient
            gradient->W_grad[i][j] += W_grad;
        }

        // Update the bias gradient
        gradient->b_grad[i] += b_grad;
    }

    // Cross-entropy loss
    return -log(activations[label]);  
}

/**
 * Run one step of gradient descent and update the neural network using double precision.
 */
double d_neural_network_training_step(mnist_dataset_t * dataset, d_neural_network_t * network, double learning_rate)
{
    d_neural_network_gradient_t gradient;
    double total_loss;
    int i, j;

    // Zero initialize gradient for weights and bias vector
    memset(&gradient, 0, sizeof(d_neural_network_gradient_t));

    // Calculate the gradient and the loss by looping through the training set
    for (i = 0, total_loss = 0.0; i < dataset->size; i++) {
        total_loss += d_neural_network_gradient_update(&dataset->images[i], network, &gradient, dataset->labels[i]);
    }

    // Apply gradient descent to the network
    for (i = 0; i < MNIST_LABELS; i++) {
        network->b[i] -= learning_rate * gradient.b_grad[i] / ((double) dataset->size);

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            network->W[i][j] -= learning_rate * gradient.W_grad[i][j] / ((double) dataset->size);
        }
    }

    return total_loss;
}
