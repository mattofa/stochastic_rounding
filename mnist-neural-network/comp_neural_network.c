#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "include/mnist_file.h"
#include "include/comp_neural_network.h"

// Convert a pixel value from 0-255 to one from 0 to 1
#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)

// Returns a random value between 0 and 1
#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))


void fastTwoSum(float a, float b, float *s, float *t)
{
  float temp;

  // calculate the result
  *s = a + b;
  // calculate the error in the sum
  temp = *s - a;
  *t = b - temp;
}

/**
 * Initialise the weights and bias vectors with values between 0 and 1
 */
void comp_neural_network_random_weights(comp_neural_network_t * network)
{
    int i, j;

    for (i = 0; i < MNIST_LABELS; i++) {
        network->b[i] = RAND_FLOAT();

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            network->W[i][j] = RAND_FLOAT();
        }
    }
}

/**
 * Calculate the softmax vector from the activations. This uses a more
 * numerically stable algorithm that normalises the activations to prevent
 * large exponents.
 */
void comp_neural_network_softmax(float * activations, int length)
{
    int i;
    float sum, max;
    float t = 0;

    for (i = 1, max = activations[0]; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++) {
        activations[i] = exp(activations[i] - max);
        float addend = activations[i] + t;
        // sum += activations[i];
        // use fast two sum to increment sum (including) previous error
        // and store the new error
        fastTwoSum(sum, addend, &sum, &t);
    }

    for (i = 0; i < length; i++) {
        activations[i] /= sum;
    }
}

/**
 * Use the weights and bias vector to forward propogate through the neural
 * network and calculate the activations.
 */
void comp_neural_network_hypothesis(mnist_image_t * image, comp_neural_network_t * network, float activations[MNIST_LABELS])
{
    int i, j;
    float t = 0;    // t stores the error term

    for (i = 0; i < MNIST_LABELS; i++) {
        activations[i] = network->b[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            float addend = network->W[i][j] * PIXEL_SCALE(image->pixels[j]);
            // activations[i] += network->W[i][j] * PIXEL_SCALE(image->pixels[j]);
            // increment activations using compensated sum and store error
            fastTwoSum(activations[i], addend, &activations[i], &t);
        }
    }

    comp_neural_network_softmax(activations, MNIST_LABELS);
}

/**
 * Update the gradients for this step of gradient descent using the gradient
 * contributions from a single training example (image).
 * 
 * This function returns the loss ontribution from this training example.
 */
float comp_neural_network_gradient_update(mnist_image_t * image, comp_neural_network_t * network, comp_neural_network_gradient_t * gradient, uint8_t label)
{
    float activations[MNIST_LABELS];
    float b_grad, W_grad;
    int i, j;
    float s = 0, t = 0; // error terms

    // First forward propagate through the network to calculate activations
    comp_neural_network_hypothesis(image, network, activations);

    for (i = 0; i < MNIST_LABELS; i++) {
        // This is the gradient for a softmax bias input
        b_grad = (i == label) ? activations[i] - 1 : activations[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            // The gradient for the neuron weight is the bias multiplied by the input weight
            W_grad = b_grad * PIXEL_SCALE(image->pixels[j]);
            // calculate the addend with previous error s
            float addend = W_grad + s;
            // Update the weight gradient using fastTwoSum and including error term s
            fastTwoSum(gradient->W_grad[i][j], addend, &gradient->W_grad[i][j], &s);
            // gradient->W_grad[i][j] += W_grad;
        }

        // calc addend with previous error t
        float addend = b_grad + t;
        // Update the bias gradient using fastTwoSum including error term t
        fastTwoSum(gradient->b_grad[i], addend, &gradient->b_grad[i], &t);

        // gradient->b_grad[i] += b_grad;
    }

    // Cross entropy loss
    return 0.0f - log(activations[label]);
}

/**
 * Run one step of gradient descent and update the neural network.
 */
float comp_neural_network_training_step(mnist_dataset_t * dataset, comp_neural_network_t * network, float learning_rate)
{
    comp_neural_network_gradient_t gradient;
    float total_loss;
    int i, j;
    float s = 0, t = 0, v = 0; // error terms

    // Zero initialise gradient for weights and bias vector
    memset(&gradient, 0, sizeof(comp_neural_network_gradient_t));

    // Calculate the gradient and the loss by looping through the training set
    for (i = 0, total_loss = 0; i < dataset->size; i++) {
        // calculate the sum of addend and error term t
        float addend = comp_neural_network_gradient_update(&dataset->images[i], network, &gradient, dataset->labels[i]) + t;
        // sum the total loss and the addent, keeping result and error
        fastTwoSum(total_loss, addend, &total_loss, &t);
    }

    // Apply gradient descent to the network
    for (i = 0; i < MNIST_LABELS; i++) {
        float addend = -(learning_rate * gradient.b_grad[i] / ((float) dataset->size) + t);
        // network->b[i] -= learning_rate * gradient.b_grad[i] / ((float) dataset->size);
        fastTwoSum(network->b[i], addend, &network->b[i], &s);

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            float addend = -(learning_rate * gradient.W_grad[i][j] / ((float) dataset->size) + t);
            // network->W[i][j] -= learning_rate * gradient.W_grad[i][j] / ((float) dataset->size);
            fastTwoSum(network->W[i][j], addend, &network->W[i][j], &v);
        }
    }

    return total_loss;
}
