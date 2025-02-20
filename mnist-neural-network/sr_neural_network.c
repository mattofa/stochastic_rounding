#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "include/mnist_file.h"
#include "include/sr_neural_network.h"

// Convert a pixel value from 0-255 to one from 0 to 1
#define PIXEL_SCALE(x) (((double) (x)) / 255.0f)
// Returns a random value between 0 and 1
#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))


// Implement SR according to the Eqn 1.
float SR(double x) {

  // calculate RA(x) and RZ(x)

  // find the bordering floats, up and down
  float closest = (float)x;
  float down, up, rz, ra;
  if (closest > x) {
    down = nextafterf(closest, -INFINITY);
    up = closest;
  } else {
    down = closest;
    up = nextafterf(closest, INFINITY);
  }
  // pick the round away and round to zero
  if (x < 0) {
    rz = up;
    ra = down;
  } else {
    ra = up;
    rz = down;
  }

  // define P, not using srand() to keep determinism
  double P = (double)rand() / ((double)RAND_MAX + 1.0);
  // calculate p = ((x-RZ(x)) / (RA(x)-RZ(x)))
  double p = (x - rz) / (ra - rz);
  // Choose either RA or RZ
  if (P < p)
    return ra;
  return rz;
}

/**
 * Initialise the weights and bias vectors with values between 0 and 1
 */
void sr_neural_network_random_weights(sr_neural_network_t * network)
{
    int i, j;
    // initialise weights and biases in Binary32
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
void sr_neural_network_softmax(double * activations, int length)
{
    int i;
    double sum, max;

    for (i = 1, max = activations[0]; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++) {
        // temp = exp((double)activations[i] - (double)max);
        activations[i] = exp(activations[i] - max);
        sum += activations[i];
         
    }

    for (i = 0; i < length; i++) {
        activations[i] /= sum;
        // activations[i] = activations[i] / (double)sum;
    }
}

/**
 * Use the weights and bias vector to forward propogate through the neural
 * network and calculate the activations.
 */
void sr_neural_network_hypothesis(mnist_image_t * image, sr_neural_network_t * network, double activations[MNIST_LABELS])
{
    int i, j;

    for (i = 0; i < MNIST_LABELS; i++) {
        activations[i] = network->b[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            activations[i] += (double)network->W[i][j] * PIXEL_SCALE(image->pixels[j]);
        }
    }

    sr_neural_network_softmax(activations, MNIST_LABELS);
}

/**
 * Update the gradients for this step of gradient descent using the gradient
 * contributions from a single training example (image).
 * 
 * This function returns the loss ontribution from this training example.
 */
double sr_neural_network_gradient_update(mnist_image_t * image, sr_neural_network_t * network, sr_neural_network_gradient_t * gradient, uint8_t label)
{
    double activations[MNIST_LABELS];
    double b_grad, W_grad, temp;
    int i, j;

    // First forward propagate through the network to calculate activations
    sr_neural_network_hypothesis(image, network, activations);

    for (i = 0; i < MNIST_LABELS; i++) {
        // This is the gradient for a softmax bias input
        b_grad = (i == label) ? activations[i] - (double)1 : activations[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            // The gradient for the neuron weight is the bias multiplied by the input weight
            W_grad = b_grad * PIXEL_SCALE(image->pixels[j]);

            // Update the weight gradient
            // gradient->W_grad[i][j] += W_grad;
            gradient->W_grad[i][j] = SR((double)gradient->W_grad[i][j] + W_grad);
        }

        // Update the bias gradient
        // gradient->b_grad[i] += b_grad;
        temp = (double)gradient->b_grad[i] + b_grad;
        gradient->b_grad[i] = SR(temp);
        
    }

    // Cross entropy loss
    return (double)0.0f - log(activations[label]);
}

/**
 * Run one step of gradient descent and update the neural network.
 */
float sr_neural_network_training_step(mnist_dataset_t * dataset, sr_neural_network_t * network, float learning_rate)
{
    sr_neural_network_gradient_t gradient;
    double total_loss;
    int i, j;
    // used wherever double precision is needed temporarily before rounding
    double temp;

    // Zero initialise gradient for weights and bias vector
    memset(&gradient, 0, sizeof(sr_neural_network_gradient_t));

    // Calculate the gradient and the loss by looping through the training set
    for (i = 0, total_loss = 0; i < dataset->size; i++) {
        // total loss as a double, and round to float after each step
        total_loss += sr_neural_network_gradient_update(&dataset->images[i], network, &gradient, dataset->labels[i]);
    }

    // Apply gradient descent to the network
    for (i = 0; i < MNIST_LABELS; i++) {
        // calculate the bias update in double then round
        temp = (double)network->b[i] - ((double)learning_rate * (double)gradient.b_grad[i] / ((double)dataset->size));
        network->b[i] = SR(temp);

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            temp = (double)network->W[i][j] - ((double)learning_rate * (double)gradient.W_grad[i][j] / (double)dataset->size);
            network->W[i][j] = SR(temp);
        }
    }

    return SR(total_loss);
}
