#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "include/mnist_file.h"
#include "include/neural_network.h"
#include "include/d_neural_network.h"
#include "include/sr_neural_network.h"
#include "include/comp_neural_network.h"

#define STEPS 70
#define BATCH_SIZE 100

/**
 * Downloaded from: http://yann.lecun.com/exdb/mnist/
 */
const char * train_images_file = "data/train-images-idx3-ubyte";
const char * train_labels_file = "data/train-labels-idx1-ubyte";
const char * test_images_file = "data/t10k-images-idx3-ubyte";
const char * test_labels_file = "data/t10k-labels-idx1-ubyte";


/**
 * Calculate the accuracy of the predictions of a neural network on a dataset.
 */
float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / ((float) dataset->size);
}

/**
 * Calculate the accuracy of the predictions of Double neural network on a dataset.
 */
double calculate_d_accuracy(mnist_dataset_t * dataset, d_neural_network_t * network)
{
    double activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        d_neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((double) correct) / ((double) dataset->size);
}


/**
 * Calculate the accuracy of the predictions of SR neural network on a dataset.
 */
double calculate_sr_accuracy(mnist_dataset_t * dataset, sr_neural_network_t * network)
{
    double activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        sr_neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / ((float) dataset->size);
}

/**
 * Calculate the accuracy of the predictions of a neural network on a dataset.
 */
float calculate_comp_accuracy(mnist_dataset_t * dataset, comp_neural_network_t * network)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        comp_neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / ((float) dataset->size);
}

void writeAccuracy(FILE *file, double *accuracies, char *header)
{
    // print the header
    fprintf(file, "%s,", header);
    // print each of the accuracies to the file
    for (int i = 0; i<STEPS-1; i++) {
        fprintf(file, "%.20f, ", accuracies[i]);
    }
    fprintf(file, "%.20f\n", accuracies[STEPS-1]);
}

int main(int argc, char *argv[])
{
    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;
    d_neural_network_t dnetwork;
    sr_neural_network_t srnetwork;
    comp_neural_network_t compnetwork;
    float loss, accuracy;
    double dloss, daccuracy;
    float srloss, sraccuracy;
    float comploss, compaccuracy;
    int i, batches;
    double daccuracies[STEPS];
    double accuracies[STEPS];
    double sraccuracies[STEPS];
    double compaccuracies[STEPS];

    // Read the datasets from the files
    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

    // Initialise weights and biases with random values
    neural_network_random_weights(&network);
    d_neural_network_random_weights(&dnetwork);
    sr_neural_network_random_weights(&srnetwork);
    comp_neural_network_random_weights(&compnetwork);

    // Calculate how many batches (so we know when to wrap around)
    batches = train_dataset->size / BATCH_SIZE;

    for (i = 0; i < STEPS; i++) {
        // Initialise a new batch
        mnist_batch(train_dataset, &batch, 100, i % batches);

        // Run one step of gradient descent and calculate the loss
        loss = neural_network_training_step(&batch, &network, 0.5);
        dloss = d_neural_network_training_step(&batch, &dnetwork, 0.5);
        srloss = sr_neural_network_training_step(&batch, &srnetwork, 0.5);
        comploss = comp_neural_network_training_step(&batch, &compnetwork, 0.5);

        // Calculate the accuracy using the whole test dataset
        accuracy = calculate_accuracy(test_dataset, &network);
        daccuracy = calculate_d_accuracy(test_dataset, &dnetwork);
        sraccuracy = calculate_sr_accuracy(test_dataset, &srnetwork);
        compaccuracy = calculate_comp_accuracy(test_dataset, &compnetwork);
        printf("Step %04d\tAverage Loss: %.10f\tAccuracy: %.10f\n", i, loss / batch.size, accuracy);
        printf("Double loss: %.10f\tAccuracy: %.10f\n", dloss / batch.size, daccuracy);
        printf("SR loss:     %.10f\tAccuracy: %.10f\n", srloss / batch.size, sraccuracy);
        printf("Comp loss:   %.10f\tAccuracy: %.10f\n", comploss / batch.size, compaccuracy);
        
        // store the accuracy values
        accuracies[i] = accuracy;
        daccuracies[i] = daccuracy;
        sraccuracies[i] = sraccuracy;
        compaccuracies[i] = compaccuracy;
    }

    // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    // Write the accuracies to a file
    FILE *file = fopen("results/accuracy_errors.csv", "w");
    if (file == NULL) {
        printf("Error opening file");
        return 0;
    }
    writeAccuracy(file, accuracies, "Binary32");
    writeAccuracy(file, daccuracies, "Binary64");
    writeAccuracy(file, sraccuracies, "Binary32 with SR");
    writeAccuracy(file, compaccuracies, "Binary32 with Compensated Sum");

    fclose(file);

    return 0;
}
