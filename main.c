
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"

float loss(float y_pred, float y_true) {
    return 0.5f * (y_pred - y_true) * (y_pred - y_true);
}

float dloss(float y_pred, float y_true) {
    return y_pred - y_true;
}

#include <stdio.h>

void print_network(Network *net, float *inputs, int input_size) {
    printf("\n=== Réseau final ===\n");

    float *layer_in = inputs;
    int size_in = input_size;

    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        float *layer_out = (float*)malloc(sizeof(float) * layer->size);

        printf("Couche %d : %d neurones\n", l+1, layer->size);

        forward_layer(layer, layer_in, size_in, layer_out);

        for (int i = 0; i < layer->size; i++) {
            Neurone *n = &layer->neurones[i];
            printf(" Neurone %d:\n", i);
            printf("  Logits (softmax probs) : ");
            for (int j = 0; j < 16; j++) {
                printf("%.3f ", n->weights[j]);
            }
            printf("\n");
            printf("  Output: %.4f\n", layer_out[i]);
        }

        if (l != 0) free(layer_in);
        layer_in = layer_out;
        size_in = layer->size;

        printf("\n");
    }

    free(layer_in);
}


int main() {
    srand(time(NULL));

    // Paramètres
    int input_size = 4;
    float inputs[] = {0.0f, 1.0f, 0.0f, 1.0f}; // exemple binaire
    float target = 0.0f;
    float lr = 0.1f;

    // Créer réseau
    Network net;
    net.num_layers = 2;
    net.layers = (Layer*)malloc(sizeof(Layer) * net.num_layers);

    // Couche 1 : 2 neurones
    net.layers[0].size = 2;
    net.layers[0].neurones = (Neurone*)malloc(sizeof(Neurone) * 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 16; j++) {
            net.layers[0].neurones[i].logits[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        normalize_neurone_softmax(&net.layers[0].neurones[i]);
    }

    // Couche 2 : 1 neurone
    net.layers[1].size = 1;
    net.layers[1].neurones = (Neurone*)malloc(sizeof(Neurone) * 1);
    for (int j = 0; j < 16; j++) {
        net.layers[1].neurones[0].logits[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    normalize_neurone_softmax(&net.layers[1].neurones[0]);

    float output[1];

    for (int epoch = 0; epoch < 100000; epoch++) {
        // Forward
        forward_network(&net, inputs, input_size, output);

        // Loss
        float l = loss(output[0], target);

        // Backward
        float grad_final[1] = { dloss(output[0], target) };
        backward_network(&net, grad_final, 1, lr, inputs);

        if (epoch % 50 == 0) {
            printf("Epoch %d, output: %.4f, loss: %.6f\n", epoch, output[0], l);
        }
    }

    // Vérif finale
    forward_network(&net, inputs, input_size, output);
    printf("Final output: %.4f\n", output[0]);
    print_network(&net,inputs,input_size);
    // Free
    free(net.layers[0].neurones);
    free(net.layers[1].neurones);
    free(net.layers);

    return 0;
}