#include "network.h"
#include <stdlib.h>

void forward_network(Network *net, float *inputs, int input_size, float *final_outputs){
    float *current_in = inputs;
    float *outputs = NULL;

    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        outputs = (float *)malloc(sizeof(float) * layer->size);
        forward_layer(layer, current_in, input_size, outputs);

        if (l != 0) free((void*)current_in);
        current_in = outputs;
        input_size = layer->size;
    }

    for (int i = 0; i < net->layers[net->num_layers-1].size; i++) {
        final_outputs[i] = current_in[i];
    }
    free(current_in);
}


void backward_network(Network *net, float *grad_final, int grad_size, float learning_rate, float *inputs) {
    float *grad_current = grad_final;
    float *grad_prev = NULL;

    for (int l = net->num_layers - 1; l >= 0; l--) {
        Layer *layer = &net->layers[l];

        grad_prev = (float *)malloc(sizeof(float) * layer->size * 2);
        backward_layer(layer, grad_current, grad_size, grad_prev, learning_rate);

        if (l != net->num_layers - 1) free(grad_current);
        grad_current = grad_prev;
        grad_size = layer->size * 2;
    }

    free(grad_current);
}