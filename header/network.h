#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

typedef struct {
    Layer *layers;
    int num_layers;
} Network;

Network *create_network(int num_layers, int *layer_sizes);
void free_network(Network *net);
void print_network(Network *net);

void forward_network(Network *net, float *inputs, int input_size, float *final_outputs);
void backward_network(Network *net, float *grad_final, int grad_size, float learning_rate, float *inputs) ;
#endif