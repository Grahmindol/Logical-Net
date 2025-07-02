#ifndef LAYER_H
#define LAYER_H

#include "neurone.h"

typedef struct {
    Neurone *neurones;
    int size;
    int input_size;
} Layer;

Layer create_layer(int size, int input_size);
void free_layer(Layer layer);

void backward_layer(Layer *l, float* inputs, float* grad_output, float* grad_in, float learning_rate);
void forward_layer(Layer *l, float* inputs, float* out);

#endif