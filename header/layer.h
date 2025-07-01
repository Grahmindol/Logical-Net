#ifndef LAYER_H
#define LAYER_H

#include "neurone.h"

typedef struct {
    Neurone *neurones;
    int size;
} Layer;

Layer *create_layer(int size);
void free_layer(Layer *layer);

void backward_layer(Layer *l, float* grad_output, int size_grad_output, float* grad_in, float learning_rate);
void forward_layer(Layer *l, float* in, int size_in, float* out);

#endif