#include "layer.h"
#include <stdlib.h>

void backward_layer(Layer *l, float* grad_output, int size_grad_output, float* grad_in, float learning_rate){
    for (int i = 0; i < 2*l->size; i++) grad_in[i] = 0;
    
    for (int i = 0; i < size_grad_output; i++){
        backward(&l->neurones[i % l->size],grad_output[i],learning_rate);
        grad_in[2*(i % l->size)] += gradient_neurone(&l->neurones[i % l->size],0) * grad_output[i];
        grad_in[2*(i % l->size)+1] += gradient_neurone(&l->neurones[i % l->size],1) * grad_output[i];
    }
}

/**
 * @brief compute the signal throug the layer
 * 
 * @param l the layer
 * @param in float array of size : "size_in"
 * @param size_in int 
 * @param out float array of size : "l.size"
 */
void forward_layer(Layer *l, float* in, int size_in, float* out){
    for (int i = 0; i < l->size; i++)
    {
        int a = (i * 2)%size_in;
        int b = (i * 2 + 1 )%size_in;
        out[i] = forward(&l->neurones[i],in[a],in[b]);
    }
}