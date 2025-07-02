#include "layer.h"
#include <stdlib.h>

// Créer une couche
Layer create_layer(int size, int input_size) {
    Layer layer;
    layer.size = size;
    layer.input_size = input_size;
    layer.neurones = (Neurone*)malloc(sizeof(Neurone) * size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < 16; j++) {
            layer.neurones[i].logits[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        normalize_neurone_softmax(&layer.neurones[i]);
    }
    return layer;
}

// Supprimer une couche
void free_layer(Layer *layer) {
    free(layer->neurones);
    free(layer);
}

void get_input_ids(int id, int size_in, int* a, int* b){
    *a = (id * 2 + (int)(id/size_in))%size_in;
    *b = (id * 2 + 1 + (int)(id/size_in))%size_in;
}


void backward_layer(Layer *l, float* grad_output, float* grad_in, float learning_rate){
    for (int i = 0; i < l->input_size; i++) grad_in[i] = 0;
    
    for (int i = 0; i < l->size; i++){
        backward(&l->neurones[i],grad_output[i],learning_rate);
        int a , b;
        get_input_ids(i,l->input_size,&a,&b);
        grad_in[a] += gradient_neurone(&l->neurones[i],0) * grad_output[i];
        grad_in[b] += gradient_neurone(&l->neurones[i],1) * grad_output[i];
    }

    // Normalisation du gradient d'entrée
    float norm = 0.0f;
    for (int i = 0; i < 2*l->size; i++) {
        norm += grad_in[i] * grad_in[i];
    }
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < 2*l->size; i++) {
            grad_in[i] /= norm;
        }
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
void forward_layer(Layer *l, float* in, float* out){
    for (int i = 0; i < l->size; i++)
    {
        int a , b;
        get_input_ids(i,l->input_size,&a,&b);
        out[i] = forward(&l->neurones[i],in[a],in[b]);
    }
}