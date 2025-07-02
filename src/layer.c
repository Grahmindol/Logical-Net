#include "layer.h"
#include <stdlib.h>

// Créer une couche
Layer create_layer(int size, int input_size) {
    Layer layer;
    layer.size = size;
    layer.input_size = input_size;
    layer.neurones = (Neurone*)malloc(sizeof(Neurone) * size);
    for (int i = 0; i < size; i++) layer.neurones[i] = create_neurone(i, input_size);
    return layer;
}

// Supprimer une couche
void free_layer(Layer *layer) {
    free(layer->neurones);
}

void backward_layer(Layer *l, float* inputs, float* grad_output, float* grad_in, float learning_rate){
    for (int i = 0; i < l->input_size; i++) grad_in[i] = 0;
    
    for (int i = 0; i < l->size; i++) backward(&l->neurones[i], inputs ,grad_output,learning_rate, grad_in);

    // Normalisation du gradient d'entrée
    float norm = 0.0f;
    for (int i = 0; i < l->input_size; i++) norm += grad_in[i] * grad_in[i];
    norm = sqrtf(norm);
    if (norm > 0.0f) for (int i = 0; i < l->input_size; i++) {
        grad_in[i] /= norm;
    }
}

void forward_layer(Layer *l, float* in, float* out){
    for (int i = 0; i < l->size; i++) out[i] = forward(&l->neurones[i],in);
}