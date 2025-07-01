#include "network.h"
#include <stdlib.h>
#include <stdio.h>

// Créer un réseau
Network *create_network(int num_layers, int *layer_sizes) {
    Network *net = (Network*)malloc(sizeof(Network));
    net->num_layers = num_layers;
    net->layers = (Layer*)malloc(sizeof(Layer) * num_layers);

    for (int i = 0; i < num_layers; i++) {
        net->layers[i].size = layer_sizes[i];
        net->layers[i].neurones = (Neurone*)malloc(sizeof(Neurone) * layer_sizes[i]);
        for (int j = 0; j < layer_sizes[i]; j++) {
            for (int k = 0; k < 16; k++) {
                net->layers[i].neurones[j].logits[k] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            }
            normalize_neurone_softmax(&net->layers[i].neurones[j]);
        }
    }
    return net;
}

// Supprimer un réseau
void free_network(Network *net) {
    for (int i = 0; i < net->num_layers; i++) {
        free(net->layers[i].neurones);
    }
    free(net->layers);
    free(net);
}

const char *logic_names[16] = {
    "FALSE", "NOR", "NAND_A", "NOT_A",
    "NAND_B", "NOT_B", "XOR", "NAND",
    "AND", "XNOR", "B", "B => A",
    "A", "A => B", "OR", "TRUE"
};

void print_network(Network *net) {
    printf("╔══════════╤");
    for (int i = 0; i < 15; i++) printf("════════╤");
    printf("════════╗\n");
    
    // En-tête
    printf("║ %8s │", "Neurone");
    for (int i = 0; i < 15; i++) printf(" %6s │", logic_names[i]);
    printf(" %6s ║\n", logic_names[15]);
    printf("╠══════════╪");
    for (int i = 0; i < 15; i++) printf("════════╪");
    printf("════════╣\n");

    int neuron_idx = 0;
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        for (int n = 0; n < layer->size; n++) {
            printf("║ L%d-N%d    ", l, n);
            for (int i = 0; i < 16; i++) {
                float w = layer->neurones[n].weights[i];
                int gray_level = 232 + (int)(w * 23); // 232 à 255 = 24 niveaux
                printf("│\x1b[38;5;%dm %6.3f \x1b[0m", gray_level, w); //---------------------------------------------------------------
            }
            printf("║\n");
        }

        // Séparateur horizontal après chaque couche
        if (l != net->num_layers - 1) {
            printf("╟──────────┼");
            for (int i = 0; i < 15; i++)  printf("────────┼");
            printf("────────╢\n");
        }
    }
    printf("╚══════════╧");
    for (int i = 0; i < 15; i++) printf("════════╧");
    printf("════════╝\n");
}

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