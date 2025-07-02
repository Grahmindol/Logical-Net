#include "network.h"
#include <stdlib.h>
#include <stdio.h>

// Créer un réseau
Network *create_network(int num_layers, int *layer_sizes) {
    Network *net = (Network*)malloc(sizeof(Network));
    net->num_layers = num_layers;
    net->layers = (Layer*)malloc(sizeof(Layer) * num_layers);
    for (int i = 0; i < num_layers; i++) net->layers[i] = create_layer(layer_sizes[i+1], layer_sizes[i]);
    return net;
}

// Supprimer un réseau
void free_network(Network *net) {
    for (int i = 0; i < net->num_layers; i++) free_layer(net->layers[i]);
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

    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        for (int n = 0; n < layer->size; n++) {
            printf("║ L%d-N%d    ", l, n);
            for (int i = 0; i < 16; i++) {
                float w = layer->neurones[n].gate_weights[i];
                int gray_level = 232 + (int)(w * 23); // 232 à 255 = 24 niveaux
                printf("│\x1b[38;5;%dm %6.3f \x1b[0m", gray_level, w); //---------------------------------------------------------------
            }
            printf("║\n");
            printf("║ link A   ", l, n);
            for (int i = 0; i < layer->neurones[n].inputs_size; i++) {
                float w = layer->neurones[n].link_weights[0][i];
                int gray_level = 232 + (int)(w * 23); // 232 à 255 = 24 niveaux
                printf("│\x1b[38;5;%dm %6.3f \x1b[0m", gray_level, w); //---------------------------------------------------------------
            }
            printf("║\n");
            printf("║ link B   ", l, n);
            for (int i = 0; i < layer->neurones[n].inputs_size; i++) {
                float w = layer->neurones[n].link_weights[1][i];
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

void forward_network(Network *net, float *inputs, float *final_outputs){
    float *current_in = inputs;
    float *outputs = NULL;

    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        outputs = (float *)malloc(sizeof(float) * layer->size);
        forward_layer(layer, current_in, outputs);

        if (current_in != inputs) free((void*)current_in);
        current_in = outputs;
    }

    for (int i = 0; i < net->layers[net->num_layers-1].size; i++) {
        final_outputs[i] = current_in[i];
    }
    free(current_in);
}


void backward_network(Network *net, float* inputs, float *grad_final, float learning_rate) {
    float *grad_current = grad_final;
    float *grad_prev = NULL;
    float* prev_out = NULL;

    for (int l = net->num_layers - 1; l >= 0; l--) {
        Layer *layer = &net->layers[l];

        if (l > 0) {
            prev_out = (float*)malloc(layer->input_size * sizeof(float));
            for (int i = 0; i < layer->input_size; i++) {
                prev_out[i] = net->layers[l-1].neurones[i].output;
            }
        } else {
            prev_out = inputs;
        }

        grad_prev = (float *)malloc(sizeof(float) * layer->input_size);
        backward_layer(layer, prev_out, grad_current, grad_prev, learning_rate);

        if (prev_out != inputs) free(prev_out);
        if (l != net->num_layers - 1) free(grad_current);
        grad_current = grad_prev;
    }

    free(grad_current);
}
