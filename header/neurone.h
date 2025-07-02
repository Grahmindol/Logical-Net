#ifndef NEURONE_H
#define NEURONE_H

#include <math.h>

#define FALSE_ID     0
#define NOR_ID       1
#define NAND_A_ID    2
#define NOT_A_ID     3
#define NAND_B_ID    4
#define NOT_B_ID     5
#define XOR_ID       6
#define NAND_ID      7
#define AND_ID       8
#define XNOR_ID      9
#define B_ID        10
#define B_IMPL_A_ID 11
#define A_ID        12
#define A_IMPL_B_ID 13
#define OR_ID       14
#define TRUE_ID     15

#define VAR_ID     16 // for gate_tree purpose


typedef struct {
    float gate_logits[16];   // poids avant softmax
    float gate_weights[16];  // poids softmaxés (proba)
    float output;
    int id;             // the id in the layer
    float* link_logits[2];
    float* link_weights[2];
    int inputs_size;
} Neurone;

Neurone create_neurone(int id, int size_in);
void free_neurone(Neurone n);

void backward(Neurone *n, float* in, float* grad_output, float learning_rate, float* grad_in);
float forward(Neurone *n, float* in);

void normalize_neurone_softmax(Neurone *n);
int get_dominante_gate_id(Neurone *n);

#endif