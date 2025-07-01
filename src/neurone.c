#include "neurone.h"
#include <stdio.h>
#include <stdlib.h>

typedef float (*LogicFunc)(float, float);
typedef float (*LogicDeriv)(float, float);

float logic_false(float a, float b) { return 0.0f; }
float logic_nor(float a, float b) { return (1 - a) * (1 - b); }
float logic_nand_a(float a, float b) { return (1 - a) * b; }
float logic_not_a(float a, float b) { return 1 - a; }
float logic_nand_b(float a, float b) { return a * (1 - b); }
float logic_not_b(float a, float b) { return 1 - b; }
float logic_xor(float a, float b) { return a + b - 2 * a * b; }
float logic_nand(float a, float b) { return 1 - a * b; }
float logic_and(float a, float b) { return a * b; }
float logic_xnor(float a, float b) { return a * b + (1 - a) * (1 - b); }
float logic_b(float a, float b) { return b; }
float logic_a_or_not_b(float a, float b) { return a + (1 - b) - a * (1 - b); }
float logic_a(float a, float b) { return a; }
float logic_not_a_or_b(float a, float b) { return (1 - a) + b - (1 - a) * b; }
float logic_or(float a, float b) { return a + b - a * b; }
float logic_true(float a, float b) { return 1.0f; }

LogicFunc logic_table[16] = {
    logic_false, logic_nor, logic_nand_a, logic_not_a,
    logic_nand_b, logic_not_b, logic_xor, logic_nand,
    logic_and, logic_xnor, logic_b, logic_a_or_not_b,
    logic_a, logic_not_a_or_b, logic_or, logic_true
};

// ∂f/∂a
float dfa_false(float a, float b) { return 0.0f; }
float dfa_nor(float a, float b) { return -(1 - b); }
float dfa_nand_a(float a, float b) { return -b; }
float dfa_not_a(float a, float b) { return -1.0f; }
float dfa_nand_b(float a, float b) { return 1 - b; }
float dfa_not_b(float a, float b) { return 0.0f; }
float dfa_xor(float a, float b) { return 1 - 2 * b; }
float dfa_nand(float a, float b) { return -b; }
float dfa_and(float a, float b) { return b; }
float dfa_xnor(float a, float b) { return b - (1 - b); }
float dfa_b(float a, float b) { return 0.0f; }
float dfa_a_or_not_b(float a, float b) { return 1 - (1 - b); }
float dfa_a(float a, float b) { return 1.0f; }
float dfa_not_a_or_b(float a, float b) { return -1 + b; }
float dfa_or(float a, float b) { return 1 - b; }
float dfa_true(float a, float b) { return 0.0f; }

// ∂f/∂b
float dfb_false(float a, float b) { return 0.0f; }
float dfb_nor(float a, float b) { return -(1 - a); }
float dfb_nand_a(float a, float b) { return 1 - a; }
float dfb_not_a(float a, float b) { return 0.0f; }
float dfb_nand_b(float a, float b) { return -a; }
float dfb_not_b(float a, float b) { return -1.0f; }
float dfb_xor(float a, float b) { return 1 - 2 * a; }
float dfb_nand(float a, float b) { return -a; }
float dfb_and(float a, float b) { return a; }
float dfb_xnor(float a, float b) { return a - (1 - a); }
float dfb_b(float a, float b) { return 1.0f; }
float dfb_a_or_not_b(float a, float b) { return -a + 1; }
float dfb_a(float a, float b) { return 0.0f; }
float dfb_not_a_or_b(float a, float b) { return 1 - (1 - a); }
float dfb_or(float a, float b) { return 1 - a; }
float dfb_true(float a, float b) { return 0.0f; }

// Tables
LogicDeriv logic_deriv[2][16] = {{
    dfa_false, dfa_nor, dfa_nand_a, dfa_not_a,
    dfa_nand_b, dfa_not_b, dfa_xor, dfa_nand,
    dfa_and, dfa_xnor, dfa_b, dfa_a_or_not_b,
    dfa_a, dfa_not_a_or_b, dfa_or, dfa_true
},{
    dfb_false, dfb_nor, dfb_nand_a, dfb_not_a,
    dfb_nand_b, dfb_not_b, dfb_xor, dfb_nand,
    dfb_and, dfb_xnor, dfb_b, dfb_a_or_not_b,
    dfb_a, dfb_not_a_or_b, dfb_or, dfb_true
}};

// Créer un neurone
Neurone *create_neurone() {
    Neurone *n = (Neurone*)malloc(sizeof(Neurone));
    for (int i = 0; i < 16; i++) {
        n->logits[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    normalize_neurone_softmax(n);
    return n;
}

// Supprimer un neurone
void free_neurone(Neurone *n) {
    free(n);
}

float gradient_neurone(Neurone *n, int input_id) {
    float sum = 0;
    for (int i = 0; i < 16; i++) {
        sum += n->weights[i] * logic_deriv[input_id&1][i](n->a, n->b);
    }
    return sum;
}

void normalize_neurone_softmax(Neurone *n) {
    float temperature = 2.0f;

    // Chercher max après division
    float max = n->logits[0] / temperature;
    for (int i = 1; i < 16; i++)
        if (n->logits[i] / temperature > max) max = n->logits[i] / temperature;

    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        n->weights[i] = expf((n->logits[i] / temperature) - max);
        sum += n->weights[i];
    }
    for (int i = 0; i < 16; i++) {
        n->weights[i] /= sum;
    }
}


float forward(Neurone *n, float a, float b) {
    n->a = a;
    n->b = b;
    float out = 0.0f;
    for (int i = 0; i < 16; i++)
        out += n->weights[i] * logic_table[i](a, b);
    n->output = out;
    return out;
}

void backward(Neurone *n, float grad_output, float learning_rate) {
    float grad_logits[16] = {0};
    // Gradient des logits via chain rule softmax
    // dL/dz_j = sum_i dL/dw_i * dw_i/dz_j
    // dw_i/dz_j = softmax_i * (delta_ij - softmax_j)
    // Ici : dL/dw_i = grad_output * logic_table[i](a,b)
    float grad_w[16];
    for (int i = 0; i < 16; i++) {
        grad_w[i] = grad_output * logic_table[i](n->a, n->b);
    }
    // Calcul dL/dz_j
    for (int j = 0; j < 16; j++) {
        float sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            float delta = (i == j) ? 1.0f : 0.0f;
            sum += grad_w[i] * n->weights[i] * (delta - n->weights[j]);
        }
        grad_logits[j] = sum;
    }
    // Mise à jour logits
    for (int i = 0; i < 16; i++) {
        n->logits[i] -= learning_rate * grad_logits[i];
    }
    normalize_neurone_softmax(n);
}