#include "neurone.h"
#include <stdio.h>
#include <stdlib.h>

typedef float (*LogicFunc)(float, float);
typedef float (*LogicDeriv)(float, float);

float logic_false(float a, float b) { (void)a; (void)b; return -1.0f; }
float logic_true(float a, float b) { (void)a; (void)b; return 1.0f; } 
float logic_not_a(float a, float b) { (void)b; return -a; }
float logic_not_b(float a, float b) { (void)a; return -b; }
float logic_and(float a, float b) { return (a * b + a + b - 1.0f) / 2.0f; }
float logic_nand(float a, float b) { return -logic_and(a, b); }
float logic_or(float a, float b) { return (a + b - a * b  + 1.0f) / 2.0f; }
float logic_nor(float a, float b) { return -logic_or(a, b); }
float logic_xor(float a, float b) { return -a * b; }
float logic_xnor(float a, float b) { return a * b; }
float logic_a(float a, float b) { (void)b; return a; }
float logic_b(float a, float b) { (void)a; return b; }
float logic_a_or_not_b(float a, float b) { return (a - b + a * b + 1.0f) / 2.0f; }
float logic_not_a_or_b(float a, float b) { return (b - a + a * b + 1.0f) / 2.0f; }
float logic_nand_a(float a, float b) { return logic_and(-a, b); }
float logic_nand_b(float a, float b) { return logic_and(a, -b); }

LogicFunc logic_table[16] = {
    logic_false, logic_nor, logic_nand_a, logic_not_a,
    logic_nand_b, logic_not_b, logic_xor, logic_nand,
    logic_and, logic_xnor, logic_b, logic_a_or_not_b,
    logic_a, logic_not_a_or_b, logic_or, logic_true
};

// ∂f/∂a
float dfa_false(float a, float b) { (void)a; (void)b; return 0.0f; }
float dfa_true(float a, float b) { (void)a; (void)b; return 0.0f; }
float dfa_not_a(float a, float b) { (void)a; (void)b; return -1.0f; }
float dfa_not_b(float a, float b) { (void)a; (void)b; return 0.0f; }
float dfa_and(float a, float b) { (void)a; return (b + 1.0f) / 2.0f; }
float dfa_nand(float a, float b) { return -dfa_and(a, b); }
float dfa_or(float a, float b) { (void)a; return (1.0f - b) / 2.0f; }
float dfa_nor(float a, float b) { return -dfa_or(a, b); }
float dfa_xor(float a, float b) { (void)a; return -b; }
float dfa_xnor(float a, float b) { (void)a; return b; }
float dfa_a(float a, float b) { (void)a; (void)b; return 1.0f; }
float dfa_b(float a, float b) { (void)a; (void)b; return 0.0f; }
float dfa_a_or_not_b(float a, float b) { (void)a; return (1.0f + b) / 2.0f; }
float dfa_not_a_or_b(float a, float b) { (void)a; return (-1.0f + b) / 2.0f; }
float dfa_nand_a(float a, float b) { (void)a; return ( -b - 1.0f ) / 2.0f; }
float dfa_nand_b(float a, float b) { (void)a; return ( -b + 1.0f ) / 2.0f; }

// ∂f/∂b
float dfb_false(float b, float a) { (void)a; (void)b; return 0.0f; }
float dfb_true(float b, float a) { (void)a; (void)b; return 0.0f; }
float dfb_not_a(float b, float a) { (void)a; (void)b; return 0.0f; }
float dfb_not_b(float b, float a) { (void)a; (void)b; return -1.0f; }
float dfb_and(float b, float a) { (void)b; return (a + 1.0f) / 2.0f; }
float dfb_nand(float b, float a) { return -dfb_and(a, b); }
float dfb_or(float b, float a) { (void)b; return (1.0f - a) / 2.0f; }
float dfb_nor(float b, float a) { return -dfb_or(a, b); }
float dfb_xor(float b, float a) { (void)b; return -a; }
float dfb_xnor(float b, float a) { (void)b; return a; }
float dfb_a(float b, float a) { (void)a; (void)b; return 0.0f; }
float dfb_b(float b, float a) { (void)a; (void)b; return 1.0f; }
float dfb_a_or_not_b(float b, float a) { (void)b; return ( -a + 1.0f ) / 2.0f; }
float dfb_not_a_or_b(float b, float a) { (void)b; return (a + 1.0f) / 2.0f; }
float dfb_nand_a(float b, float a) { (void)b; return ( -a + 1.0f ) / 2.0f; }
float dfb_nand_b(float b, float a) { (void)b; return ( -a - 1.0f ) / 2.0f; }

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

void get_input_ids(int id, int size_in, int* a, int* b){
    *a = (id * 2 + (int)(id/size_in))%size_in;
    *b = (id * 2 + 1 + (int)(id/size_in) + (int)(id/(2*size_in)))%size_in;
}

// Créer un neurone
Neurone create_neurone(int i, int size_in) {
    Neurone n;
    n.id = i;
    n.inputs_size = size_in;
    
    for (int i = 0; i < 16; i++) n.gate_logits[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    n.link_logits[0] = (float*)malloc(size_in * sizeof(float));
    for (int i = 0; i < size_in; i++) n.link_logits[0][i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    n.link_logits[1] = (float*)malloc(size_in * sizeof(float));
    for (int i = 0; i < size_in; i++) n.link_logits[1][i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    n.link_weights[0] = (float*)malloc(size_in * sizeof(float));
    n.link_weights[1] = (float*)malloc(size_in * sizeof(float));

    int a,b;
    get_input_ids(i, size_in, &a, &b);
    n.link_logits[0][a] = 75.0f;
    n.link_logits[1][b] = 75.0f;

    normalize_neurone_softmax(&n);
    return n;
}

void free_neurone(Neurone n){
    free(n.link_logits[0]);
    free(n.link_logits[1]);
    free(n.link_weights[0]);
    free(n.link_weights[1]);
}

// Trouver l'ID de porte logique dominant
int get_dominante_gate_id(Neurone *n){
    int best_id = 0;
    float best_w = n->gate_weights[0];
    for (int i = 1; i < 16; ++i)
        if (n->gate_weights[i] > best_w) {
            best_w = n->gate_weights[i];
            best_id = i;
    }
    return best_id;
}

void softmax(float* in, float* out, float temperature, int size){
    // Chercher max après division
    float max = in[0] / temperature;
    for (int i = 1; i < size; i++)
        if (in[i] / temperature > max) max = in[i] / temperature;

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        out[i] = expf((in[i] / temperature) - max);
        sum += out[i];
    }
    for (int i = 0; i < size; i++) {
        out[i] /= sum;
    }
}

void normalize_neurone_softmax(Neurone *n) {
    softmax(n->gate_logits, n->gate_weights, 3.0f, 16);
    softmax(n->link_logits[0], n->link_weights[0], 1.0f, n->inputs_size);
    softmax(n->link_logits[1], n->link_weights[1], 1.0f, n->inputs_size);
}


float forward(Neurone *n, float* in) {
    float out = 0.0f;
    for (int ai = 0; ai < n->inputs_size; ai++)
    for (int bi = 0; bi < n->inputs_size; bi++)
    for (int i = 0; i < 16; i++)
        out += n->link_weights[0][ai] * n->link_weights[1][bi] *  n->gate_weights[i] * logic_table[i](in[ai], in[bi]);
    return n->output = out;
}

float gradient_weights_gate(Neurone *n, float* inputs, int weights_id) {
    float sum = 0;
    for (int ai = 0; ai < n->inputs_size; ai++)
    for (int bi = 0; bi < n->inputs_size; bi++) // somme pour tout les couple d'entre 
        sum += n->link_weights[0][ai] * n->link_weights[1][bi] * logic_table[weights_id](inputs[ai], inputs[bi]);
    return sum;
}

float gradient_weights_link(Neurone *n, float* inputs, int input_id, int port) {
    float sum = 0;
    for (int j = 0; j < n->inputs_size; j++) // somme pour toute les deuxieme entree posible 
    for (int i = 0; i < 16; i++) { // pour toute les porte 
        sum += inputs[input_id] * n->link_weights[!port][j] * n->gate_weights[i] * logic_deriv[port][i](inputs[input_id], inputs[j]);
    }
    return sum;
}

float gradient_input(Neurone *n, float* inputs, int input_id) {
    float sum = 0;
    for (int j = 0; j < n->inputs_size; j++) // somme pour toute les deuxieme entree posible 
    for (int i = 0; i < 16; i++) { // pour toute les porte 
        sum += n->link_weights[0][input_id] * n->link_weights[1][j] * n->gate_weights[i] * logic_deriv[0][i](inputs[input_id], inputs[j]);
        sum += n->link_weights[1][input_id] * n->link_weights[0][j] * n->gate_weights[i] * logic_deriv[1][i](inputs[input_id], inputs[j]);
    }
    return sum;
}

void gradient_logit_from_gradient_weights(float* w, float* grad_w, int size, float* grad_l){
    for (int j = 0; j < size; j++) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            float delta = (i == j) ? 1.0f : 0.0f;
            sum += grad_w[i] * w[i] * (delta - w[j]);
        }
        grad_l[j] = sum;
    }
}

void backward(Neurone *n,float* inputs, float* grad_output, float learning_rate, float* grad_in) {
    
    // Gradient des logits via chain rule softmax
    // dL/dz_j = sum_i dL/dw_i * dw_i/dz_j
    // dw_i/dz_j = softmax_i * (delta_ij - softmax_j)
    // Ici : dL/dw_i = grad_output * logic_table[i](a,b)
    
    float grad_wg[16];
    for (int i = 0; i < 16; i++) grad_wg[i] = grad_output[n->id] * gradient_weights_gate(n,inputs,i);
    
    // Calcul dL/dz_j
    float grad_lg[16];
    gradient_logit_from_gradient_weights(n->gate_weights, grad_wg, 16, grad_lg);
    // Mise à jour logits
    for (int i = 0; i < 16; i++) n->gate_logits[i] -= learning_rate * grad_lg[i];
    
    // calcul backwrd lien  
    float grad_wl[2][n->inputs_size];
    for (int i = 0; i < n->inputs_size; i++) grad_wl[0][i] += grad_output[n->id] * gradient_weights_link(n,inputs,i,0);
    for (int i = 0; i < n->inputs_size; i++) grad_wl[1][i] += grad_output[n->id] * gradient_weights_link(n,inputs,i,1);
    float grad_ll[2][n->inputs_size];
    gradient_logit_from_gradient_weights(n->link_weights[0], grad_wl, n->inputs_size, grad_ll[0]);
    gradient_logit_from_gradient_weights(n->link_weights[1], grad_wl, n->inputs_size, grad_ll[1]);

    //printf("%f\n",grad_ll[0][0]);

    for (int i = 0; i < n->inputs_size; i++) n->link_logits[0][i] -= learning_rate * grad_ll[0][i] * 0.001;
    for (int i = 0; i < n->inputs_size; i++) n->link_logits[1][i] -= learning_rate * grad_ll[1][i] * 0.001;

    normalize_neurone_softmax(n);

    for (int i = 0; i < n->inputs_size; i++)
        grad_in[i] += gradient_input(n, inputs, i) * grad_output[n->id];
}