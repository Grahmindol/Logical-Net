#ifndef NEURONE_H
#define NEURONE_H

#include <math.h>

typedef struct {
    float logits[16];   // poids avant softmax
    float weights[16];  // poids softmaxés (proba)
    float output;
    float a, b;
} Neurone;

void backward(Neurone *n, float grad_output, float learning_rate);
float forward(Neurone *n, float a, float b);

void normalize_neurone_softmax(Neurone *n);

#endif