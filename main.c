
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "neurone.h"


int main() {
    srand(time(NULL));
    Neurone n;

    // Initialisation logits aléatoire
    for (int i = 0; i < 16; i++) {
        n.logits[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }
    normalize_neurone_softmax(&n);

    float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    float targets[4] = {0, 0, 0, 1}; // AND

    float lr = 0.1f;
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss = 0.0f;
        for (int i = 0; i < 4; i++) {
            float a = inputs[i][0];
            float b = inputs[i][1];
            float y = targets[i];
            float out = forward(&n, a, b);
            float err = out - y;
            loss += err * err * 0.5f;
            backward(&n, err, lr);
        }
        if (epoch % 1000 == 0) {
            printf("Epoch %d, loss = %.6f\n", epoch, loss);
        }
    }

    // Test final
    printf("\nTest final:\n");
    for (int i = 0; i < 4; i++) {
        float a = inputs[i][0];
        float b = inputs[i][1];
        float out = forward(&n, a, b);
        printf("AND(%.0f, %.0f) = %.4f\n", a, b, out);
    }
    return 0;
}
