
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "network.h"

float loss(float y_pred, float y_true) {
    return 0.5f * (y_pred - y_true) * (y_pred - y_true);
}

float dloss(float y_pred, float y_true) {
    return y_pred - y_true;
}

// Fonction cible
void target(bool* input, float* output){
    output[0] = (input[0] || input[1]) && (input[2] || input[3]);
}

int main() {
    srand(time(NULL));

    // Architecture : 2 couches → 2 neurones, puis 1 neurone en sortie
    int layer_sizes[3] = {4, 2, 1};
    Network *net = create_network(2, layer_sizes);

    const float lr = 0.01f;
    const int epochs = 100000;

    for (int e = 0; e < epochs; e++) {
        float total_loss = 0;

        for (int i = 0; i < 16; i++) {
            // Préparer les entrées binaires
            bool bin_inputs[4] = {
                (i >> 3) & 1,
                (i >> 2) & 1,
                (i >> 1) & 1,
                (i >> 0) & 1
            };
            float inputs[4] = {
                (float)bin_inputs[0],
                (float)bin_inputs[1],
                (float)bin_inputs[2],
                (float)bin_inputs[3]
            };

            float outputs[1];
            forward_network(net, inputs, outputs);

            // Cible
            float target_outputs[1];
            target(bin_inputs, target_outputs);

            // Gradient sortie
            float grad_out[1] = {
                dloss(outputs[0], target_outputs[0])
            };

            total_loss += loss(outputs[0], target_outputs[0]);

            // Rétroprop (attention : grad_size = 1 ici)
            backward_network(net, grad_out, lr);
        }

        if (e % 5000 == 0) {
            printf("Epoch %d, loss = %f\n", e, total_loss);
        }
    }

    // Test final
    printf("\nRésultats après entraînement :\n");
    for (int i = 0; i < 16; i++) {
        bool bin_inputs[4] = {
            (i >> 3) & 1,
            (i >> 2) & 1,
            (i >> 1) & 1,
            (i >> 0) & 1
        };
        float inputs[4] = {
            (float)bin_inputs[0],
            (float)bin_inputs[1],
            (float)bin_inputs[2],
            (float)bin_inputs[3]
        };
        float outputs[1];
        forward_network(net, inputs, outputs);

        float target_outputs[1];
        target(bin_inputs, target_outputs);

        printf("a=%d b=%d c=%d d=%d → out=%.3f (target=%d)\n",
               bin_inputs[0], bin_inputs[1], bin_inputs[2], bin_inputs[3],
               outputs[0], (int)target_outputs[0]);
    }
    print_network(net);

    free_network(net);
    return 0;
}