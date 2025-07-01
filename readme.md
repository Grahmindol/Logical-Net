# 🧠 Réseau de neurones logiques différentiables en C

Ce projet implémente un réseau de neurones basé sur des **portes logiques différentiables**, inspiré du concept de *Logic Gate Networks* (LGN) décrit dans [l'article arXiv:2411.04732](https://arxiv.org/pdf/2411.04732).

## 💡 Concept

- Chaque neurone applique une combinaison pondérée de **portes logiques** (XOR, AND, OR, etc.).
- Les poids sont appris via un softmax différentiable sur les "logits" des fonctions logiques.
- La structure du réseau est composée de plusieurs couches, chaque neurone ayant 2 entrées et 16 fonctions logiques possibles.

## ⚙️ Fonctionnalités

- **Forward pass** complet
- **Backward pass** manuel, propagation du gradient couche par couche
- **Normalisation des poids** avec softmax à température ajustable
- **Affichage détaillé du réseau** sous forme de tableau, avec les poids probabilisés par neurone et par fonction logique
- Support de l’entraînement supervisé avec gradient et loss (log loss)

## 📄 Structure du code

```
.
├── bin
│   ├── main
│   └── obj
│       ├── layer.o
│       ├── main.o
│       ├── network.o
│       └── neurone.o
├── header
│   ├── layer.h
│   ├── network.h
│   └── neurone.h
├── src                 -----> libs bien nommé
│   ├── layer.c
│   ├── network.c
│   └── neurone.c     
├── Makefile
├── readme.md
└── main.c              -----> exemple d'utilisation

```

## 🛠️ Compilation

```bash
make && ./bin/main
```


