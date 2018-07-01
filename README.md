## Purpose:
Code to create , run and test a vanilla neural network with variable number of layers/ number of nodes per layer (dropout option for training) . Supports sigmoid activation for now, other activation functions will be added in the future. No deep learning frameworks (Tensorflow , caffe, torch etc ... ) were used.

## Dependencies:
- Numpy
- scipy
- scikit

## Run Instructions:

If you want to use the same architecture as the one I chose, run main.py.
If you want to specify your own parameters (Number of Layers/Nodes per Layer), please add them as arguments.
Run main.py -h for more instructions

## Implementation:

- Two optimization implementations are given, an optimized one using a scikit and a classical one.

- Dropout is implemented 

## Results:

- Prints accuracy on test set as well as results .

Parts were slightly adapted from https://github.com/tombarratt46/Python-Iris-Neural-Net-Classification.
