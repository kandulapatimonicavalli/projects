# Artificial Neural Networks (ANN)

## Forward Propagation

Forward propagation is the process of passing input data through a neural network layer by layer to produce a prediction. Each neuron computes a weighted sum of its inputs plus a bias, then applies an activation function. For a network with layers L1, L2, and L3, the output of layer L1 becomes the input to L2, and so on until the final layer produces the prediction. Matrix multiplication makes this efficient: if X is the input batch and W is the weight matrix, the pre-activation is Z = XW + b. The choice of activation function at each hidden layer determines whether the network can model non-linear relationships. Without non-linear activations, stacking linear layers would still be equivalent to a single linear transformation.

## Backpropagation

Backpropagation trains neural networks by computing gradients of the loss with respect to every weight using the chain rule. Training starts with forward propagation to compute the loss. The error signal is then propagated backward from the output layer to earlier layers. Each weight is updated in the opposite direction of its gradient, scaled by a learning rate. The key insight is that gradients for earlier layers are products of local derivatives along the path from output to that layer. This is why very deep networks without careful design can suffer from vanishing gradients: repeated multiplication of small derivatives shrinks the signal before it reaches the first layers.

## Activation Functions

Activation functions introduce non-linearity so neural networks can learn complex patterns. The sigmoid function maps inputs to (0, 1) but saturates at extremes, which slows learning. ReLU (rectified linear unit) outputs max(0, x) and is the default choice in modern feedforward networks because it avoids saturation for positive inputs and computes quickly. Tanh is similar to sigmoid but zero-centered, which can help convergence. In interview settings, you should explain that activation choice affects gradient flow: ReLU can cause dead neurons when inputs are always negative, while sigmoid and tanh can cause vanishing gradients when activations saturate.

## Loss Functions and Training

The loss function measures how wrong the network's predictions are. Mean squared error is common for regression; cross-entropy loss is standard for classification because it penalizes confident wrong predictions heavily. Stochastic gradient descent and its variants (Adam, RMSprop) update weights using mini-batches of data rather than the full dataset, which makes training feasible on large datasets. Regularization techniques such as L2 weight decay and dropout reduce overfitting by discouraging overly complex solutions. A well-prepared candidate can connect loss design to the learning objective: classification, regression, and multi-label problems each imply different loss choices.
