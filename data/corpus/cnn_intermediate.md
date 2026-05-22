# Convolutional Neural Networks (CNN)

## The Convolution Operation

Convolution applies a small learnable filter across spatial locations of an input, producing feature maps that detect local patterns such as edges, textures, or shapes. Unlike fully connected layers, convolutions share weights across positions, which dramatically reduces parameter count and encodes translation equivariance: the same pattern detected in one region can be detected elsewhere. Each filter produces one channel in the output feature map. Stacking multiple filters in a convolutional layer allows the network to learn many pattern detectors in parallel. Padding and stride control output spatial dimensions: valid convolution shrinks the map, while same padding preserves size when stride is one.

## Pooling Layers

Pooling downsamples feature maps to reduce spatial resolution and computational cost while providing a degree of translation invariance. Max pooling selects the maximum value in each local window and is the most common choice because it preserves strong activations. Average pooling takes the mean and can smooth responses. Pooling does not have learnable parameters. In classic architectures, pooling follows convolution and activation, progressively shrinking the representation until global features are fed to fully connected or global pooling layers for classification.

## LeNet and the CNN Design Pattern

LeNet-5 (LeCun et al., 1998) established the canonical CNN pattern for digit recognition: convolution, activation, pooling, repeated, then fully connected classifier. This demonstrated that hierarchical local feature learning works for images. The design insight is that early layers capture generic low-level features while deeper layers combine them into class-specific representations. Interview questions often ask you to walk through dimensions: input image, conv layer output size formula, pooling effect, and final vector length before the softmax layer.

## AlexNet and Deep CNN Training

AlexNet (Krizhevsky et al., 2012) scaled CNNs to ImageNet by using ReLU activations, dropout, data augmentation, and GPU training. It showed that depth and data matter for large-scale visual recognition. AlexNet used overlapping max pooling and multiple parallel convolution streams in one layer (the grouped convolution idea in early form). Its success triggered the modern deep learning era for computer vision. When discussing AlexNet, mention that ReLU reduced saturation compared to tanh, and that dropout regularization made training deep models on limited labeled data more reliable.
