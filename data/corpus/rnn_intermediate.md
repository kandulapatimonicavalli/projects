# Recurrent Neural Networks (RNN)

## Hidden State and Sequence Processing

Recurrent neural networks process sequences by maintaining a hidden state that is updated at each time step. At step t, the hidden state h_t is a function of the current input x_t and the previous hidden state h_{t-1}. This recurrence allows the network to carry information across time, which is essential for language, speech, and time series. The same weight matrices are applied at every time step, giving parameter sharing across sequence positions. However, standard RNNs struggle to learn long-range dependencies because gradients propagated through many time steps tend to vanish or explode without stabilization techniques.

## Backpropagation Through Time (BPTT)

BPTT unrolls the recurrent network across time steps and applies backpropagation on the unrolled graph. The gradient with respect to early time steps involves products of repeated Jacobian terms. When these terms are consistently smaller than one, gradients vanish and early inputs barely affect learning. When larger than one, gradients explode. Truncated BPTT limits unrolling to a fixed window to save memory and stabilize training. In interviews, explain that BPTT is why vanilla RNNs are poor at remembering information from many steps ago compared to gated architectures such as LSTM.

## Vanishing Gradients in RNNs

Vanishing gradients occur when error signals shrink exponentially as they propagate backward through long unrolled chains. For language modeling or long sequences, the model may fail to credit words or events far in the past. Techniques include careful initialization, gradient clipping for explosion, ReLU-like variants in recurrent cells, and gated units. LSTM and GRU were designed specifically to create paths where gradients can flow with less attenuation. Comparing feedforward vanishing gradients to RNN vanishing gradients is a strong interview move: in RNNs the problem is worse because depth in time can be hundreds of steps.

## When to Use RNNs vs Feedforward Models

Use RNNs when the order and temporal structure of inputs matter: machine translation before transformers, speech recognition, sensor streams, and character-level language models. For fixed-size inputs without order, feedforward or CNN models are simpler and easier to train. Today many sequence tasks use transformers, but understanding RNNs remains important for interviews because they illustrate sequence modeling fundamentals, BPTT, and the motivation for attention and gating. A practical answer acknowledges tradeoffs: RNNs are sequential and harder to parallelize; transformers parallelize across positions but require more data and compute.
