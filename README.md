# Mean teachers are better role models
#### Weight-averaged consistency targets improve semi-supervised deep learning results

* [Paper](https://arxiv.org/abs/1703.01780)
* [NIPS 2017 poster](nips_2017_poster.pdf)
* [NIPS 2017 spotlight slides](nips_2017_slides.pdf)

By Antti Tarvainen, Harri Valpola

[The Curious AI Company](https://thecuriousaicompany.com/)

## Approach

Mean Teacher is a simple method for semi-supervised learning. It consists of the following steps:

1. Take a supervised architecture and make a copy of it. Let's call the original model the **student** and the new one the **teacher**.
2. At each training step, use the same minibatch as inputs to both the student and the teacher but add random augmentation or noise to the inputs separately.
3. Add an additional *consistency* cost between the student and teacher output (after softmax).
4. Let the optimizer update the student weights normally.
5. Let the teacher weights be an exponential moving average (EMA) of the student weights. That is, after each training step, update the teacher weights a little bit toward the student weights.

Our contribution is the last step. Laine and Aila [\[paper\]](https://arxiv.org/abs/1610.02242) used shared parameters between the student and the teacher, or used a temporal ensemble of teacher predictions. In comparison, Mean Teacher is more accurate and applicable to large datasets.

![Mean Teacher model](mean_teacher.png)

Mean Teacher works well with modern architectures. Combining Mean Teacher with ResNets, we improved the state of the art in semi-supervised learning on the ImageNet and CIFAR-10 datasets.

ImageNet using 10% of the labels | top-5 validation error
---------------------------------|------------------------
Variational Auto-Encoder [\[paper\]](https://arxiv.org/abs/1609.08976) | 35.42 ± 0.90
Mean Teacher ResNet-152          |  **9.11 ± 0.12**
All labels, state of the art [\[paper\]](https://arxiv.org/pdf/1709.01507.pdf) |  3.79

CIFAR-10 using 4000 labels   | test error
-----------------------------|-----------
CT-GAN [\[paper\]](https://openreview.net/forum?id=SJx9GQb0-) | 9.98 ± 0.21
Mean Teacher ResNet-26	     | **6.28 ± 0.15**
All labels, state of the art [\[paper\]](https://arxiv.org/abs/1705.07485) | 2.86


## Implementation

See tensorflow directory for the TensorFlow implementation and instructions on how to use it.
PyTorch implementation coming up!
