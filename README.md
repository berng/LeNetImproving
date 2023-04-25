# LeNetImproving
Variants of LeNet-like architectures with increased accuracy at MNIST dataset by using Absolute activation function and modified ADAM optimizer

Code to the paper: https://arxiv.org/abs/2304.11758

Improving Classification Neural Networks by
using Absolute activation function
(MNIST/LeNET-5 example)
Oleg I.Berngardt
April 25, 2023

Abstract
The paper discusses the use of the Absolute activation function in
classification neural networks. An examples are shown of using this acti-
vation function in simple and more complex problems. Using as a baseline
LeNet-5 network for solving the MNIST problem, the efficiency of Abso-
lute activation function is shown in comparison with the use of Tanh,
ReLU and SeLU activations. It is shown that in deep networks Absolute
activation does not cause vanishing and exploding gradients, and therefore
Absolute activation can be used in both simple and deep neural networks.
Due to high volatility of training networks with Absolute activation, a
special modification of ADAM training algorithm is used, that estimates
lower bound of accuracy at any test dataset using validation dataset anal-
ysis at each training epoch, and uses this value to stop/decrease learning
rate, and reinitializes ADAM algorithm between these steps. It is shown
that solving the MNIST problem with the LeNet-like architectures based
on Absolute activation allows to significantly reduce the number of trained
parameters in the neural network with improving the prediction accuracy.

