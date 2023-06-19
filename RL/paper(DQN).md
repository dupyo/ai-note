# Playing Atari with Deep Reinforcement Learning


## Abstract

We present the first deep learning model to successfully learn control policies directly from high dimensional sensory input using reinforcement learning. 
The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. 
We apply our method to seven Atari 2600 games from the Arcade Learning Environment, with no adjustment of the architecture or learning algorithm. 
We find that it out performs all previous approaches on six of the games and surpasses a human expert on three of them.

## Introduction

Learning to control agents directly from high-dimensional sensory inputs like vision and speech is one of the long-standing challenges of reinforcement learning(RL). 
Most successful RL applications that operate on these domains have relied on hand-crafted features combined with linear value functions or policy representations. 
Clearly, the performance of such system sheavily relies on the quality of the feature representation. 

Recent advances in deep learning have made it possible to extract high-level features from raw sensory data, leading to break throughs in computer vision[11, 22, 16] and speech recognition[6, 7]. 
These methods utilise a range of neural network architectures, including convolutional networks, multilayer perceptrons, restricted Boltzmann machines and recurrent neural networks, and have exploited both supervised and unsupervised learning. 
It seems natural to ask whether similar techniques could also be beneficial for RL with sensory data. 

However reinforcement learning presents several challenges from a deep learning perspective. 
Firstly, most successful deep learning applications to date have required large amounts of hand labelled training data. 
RL algorithms, on the other hand, must be able to learn from a scalar reward signal that is frequently sparse, noisy and delayed. 
The delay between actions and resulting rewards, which can be thousands of time steps long, seems particularly daunting when compared to the direct association between inputs and targets found in supervised learning.
Another issue is that most deep learning algorithms assume the data samples to be independent, while in reinforcement learning one typically encounters sequences of highly correlated states. 
Furthermore, in RL the data distribution changes as the algorithm learns new behaviours, which can be problematic for deep learning methods that assume a fixed under lying distribution.
