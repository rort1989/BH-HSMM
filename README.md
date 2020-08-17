## Hierarchical Dynamic Model (HDM)

Hierarchical dynamic model (HDM) is a probabilistic dynamic model which explicitly models spatial and temporal variations in the dynamic data. The temporal variation is handled in two aspects. First, we incorporate a probabilistic duration mechanism to allow flexible speed at each phase of an activity. Second, the transitions among different phases of an activity are modeled by transition probabilities among different hidden states. The spatial variation is modeled by probability distribution on observations in each individual frame. To further improve the capability of handling intra-class variation, we extend the model following the Bayesian framework, by allowing the parameters to vary across data, yielding a hierarchical structure.

## How to Use

This repository provides an implementation of HDM for synthesis task. It includes the following key components:

1. Adversarial Bayesian inference

2. Synthesizing motion data

3. Quantitatively evaluation of synthesis quality

4. Visualization of mocap data

To perform 1-4, run 'script_data_synthesis_adversarial_complete_hdm_real_revised.m' script in Matlab, follow the prompt in command window to select available dataset. 

## Dependencies

Bayes Net Toolbox (BNT)

Minimum Mean Discrepancy Toolbox (MMD)

NLTK (also requires Matlab to call python script)

## Related Publication

The code and data are used to produce the experiment results reported in the following reference.

Rui Zhao, Hui Su and Qiang Ji, "Bayesian Adversarial Human Motion Synthesis," IEEE Conference on Computer Vision Pattern Recognition, 2020.

Another related publication.

Rui Zhao and Qiang Ji. An Adversarial Hierarchical Hidden Markov Model for Human Pose Modeling and Generation. AAAI, 2018.

## License Condition

Copyright (C) 2020 Rui Zhao 

Distribution code version 1.0 - 08/16/2020. MIT License.
