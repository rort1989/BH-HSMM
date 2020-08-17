## Bayesian Hierarchical Hidden Semi-Markov Model (BH-HSMM)

BH-HSMM is a probabilistic dynamic model which explicitly models spatial and temporal variations in the dynamic data. The temporal variation is handled in two aspects. First, we incorporate a probabilistic duration mechanism to allow flexible speed at each phase of an activity. Second, the transitions among different phases of an activity are modeled by transition probabilities among different hidden states. The spatial variation is modeled by probability distribution on observations in each individual frame. To further improve the capability of handling intra-class variation, we extend the model following the Bayesian framework, by allowing the parameters to vary across data, yielding a hierarchical structure.

## How to Use

This repository provides an implementation of HDM for synthesis task. It includes the following key components:

1. Adversarial Bayesian inference

2. Synthesizing motion data

3. Quantitatively evaluation of synthesis quality

4. Visualization of mocap data

To perform 1-4, first download the [data](https://drive.google.com/drive/folders/1scWaMzs7V-To5jKHh6QWIolqrXUglenK?usp=sharing) and add the entire data folder to the repository root directory. Then run 'script_data_synthesis_adversarial_complete_hdm_real_revised.m' script in Matlab, follow the prompt in command window to select available dataset. 

## Dependencies

[Bayes Net Toolbox (BNT)](https://github.com/bayesnet/bnt)

[Minimum Mean Discrepancy Toolbox (MMD)](http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm)

[NLTK (also requires Matlab to call python script)](https://www.nltk.org/)

## Related Publications

The code and data are used to produce the experiment results reported in the following reference.

Rui Zhao, Hui Su and Qiang Ji, "Bayesian Adversarial Human Motion Synthesis," IEEE Conference on Computer Vision Pattern Recognition, 2020. [[pdf]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Bayesian_Adversarial_Human_Motion_Synthesis_CVPR_2020_paper.pdf)[[supp]](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Zhao_Bayesian_Adversarial_Human_CVPR_2020_supplemental.zip)

```
@InProceedings{Zhao_2020_CVPR,
author = {Zhao, Rui and Su, Hui and Ji, Qiang},
title = {Bayesian Adversarial Human Motion Synthesis},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

Another related publication.

Rui Zhao and Qiang Ji. An Adversarial Hierarchical Hidden Markov Model for Human Pose Modeling and Generation. AAAI, 2018. [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16391/15983)[[supp]](https://ecse.rpi.edu/~cvrl/zhaor/files/Zhao2018_supp.pdf)
```
@paper{AAAI1816391,
	author = {Rui Zhao and Qiang Ji},
	title = {An Adversarial Hierarchical Hidden Markov Model for Human Pose Modeling and Generation},
	conference = {AAAI Conference on Artificial Intelligence},
	year = {2018},
}
```

## License Condition

Copyright (C) 2020 Rui Zhao 

Distribution code version 1.0 - 08/16/2020. MIT License.
