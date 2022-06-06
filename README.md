# README

<font color="red">You will need two oracles saved as .h5 to run this project. These are not included in this repo as the file sizes are too large. Please contact owner of this repo for the files.

This repository is for the final project of DSA5204. In this project we reproduce and extend the paper titled "Generative Adversarial Active Learning" (GAAL) http://arxiv.org/abs/1702.07956.

# Group Members

# About this Repo

## Repo File Structure

```
GAAL______
    |_ gans                              # folders containing generators from different GANs  
        |_ cifar10-dcgan                 # generators from a DC-GAN  
        |_ cifar10-doublelastlayer       # generators from a DC-GAN with a double sized last Conv2DTranspose layer  
        |_ cifar10-wgan                  # generators from a Wasserstein GAN  
        |_ cifar10-wgan-doublelastlayer  # generators from a Wasserstein GAN with a double sized last Conv2DTranspose layer  
        |_ mnist                         # generators from a DC-GAN  
    |_ oracles                           # oracles used for labelling synthetic images, please approach "shaunn .dot. tan .at. gmail.com" for the files to be placed in this folder  
    |_ Report  
        |_ Assets                        # Assets used in our report  
    |_ results                           # pickle files with results from our replication and extensions, use `Read results.ipynb` to generate charts  
    |_ trainers                          # .py files used in conjunction with the Jupyter Notebook `Generative Adversarial Active Learning.ipynb` to perform various training algorithms  
    |_ usps                              # usps digits dataset used as test dataset in replication  
    Generative Adversarial Active Learning.ipynb       # main file used to review the work  
    README.md                            # this readme
    Report.pdf                           # a write-up submitted on this topic
    requirements.txt                     # for installing prequisuites with pip  
````
## Prerequisites

After cloning this repository, please the following command in command prompt/terminal to ensure the that prerequisites are installed.

```python
pip install -r requirements.txt
```
## How to Run This Project

Please open the jupyter notebook named `Generative Adversarial Active Learning.ipynb` for more details.

Each .py trainer has an accompanying docstring that explains its use.

# Generative Adversarial Active Learning (GAAL)

## Abstract

Labelling of data for supervised training is time consuming and costly. The authors in the selected paper proposed a novel training framework combining Active Learning and Generative Adversarial Networks. Generative Adversarial Active Learning (GAAL) attempts to utilize informative synthetic data generated from a GAN to increase training speed of a learner (a Support Vector Classifier in the authors' work). We replicated the main results from the authors' experiments, compared our results and proposed extensions, which includes replacing the Deep Convolutional-GAN with a Wasserstein GAN and incorporating diversity measure in the objective function.

## Introduction

Compared to labelled data, unlabelled data is relatively abundant. However, in order to be able to perform a supervised learning task, these unlabelled data will need to be labelled and the process is time consuming and costly.  

Active learning algorithms seek to maximize the accuracy of trained learners with fewer labelled training samples by strategic selection of samples from a pool of unlabelled data (queries), labelling these selected samples, and adding these samples to a labelled pool to update the learner.  

The selected paper introduces a novel active learning framework by introducing a query synthesis approach that combines aspects of Active Learning and GANs, aptly named "Generative Adversarial Active Learning". Firstly, utilising a generator from a GAN trained on the pool of unlabelled data, synthesize informative training samples that are adapted to the learner that is being trained. Then, human oracles label these synthetic samples and add them to the labelled pool to update the learner. Iterate these steps till the labelling budget is reached.

In the selected paper, the authors specified the learner to be trained as a Support Vector Classifier (SVC)

## Summary & Further Work

In our report, we presented the results from our replication of the experiments conducted for the paper by on GAAL. We noted some differences in our replication results and provided some insights to the likely causes of the differences. We proposed to extensions to the authors' work, replacing the DC-GAN with a WGAN in view of some of its benefits, changing the size of the generator and adding a diversity measure to the GAAL objective function. 

We noted no clear benefits from using a Generator from a WGAN, but as expected, increasing the size of the Generator has a positive impact on the accuracy of the SVC. The included diversity measure also had an impact on the accuracy of the SVC, and its impact is affected by the tunable hyperparameter.

From our observations, the GAAL training algorithm is sensitive to the quality of the Generator. Additional work into improving the Generator is likely able to push the learner's accuracy even closer to the fully supervised level with much fewer samples. 

Noting that diversity measures appear to have an impact on the synthetic images, and consequently the accuracy of the learner, we could consider utilising other varieties of measures, such as cosine similarity, in the objective function. Further, the penalty factor is tunable and could be fine-tuned to improve the accuracy of the learner when its trained under GAAL.
