# README

This repository is for the final project of DSA5204.  In this project we reproduce and extend the paper "Generative Adversarial Active Learning"(GAAL) http://arxiv.org/abs/1702.07956

[TOC] 

## Group Member

## Prerequisite

Run following commands to help to set up the environment

```python
pip install -r requirements.txt
```

## File Structure



## How to Run This Project

### Get SVM Model by Active Learning 

Our basic idea is to train a SVM model with "Generative Adversarial Active Learning"(GAAL) to utilize the advantages of generating more informative instances. 

Following commands can help you train a SVM model with a pre-trained oracle and a pre-trained generator

```python

```

### Train a Oracle for Active Learning

In the paper "GAAL", a human oracle for delivering response of most informative queries is required. But in our project, we replace the human oracle with a high-performance neural network based on VGG16.

You can run the following commands to get a new oracle with default setting.

```python

```

### Train a Generator for Active Learning

In our project we develop two ways to train a generator.

**Train a Simple GAN **

```python
```

**Train a Wasserstein GAN**

```python
```

### Utilize Existing Model/Oracle/Generator

We save the pre-train model/oracle/generator in our project, and you can utilize them with following commands

**Call Oracle to classify image**

```python
```

**Call Generator to generate new image**

```python

```

**Call **
