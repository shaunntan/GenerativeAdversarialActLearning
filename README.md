# README

<font color="red">You will need two oracles saved as .h5 to run this project. These are not included in this repo as the file size is too large. Please contact shaunn .dot. tan .at. gmail.com to obtain the files.</font>

This repository is for the final project of DSA5204. In this project we reproduce and extend the paper titled "Generative Adversarial Active Learning" (GAAL) http://arxiv.org/abs/1702.07956.

# Group Members

# About this Repo

## Repo File Structure

GAAL______
    |_ gans                              # folders containing generators from different GANs
        |_ cifar10-dcgan                 # generators from a DC-GAN
        |_ cifar10-doublelastlayer       # generators from a DC-GAN with a double sized last Conv2DTranspose layer
        |_ cifar10-wgan                  # generators from a Wasserstein GAN
        |_ cifar10-wgan-doublelastlayer  # generators from a Wasserstein GAN with a double sized last Conv2DTranspose layer
        |_ mnist                         # generators from a DC-GAN
    |_ oracles                           # oracles used for labelling synthetic images, please approach shaunn .dot. tan .at. gmail.com for the files to be placed in this folder.
    |_ Report
        |_ Assets                        # Assets used in our report
    |_ results                           # pickle files with results from our replication and extensions, use `Read results.ipynb` to generate charts
    |_ trainers                          # .py files used in conjunction with the Jupyter Notebook `Generative Adversarial Active Learning.ipynb` to perform various training algorithms
    |_ usps                              # usps digits dataset used as test dataset in replication
    Generative Adversarial Active Learning.ipynb       # main file used to review the work
    README.md                            # this readme
    requirements.txt                     # for installing prequisuites with pip

## Prerequisites

After cloning this repository, please the following command in command prompt/terminal to ensure the that prerequisites are installed.

```python
pip install -r requirements.txt
```
## How to Run This Project

Please open the jupyter notebook named `Generative Adversarial Active Learning.ipynb` for more details.

Each .py trainer has an accompanying docstring that explains it's use.

