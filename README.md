# Drug-Target Affinity Prediction

This repository contains code for predicting drug-target affinity using a graph neural network (GCN) model. The model processes chemical structures of drugs and protein sequences of targets to make predictions.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict drug-target affinity using a Graph Neural Network (GCN) model. The model takes as input the chemical structure of a drug and the protein sequence of a target and outputs a prediction of the affinity between them. The project consists of the following main components:

- `data_prep.py`: Contains data preprocessing functions and a custom dataset class for loading and processing the dataset.
- `model.py`: Defines the GCN model architecture using PyTorch and PyTorch Lightning.
- `train.py`: Implements hyperparameter tuning using the Optuna library and trains the GCN model.

## Getting Started

To get started with this project, follow these steps:
Clone this repository:

   ```bash
   git clone https://github.com/your-username/GRL-DTA.git
   cd GRL-DTA



You can install the required Python packages using the provided `Pipfile` and `Pipfile.lock`:

```bash
pipenv install



This will create a virtual environment and install all the necessary Python dependencies listed in the `Pipfile.lock file.
Activate the virtual environment:



```bash
pipenv shell
