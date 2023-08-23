# Drug-Target Affinity Prediction

This repository contains code for predicting drug-target affinity using a graph neural network (GCN) model. The model processes chemical structures of drugs and protein sequences of targets to make predictions.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
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

### Dependencies

To run the code in this repository, you need the following dependencies:

- Python (>=3.6)
- PyTorch (>=1.6)
- PyTorch Lightning (>=1.0)
- Optuna (>=2.0)
- RDKit (for chemical structure processing)
- NetworkX (for graph manipulation)
- ...

You can install the required Python packages using the provided `Pipfile` and `Pipfile.lock`:

```bash
pipenv install
