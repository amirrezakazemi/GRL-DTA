# Drug-Target Affinity Prediction

This repository contains code for predicting drug-target affinity using a graph neural network (GCN) model. The model processes chemical structures of drugs and protein sequences of targets to make predictions.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Results](#Results)
- [Contributing](#Contributing) 
- [License](#license)

## Introduction

This project aims to predict drug-target affinity using a Graph Neural Network (GCN) model. The model takes as input the chemical structure of a drug and the protein sequence of a target and outputs a prediction of the affinity between them. The project consists of the following main components:

- `data_prep.py`: Contains data preprocessing functions and a custom dataset class for loading and processing the dataset.
- `model.py`: Defines the GCN model architecture using PyTorch and PyTorch Lightning.
- `train.py`: Implements hyperparameter tuning using MLflow and Optuna libraries and trains the GCN model.
- `app.py`: Deploys the trained model using Flask to provide real-time predictions.

## Getting Started

`To get started with this project, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/GRL-DTA.git
   cd GRL-DTA‍

2.  Build the Docker image using the following command:

    `docker build -t dta-predictor .`

    This command will build a Docker image named `dta-predictor` using the instructions in the `Dockerfile`.

3.  Once the image is built, you can run a container using the following command:

    `docker run -p 9696:9696 dta-predictor`

    This command will start a Docker container based on the `dta-predictor` image and map port `9696` on the host to port `9696` in the container.

4.  The Flask application should now be accessible at `http://localhost:9696`. You can make predictions by sending POST requests to the `/app` endpoint.

## Results


You can find the results of model training and evaluation in the MLflow tracking server. The experiment named "drug-target-affinity-exp" contains logged hyperparameters, metrics, and artifacts.

## Contributing

Contributions to this project are welcome! If you find any issues or want to improve the code, feel free to open pull requests or issues in this repository.

## License

This project is licensed under the [MIT License](https://chat.openai.com/c/LICENSE).
    
