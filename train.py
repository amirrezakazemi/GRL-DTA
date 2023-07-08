import torch
import torch.nn as nn
from model import GCN
from data_prep import DTADataset
import optuna
import mlflow
import boto3
import pytorch_lightning as pl
import mlflow.pytorch
import os

# Set the tracking URI (MLflow server address)
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "drug-target-affinitiy-exp"

mlflow.set_tracking_uri(TRACKING_URI)
# Set the active experiment
mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.pytorch.autolog()

def create_dir(directory_path):
    """
    Create a directory if it does not already exist.
    
    Args:
        directory_path (str): The path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")



def test(loader, model):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for drug, target, y in loader:
            output = model(drug.to(device), target.to(device))
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()
    
        

def objective(trial, train_loader, test_loader):
    with mlflow.start_run():
        drug_conv_hidden_dims_cat = [
                str([78, 78*2, 78*4]),
                str([78, 78*2, 78*3]),
                str([78, 78, 78*2]),
                str([78, 78*2]),
                str([78, 78*4])
        ]

        drug_fc_hidden_dims_cat = [
            str([256]), str([512]), str([1024])
        ]

        rep_dim_cat = [
            str(64), str(128), str(512)
        ]

        comb_fc_hidden_dims_cat = [
            str([1024, 512]),
            str([512, 256]),
            str([256, 128])
        ]
        
        LR = trial.suggest_float('LR', 1e-6, 1e-5)

        params = {
            'drug_conv_hidden_dims': eval(trial.suggest_categorical('drug_conv_hidden_dims', drug_conv_hidden_dims_cat)),
            'drug_fc_hidden_dims': eval(trial.suggest_categorical('drug_fc_hidden_dims', drug_fc_hidden_dims_cat)),
            'rep_dim': eval(trial.suggest_categorical('rep_dim', rep_dim_cat)),
            'comb_fc_hidden_dims' : eval(trial.suggest_categorical('comb_fc_hidden_dims', comb_fc_hidden_dims_cat)),
            'lr': LR
        }

        model = GCN(**params)
        # logger = pl.loggers.MLFlowLogger(experiment_name=EXPERIMENT_NAME)
        # logger.log_hyperparams(params)  # Log hyperparameters if needed



        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(model, train_loader, test_loader)
        callbacks = trainer.callback_metrics
        print(callbacks)
        mlflow.log_params(params)
        val_loss = callbacks['val_loss']

    return val_loss

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = torch.load('data/kiba_train_dl.pth')
    test_loader = torch.load('data/kiba_test_dl.pth')

    # Create and run the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, test_loader), n_trials=1)






