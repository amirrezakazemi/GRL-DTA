from flask import Flask, request, jsonify
import mlflow
import numpy as np
import torch
from data_prep import preprocess_drug, preprocess_target

RUN_ID = '67479ff9fcaf455383c0ca08ee7618d0'



def feature_prep(json):
    protein = json["protein"]
    drug_smiles = json["drug_smiles"]
    target = preprocess_target(protein)
    drug = preprocess_drug(drug_smiles)

    return drug, target


def predict(drug, target):
    logged_model = f's3://aws-large-bucket/DTA-artifacts/1/{RUN_ID}/artifacts/model'
    model = mlflow.pytorch.load_model(logged_model)
    pred = model(torch.tensor(drug.x), torch.tensor(drug.edge_index), drug.batch, torch.tensor(target).unsqueeze(0))
    return pred

application = Flask('dta-predictor')
@application.route('/app', methods=['POST'])
def end_point():
    json = request.get_json()
    drug, target = feature_prep(json)
    pred = predict(drug, target)
    result = {
        "affinity": int(pred[0])
    }
    return jsonify(result)

if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=9696)

