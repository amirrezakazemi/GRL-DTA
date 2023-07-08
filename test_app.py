import requests

if __name__ == '__main__':
    protein = "SNAPRLIRPYMEP"
    drug_smiles = "CC(C)(C)C1=CC(=NO1)NC(=O)NC2=CC=C(C=C2)C3=CN4C5=C(C=C(C=C5)OCCN6CCOCC6)SC4=N3"
    data = {
        "protein": protein,
        "drug_smiles": drug_smiles
    }
    url = 'http://localhost:9696/app'
    resp = requests.post(url=url, json=data)
    print(resp.json())