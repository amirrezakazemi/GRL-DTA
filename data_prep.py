import pandas as pd
import numpy as np
import os
import json,pickle
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from torch_geometric.data import Data, Batch
import torch
   
# Custom dataset class
class DTADataset(Dataset):
    def __init__(self, samples):
        self.samples = self.process_samples(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        drug, target, y = self.samples[index]
        return drug, target, y

    def collate_fn(self, batch):
        drugs, targets, ys = zip(*batch)
        drug_batch = Batch.from_data_list(drugs)
        target_batch = torch.tensor(np.stack(targets))
        y_batch = torch.tensor(ys, dtype=torch.float)
        return drug_batch, target_batch, y_batch

    def process_samples(self, samples):
        processed_samples = []
        for i in range(len(samples)):
            print(f'{i}/{len(samples)}')
            drug, target, y = samples[i]
            processed_samples.append((preprocess_drug(drug), preprocess_target(target), y))
        return processed_samples

    def create_dataloader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def node_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 
                                                'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I',
                                                'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag',
                                                'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge',
                                                'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr',
                                                'Pt', 'Hg', 'Pb', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
        [atom.GetIsAromatic()]
    )


def preprocess_drug(drug):
    
    mol = Chem.MolFromSmiles(drug)
    atom_nums = mol.GetNumAtoms()
    
    atom_features = []
    for atom in mol.GetAtoms():
        atom_feature = node_features(atom)
        atom_features.append(atom_feature/sum(atom_feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    graph = nx.Graph(edges).to_directed()

    edge_index = []
    for e1, e2 in graph.edges:
        edge_index.append([e1, e2])
        
    graph = Data(x=torch.tensor(np.array(atom_features), dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                    num_nodes=atom_nums)
    return graph

def preprocess_target(target):
    ### Protein 
    tar_vocab = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    tar_dict = {v:(i+1) for i,v in enumerate(tar_vocab)}
    max_tar_len = 1000

    x = np.zeros(max_tar_len)
    for i, ch in enumerate(target[:max_tar_len]): 
        x[i] = tar_dict[ch]
    return x


def read_data(dataset, fold_type, data_dir="./data"):

    fdir = f'{data_dir}/{dataset}/'

    with open(os.path.join(fdir, f'folds/{fold_type}_fold_setting1.txt')) as f_f_in:
        fold = json.load(f_f_in)
        np_fold = np.array(fold)
        if np_fold.ndim > 1:
            np_fold = np_fold.reshape(-1, )

    with open(os.path.join(fdir, 'ligands_can.txt')) as f_l_in:
        ligands = json.load(f_l_in)
    
    with open(os.path.join(fdir, 'proteins.txt')) as f_p_in:
        proteins = json.load(f_p_in)

    with open(os.path.join(fdir, 'Y'), 'rb') as f_a_in:
        affinity = pickle.load(f_a_in, encoding='latin1')
    
    return list(np_fold), ligands, proteins, affinity


def load_dataloader(dataset="kiba", fold_type="train", data_dir="./data"):

    print(f'Loading dataloader for {dataset} dataset in {fold_type} fold')
    
    print("Reading data ...")
    fold, drug, target, affinity = read_data(dataset, fold_type, data_dir)
    print("Finished reading data.")

    print("Creating DTA dataset and dataloader ...")
    drug_vals = list(drug.values())
    target_vals = list(target.values())

    drug_idx, tar_idx = np.where(np.isnan(affinity)==False)
    fold_drug_idx, fold_tar_idx = drug_idx[fold], tar_idx[fold]
    data = []
    for i in range(len(fold_tar_idx)):
        sample = [
            drug_vals[fold_drug_idx[i]],
            target_vals[fold_tar_idx[i]],
            affinity[fold_drug_idx[i], fold_tar_idx[i]]
        ]
        data.append(sample)
    Dataset = DTADataset(data)
    data_loader = Dataset.create_dataloader()

    torch.save(data_loader, os.path.join(data_dir, f'{dataset}_{fold_type}_dl.pth'))
    print("Saved dataloader.")


if __name__ == '__main__':
    load_dataloader(dataset="kiba", fold_type="test")
    load_dataloader(dataset="kiba", fold_type="train")

    


    
    



   


