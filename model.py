import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp
import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCN(pl.LightningModule):
    def __init__(self, drug_inp_dim=78,
                drug_conv_hidden_dims=[78, 78*2, 78*4], 
                drug_fc_hidden_dims=[1024],
                rep_dim=128,
                target_inp_dim=26, 
                target_embed_dim=128,
                target_kernel_size=8,
                comb_fc_hidden_dims=[1024, 512],
                output_dim=1, 
                n_filters=32, dropout=0.2, lr=1e-5):

        super(GCN, self).__init__()

        self.lr = lr
        self.output_dim = output_dim
        self.drug_conv_model = nn.Sequential()
        self.drug_fc_model = nn.Sequential()
        self.comb_fc_model = nn.Sequential()

        tmp_dim = drug_inp_dim
        for i, dim in enumerate(drug_conv_hidden_dims):
            self.drug_conv_model.add_module(f'gcconv{i}', GCNConv(tmp_dim, dim))
            self.drug_conv_model.add_module(f'relu{i}', nn.ReLU())
            tmp_dim = dim
        
        for i, dim in enumerate(drug_fc_hidden_dims):
            self.drug_fc_model.add_module(f'linear{i}', nn.Linear(tmp_dim, dim))
            self.drug_fc_model.add_module(f'relu{i}', nn.ReLU())
            self.drug_fc_model.add_module(f'dropout{i}', nn.Dropout(dropout))
            tmp_dim=dim

        self.drug_fc_model.add_module(f'linear{i+1}',nn.Linear(tmp_dim, rep_dim))
        self.drug_fc_model.add_module(f'dropout{i+1}', nn.Dropout(dropout))


        # protein sequence branch (1d conv)
        self.target_embedding = nn.Embedding(target_inp_dim, target_embed_dim)
        self.target_conv1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=target_kernel_size)
        self.target_fc1 = nn.Linear(32*121, rep_dim)

        tmp_dim = 2*rep_dim
        for i, dim in enumerate(comb_fc_hidden_dims):
            self.comb_fc_model.add_module(f'linear{i}', nn.Linear(tmp_dim, dim))
            self.comb_fc_model.add_module(f'relu{i}', nn.ReLU())
            self.comb_fc_model.add_module(f'droput{i}', nn.Dropout())
            tmp_dim = dim
        self.comb_fc_model.add_module(f'linear{i+1}', nn.Linear(tmp_dim, output_dim))


    def forward(self, x, edge_index, batch, target):
        for module_name, module in self.drug_conv_model.named_children():
            if 'relu' in module_name:
                x = module(x)
            else:
                x = module(x, edge_index)
        x = gmp(x, batch)
        x = self.drug_fc_model(x)

        # 1d conv layers
        target = self.target_embedding(target.long())
        target = self.target_conv1(target)
        # flatten
        target = target.view(-1, 32 * 121)
        target = self.target_fc1(target)

        com_input = torch.cat((x, target), 1)
        out = self.comb_fc_model(com_input)
        return out

    def training_step(self, batch, batch_nb):

        drug, target, y = batch
        x, edge_index, b = drug.x, drug.edge_index, drug.batch
        output = self(x.to(device),edge_index.to(device), b, target.to(device))
        loss = nn.MSELoss()(output, y.view(-1, 1).float().to(device))
        self.log('train_loss', loss, batch_size = batch_nb, )
        return loss


    def validation_step(self, batch, batch_nb):
        drug, target, y = batch
        x, edge_index, b = drug.x, drug.edge_index, drug.batch
        output = self(x.to(device),edge_index.to(device), b, target.to(device))
        loss = nn.MSELoss()(output, y.view(-1, 1).float().to(device))
        self.log("val_loss", loss, batch_size = batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


