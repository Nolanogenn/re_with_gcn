import json
from datetime import datetime

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, RGCNConv
import torch.nn.functional as F

from pyg_dataset import REDataset

BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RGCNEncoder(torch.nn.Module):
    def __init__(self, feature_channels, hidden_channels, num_relations):
        super().__init__()
        # todo try FastRGCNConv as well, as we have GPUs with each 48GB VRAM
        self.conv1 = RGCNConv(feature_channels, hidden_channels, num_relations,
                              num_blocks=8)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()

        self.linear = torch.nn.Linear(hidden_channels * 2, num_relations)

    def forward(self, z, edge_index):
        z_src, z_dst = z[edge_index.T[0]], z[edge_index.T[2]]
        concat = torch.cat((z_src, z_dst), 1)
        x = self.linear(concat)
        return torch.nn.ReLU()(x)


def score(y_true, y_pred):
    return {
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        #'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        #'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        #'recall_macro': recall_score(y_true, y_pred, average='macro'),
    }


if __name__ == '__main__':
    print('start training')

    dataset = REDataset('./data/trex', debug=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = GAE(
        RGCNEncoder(feature_channels=768,
                    hidden_channels=200,
                    num_relations=dataset.num_relations),
        Decoder(dataset.num_classes,
                hidden_channels=200),
    )

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_function = nn.CrossEntropyLoss()
    model.train()

    loss_values = []
    score_values = []
    for epoch in range(0, 1):
        for batch in dataloader:
            optimizer.zero_grad()

            # the nodes indices in the batch are changes, therefore, adapt edge indices of the labeled data
            batch_node_idx = batch.y[:, [0, 2]] + batch.ptr[:-1].repeat(2, 1).T

            batch.to(device=DEVICE)

            z = model.encode(batch.x, batch.edge_index, batch.edge_type)
            y_pred = model.decode(z, batch.y)

            cross_entropy_loss = loss_function(y_pred, batch.y.T[1])
            print(cross_entropy_loss.item())
            loss_values.append(cross_entropy_loss.item())

            scores = score(y_true=batch.y.T[1], y_pred=torch.softmax(y_pred, dim=1).argmax(dim=1))
            score_values.append(scores)

            if torch.is_tensor(cross_entropy_loss) and cross_entropy_loss.requires_grad:
                cross_entropy_loss.backward()
                optimizer.step()

    with open(f'logs_{datetime.now().strftime("%Y-%m-%d")}.log', 'w') as logs_out:
        json.dump({'loss': loss_values,
                   'scores': score_values}, logs_out, indent=4)
