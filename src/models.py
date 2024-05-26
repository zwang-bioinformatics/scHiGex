# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import torch
from custom_transformer_conv import TransformerConv
from torch_geometric.nn.norm import LayerNorm
from torch import nn

def create_model(model_name, config):
    if model_name == "GTransCustomModular": return GTransCustomModular(ini_node_dim=config["node_dim"], ini_edge_dim=config["edge_dim"], num_layers=2, num_heads=10, out_channels=[25, 10]).to(config["device"])
    else: raise ValueError(f"Model {model_name} not found")


class GTransCustomModular(torch.nn.Module):
    """
    GTrans Custom Gated Attention
    """
    def __init__(self, ini_node_dim, ini_edge_dim, num_layers, num_heads, out_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms_layer = nn.ModuleList()
        self.edge_dims = [ini_edge_dim + num_heads for i in range(1, num_layers + 1)]
        self.norms_edge = nn.ModuleList()

        in_channels = ini_node_dim
        edge_dim = ini_edge_dim

        for i in range(num_layers):
            self.layers.append(TransformerConv(in_channels=in_channels, out_channels=out_channels[i], heads=num_heads, concat=True, beta=True, beta_attention=True, dropout=0.1, edge_dim=edge_dim))
            self.norms_layer.append(LayerNorm(out_channels[i] * num_heads))
            self.norms_edge.append(LayerNorm(self.edge_dims[i]))
            in_channels = out_channels[i] * num_heads
            edge_dim = self.edge_dims[i]

        self.final_layer = TransformerConv(in_channels=in_channels, out_channels=1, heads=num_heads, concat=False, beta=True, beta_attention=True, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the network
        """
        ini_edge_attr = edge_attr
        for i in range(len(self.layers)):
            x, (edge_index, edge_attr) = self.layers[i](x, edge_index, edge_attr, return_attention_weights=True)
            edge_attr = torch.cat((ini_edge_attr, edge_attr), dim=1)

            x = self.norms_layer[i](x)
            # edge_attr = self.norms_edge[i](edge_attr).relu()

        x, _ = self.final_layer(x, edge_index, edge_attr, return_attention_weights=False)
        return torch.sigmoid(x)