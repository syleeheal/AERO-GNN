import numpy as np
import pdb
import os

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch_sparse import SparseTensor, matmul

import torch_geometric
from torch_geometric.nn import GCN2Conv, GCNConv, GATConv, GATv2Conv, FAConv, TransformerConv
from torch_geometric.nn import aggr, inits
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones, zeros
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import  OptTensor
from torch_scatter import scatter_add

from layers import GATv2_Conv, FA_Conv, MixHopConv, AntiSymmetricConv



class AERO_GNN_Model(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super().__init__(node_dim=0, )

        self.args = args

        self.num_nodes = graph.x.size(0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = self.args.num_heads
        self.hid_channels = hid_channels
        self.hid_channels_ = self.heads * self.hid_channels
        self.K = self.args.iterations
                
        self.setup_layers()
        self.reset_parameters()


    def setup_layers(self):

        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        self.dense_lins = nn.ModuleList()
        self.atts = nn.ParameterList()
        self.hop_atts = nn.ParameterList()
        self.hop_biases = nn.ParameterList()
        self.decay_weights = []

        self.dense_lins.append(Linear(self.in_channels, self.hid_channels_, bias=True, weight_initializer='glorot'))
        for _ in range(self.args.num_layers - 1): self.dense_lins.append(Linear(self.hid_channels_, self.hid_channels_, bias=True, weight_initializer='glorot'))
        self.dense_lins.append(Linear(self.hid_channels_, self.out_channels, bias=True, weight_initializer='glorot'))

        for k in range(self.K + 1): 
            self.atts.append(nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels)))
            self.hop_atts.append(nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels*2)))
            self.hop_biases.append(nn.Parameter(torch.Tensor(1, self.heads)))
            self.decay_weights.append( np.log((self.args.lambd / (k+1)) + (1 + 1e-6)) )
        self.hop_atts[0]=nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels))
        self.atts = self.atts[1:]


    def reset_parameters(self):
        
        for lin in self.dense_lins: lin.reset_parameters()
        for att in self.atts: glorot(att) 
        for att in self.hop_atts: glorot(att) 
        for bias in self.hop_biases: ones(bias) 


    def hid_feat_init(self, x):
        
        x = self.dropout(x)
        x = self.dense_lins[0](x)

        for l in range(self.args.num_layers - 1):
            x = self.elu(x)
            x = self.dropout(x)
            x = self.dense_lins[l+1](x)
        
        return x


    def aero_propagate(self, h, edge_index):
        
        self.k = 0
        h = h.view(-1, self.heads, self.hid_channels)
        g = self.hop_att_pred(h, z_scale=None)
        z = h * g
        z_scale = z * self.decay_weights[self.k]

        for k in range(self.K):

            self.k = k+1
            h = self.propagate(edge_index, x = h, z_scale = z_scale)            
            g = self.hop_att_pred(h, z_scale)
            z += h * g
            z_scale = z * self.decay_weights[self.k]
                
        return z


    def node_classifier(self, z):
        
        z = z.view(-1, self.heads * self.hid_channels)
        z = self.elu(z)
        if self.args.add_dropout == True: z = self.dropout(z)
        z = self.dense_lins[-1](z)
        
        return z


    def forward(self, x, edge_index):
        
        h0 = self.hid_feat_init(x)
        z_k_max = self.aero_propagate(h0, edge_index)
        z_star =  self.node_classifier(z_k_max)

        return z_star


    def hop_att_pred(self, h, z_scale):

        if z_scale is None: 
            x = h
        else:
            x = torch.cat((h, z_scale), dim=-1)

        g = x.view(-1, self.heads, int(x.shape[-1]))
        g = self.elu(g)
        g = (self.hop_atts[self.k] * g).sum(dim=-1) + self.hop_biases[self.k]
        
        return g.unsqueeze(-1)


    def edge_att_pred(self, z_scale_i, z_scale_j, edge_index):
        
        # edge attention (alpha_check_ij)
        a_ij = z_scale_i + z_scale_j
        a_ij = self.elu(a_ij)
        a_ij = (self.atts[self.k-1] * a_ij).sum(dim=-1)
        a_ij = self.softplus(a_ij) + 1e-6

        # symmetric normalization (alpha_ij)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(a_ij, col, dim=0, dim_size=self.num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        a_ij = deg_inv_sqrt[row] * a_ij * deg_inv_sqrt[col]        

        return a_ij


    def message(self, edge_index, x_j, z_scale_i, z_scale_j):
        a = self.edge_att_pred(z_scale_i, z_scale_j, edge_index)
        return a.unsqueeze(-1) * x_j


class APPNP_Model(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super().__init__()

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_nodes = graph.x.size(0)

        self.K = self.args.iterations
        self.alpha = self.args.alpha
        
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):

        self.dropout = nn.Dropout(self.args.dropout)
        
        self.relu = nn.ReLU()
        
        self.linear_node_1 = Linear(self.in_channels, self.hid_channels, bias=True, weight_initializer='glorot')
        self.linear_node_2 = Linear(self.hid_channels, self.out_channels, bias=True,weight_initializer='glorot')


    def reset_parameters(self):

        self.linear_node_1.reset_parameters()
        self.linear_node_2.reset_parameters()

    def node_label_pred(self, x):
        
        x = self.dropout(x)

        h = self.linear_node_1(x)
        h = self.relu(h)
        h = self.dropout(h)
                
        h = self.linear_node_2(h)

        return h

    def ppr_propagate(self, a, h, edge_index):

        z = h

        for k in range(self.K):
            a_drop = self.dropout(a)

            z = self.propagate(edge_index = edge_index, 
                                x=z, 
                                edge_weight = a_drop,
                                )

            z = z * (1-self.alpha)
            z += h * self.alpha

        return z
        
    def forward(self, x, edge_index):
        
        h = self.node_label_pred(x)
        edge_index, a = gcn_norm(edge_index, num_nodes = self.num_nodes, add_self_loops=False)
        z = self.ppr_propagate(a, h, edge_index)

        return z

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j * edge_weight.view(-1, 1)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

class GPR_GNN_Model(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super(GPR_GNN_Model, self).__init__(aggr='add')

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_nodes = graph.x.size(0)

        self.K = self.args.iterations
        self.alpha = self.args.alpha

        TEMP = self.alpha*(1-self.alpha)**np.arange(self.K+1)
        TEMP[-1] = (1-self.alpha)**self.K
        self.temp = nn.Parameter(torch.tensor(TEMP))

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):

        self.dropout = nn.Dropout(self.args.dropout)
        self.relu = nn.ReLU()
        
        self.linear_node_1 = Linear(self.in_channels, self.hid_channels, bias=True, weight_initializer='glorot')
        self.linear_node_2 = Linear(self.hid_channels, self.out_channels, bias=True,weight_initializer='glorot')


    def reset_parameters(self):

        self.linear_node_1.reset_parameters()
        self.linear_node_2.reset_parameters()

        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def node_label_pred(self, x):
        
        x = self.dropout(x)
        
        h = self.linear_node_1(x)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.linear_node_2(h)

        return h

    def gpr_propagate(self, a, h, edge_idx):
        
        z = h*(self.temp[0])

        for k in range(self.K):

            h = self.propagate(edge_idx, x=h, norm = a)

            gamma = self.temp[k+1]
            z = z + gamma*h

        return z
        
    def forward(self, x, edge_idx):
        
        h = self.node_label_pred(x)
        edge_idx, a = gcn_norm(edge_idx, num_nodes = self.num_nodes, add_self_loops=False)
        z = self.gpr_propagate(a, h, edge_idx)
        
        return z

    def message(self, x_j, norm):
        return x_j * norm.view(-1, 1)

class GCNII_Model(MessagePassing):
    
    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super().__init__()

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_nodes = graph.x.size(0)

        self.setup_hyperparameters()
        self.setup_layers()

    def setup_hyperparameters(self):
        self.alpha = self.args.alpha
        self.theta = self.args.lambd
        self.num_layer = self.args.num_layers

    def setup_layers(self):

        self.dropout = nn.Dropout(self.args.dropout)
        self.relu = nn.ReLU()

        self.lins = nn.ModuleList()
        self.lins.append(Linear(self.in_channels, self.hid_channels))
        self.lins.append(Linear(self.hid_channels, self.out_channels))
        
        self.convs = nn.ModuleList()

        for layer in range(self.num_layer):
            self.convs.append(
                                GCN2Conv(channels = self.hid_channels,
                                alpha = self.alpha,
                                theta = self.theta,
                                layer = layer + 1,
                                shared_weights = True,
                                normalize = False,
                                )
            )
        
    def forward(self, x, edge_index):

        edge_index, edge_weight = gcn_norm(edge_index, num_nodes = self.num_nodes, add_self_loops=False, dtype=x.dtype)
        
        x = self.dropout(x)
        x = x_0 = self.lins[0](x).relu()
        
        for conv in self.convs:
            x = self.dropout(x)
            x = conv(x, x_0, edge_index, edge_weight)
            x = x.relu()

        if self.args.add_dropout: x = self.dropout(x)
        x = self.lins[1](x)

        return x

class GCN_Model(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super().__init__()

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_nodes = graph.x.size(0)

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):

        self.convs = nn.ModuleList()

        self.convs.append(
            GCNConv(self.in_channels, self.hid_channels, cached=True, normalize=True))

        self.convs.append(
            GCNConv(self.hid_channels, self.out_channels, cached=True, normalize=True))

        self.dropout = nn.Dropout(self.args.dropout)
        self.relu = F.relu

    def reset_parameters(self):
        for conv in self.convs: conv.reset_parameters()

    def forward(self, x, edge_index):
        
        x = self.dropout(x)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)

        return x

class GAT_v2_Model(nn.Module):
    
    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super(GAT_v2_Model, self).__init__()
        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_heads = self.args.num_heads

        self.setup_layers()
        self.reset_parameters()
        
    def setup_layers(self):

        self.convs = nn.ModuleList()
        for i in range(self.args.num_layers):
            self.convs.append(
                GATv2_Conv(self.hid_channels * self.num_heads, self.hid_channels,
                            heads = self.num_heads,
                            concat = True,
                            negative_slope=0.2,
                            dropout = self.args.dropout,
                            add_self_loops = False,
                            share_weights = True,
                            )
                )

        self.convs[0] = GATv2_Conv(self.in_channels, self.hid_channels,
                            heads = self.num_heads,
                            concat = True,
                            negative_slope=0.2,
                            dropout = self.args.dropout,
                            add_self_loops = False,
                            share_weights = True,
                            )

        self.convs[-1] = GATv2_Conv(self.hid_channels * self.num_heads, self.out_channels,
                            heads = self.num_heads,
                            concat = False,
                            negative_slope=0.2,
                            dropout = self.args.dropout,
                            add_self_loops = False,
                            share_weights = True,
                            )


        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = F.elu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        

        x = self.dropout(x)
        for i in range(self.args.num_layers - 1):
            x, alpha_unnorm, alpha_norm, x_lin = self.convs[i](x, edge_index, return_attention_weights = True)
            x = self.elu(x)
            x = self.dropout(x)

        x, alpha_unnorm, alpha_norm, x_lin = self.convs[-1](x, edge_index, return_attention_weights = True)
                
        return x

class GAT_Model(nn.Module):
    
    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super(GAT_Model, self).__init__()
        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_heads = self.args.num_heads

        self.setup_layers()
        self.reset_parameters()
        
    def setup_layers(self):

        self.convs = nn.ModuleList()
        for i in range(self.args.num_layers):
            self.convs.append(
                GATConv(self.hid_channels * self.num_heads, self.hid_channels,
                            heads = self.num_heads,
                            concat = True,
                            negative_slope=0.2,
                            dropout = self.args.dropout,
                            add_self_loops = False,
                            share_weights = True,
                            )
                )

        self.convs[0] = GATConv(self.in_channels, self.hid_channels,
                            heads = self.num_heads,
                            concat = True,
                            negative_slope=0.2,
                            dropout = self.args.dropout,
                            add_self_loops = False,
                            share_weights = True,
                            )

        self.convs[-1] = GATConv(self.hid_channels * self.num_heads, self.out_channels,
                            heads = self.num_heads,
                            concat = False,
                            negative_slope=0.2,
                            dropout = self.args.dropout,
                            add_self_loops = False,
                            share_weights = True,
                            )

        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = F.elu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        
        x = self.dropout(x)
        for i in range(self.args.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.elu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)

        return x

class GAT_v2_Res_Model(nn.Module):
    
    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super(GAT_v2_Res_Model, self).__init__()
        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_heads = self.args.num_heads
        self.alpha = self.args.alpha

        self.setup_layers()
        self.reset_parameters()
        
    def setup_layers(self):


        self.convs = nn.ModuleList()
        for i in range(self.args.num_layers):
            self.convs.append(
                GATv2_Conv(self.hid_channels * self.num_heads, self.hid_channels,
                            heads = self.num_heads,
                            concat = True,
                            negative_slope=0.2,
                            dropout = 0,
                            add_self_loops = False,
                            share_weights = True,
                            )
                )

        self.convs[0] = GATv2_Conv(self.in_channels, self.hid_channels,
                            heads = self.num_heads,
                            concat = True,
                            negative_slope=0.2,
                            dropout = 0,
                            add_self_loops = False,
                            share_weights = True,
                            )

        self.convs[-1] = GATv2_Conv(self.hid_channels * self.num_heads, self.out_channels,
                            heads = self.num_heads,
                            concat = False,
                            negative_slope=0.2,
                            dropout = 0,
                            add_self_loops = False,
                            share_weights = True,
                            )

        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = F.elu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, x, edge_index):
        
        x = self.dropout(x)
        for i in range(self.args.num_layers - 1):
            x, alpha_unnorm, alpha_norm, x_lin = self.convs[i](x, edge_index, return_attention_weights = True)        
            if i == 0: x0 = x_lin.view(-1, self.hid_channels * self.num_heads)
            x = x * (1 - self.alpha)  +  x0 * self.alpha            
            x = self.elu(x)
            x = self.dropout(x)
        x, alpha_unnorm, alpha_norm, x_lin = self.convs[-1](x, edge_index, return_attention_weights = True)
        
        return x

class DAGNN_Model(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super(DAGNN_Model, self).__init__(aggr='add')

        self.args = args

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels

        self.num_nodes = graph.x.size(0)

        self.K = self.args.iterations
        
        self.setup_layers()
        self.reset_parameters()


    def setup_layers(self):
        self.lin1 = Linear(self.in_channels, self.hid_channels)
        self.lin2 = Linear(self.hid_channels, self.out_channels)
        self.proj = Linear(self.out_channels, 1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.proj.reset_parameters()

    def prop(self, x, edge_index):
        edge_index, norm = gcn_norm(edge_index, num_nodes = self.num_nodes, add_self_loops=False)
        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
           
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()

        return out

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

class FAGCN_Model(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super(FAGCN_Model, self).__init__(aggr='add')

        self.args = args

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels

        self.num_nodes = graph.x.size(0)

        self.K = self.args.iterations
        self.eps = self.args.alpha
        
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):

        self.lin1 = Linear(self.in_channels, self.hid_channels)
        self.lin2 = Linear(self.hid_channels, self.out_channels)

        self.convs = nn.ModuleList()
        for i in range(self.K):
            self.convs.append(FA_Conv(
                                    channels = self.hid_channels,
                                    dropout = self.args.dropout,
                                    eps = self.eps,
                                    cached = True,
                                    add_self_loops = False,
                                    normalize = True,
                                    return_attention_weights = True,)
                                    )

        self.dropout = nn.Dropout(self.args.dropout)
        self.relu = nn.ReLU()
        
    def reset_parameters(self):
        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for conv in self.convs: conv.reset_parameters()

    def forward(self, x, edge_index):
        
        x = self.dropout(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x_0 = x
        for i in range(self.K):
            x, alpha_unnorm, alpha_norm = self.convs[i](x, x_0, edge_index)        
        x = self.lin2(x)
    
        return x

class GT_Model(nn.Module):
    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super(GT_Model, self).__init__()
        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_heads = self.args.num_heads

        self.setup_layers()
        self.reset_parameters()
        

    def setup_layers(self):

        self.convs = nn.ModuleList()
        for i in range(self.args.num_layers):
            self.convs.append(
                TransformerConv(self.hid_channels * self.num_heads, self.hid_channels,
                            heads = self.num_heads,
                            concat = True,
                            dropout = self.args.dropout,
                            beta = False,
                            )
                )

        self.convs[0] = TransformerConv(self.in_channels, self.hid_channels,
                            heads = self.num_heads,
                            concat = True,
                            dropout = self.args.dropout,
                            beta = False,
                            )

        self.convs[-1] = TransformerConv(self.hid_channels * self.num_heads, self.out_channels,
                            heads = self.num_heads,
                            concat = False,
                            dropout = self.args.dropout,
                            beta = False,
                            )

        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = F.elu


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, edge_index):
        
        x = self.dropout(x)
        for i in range(self.args.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.elu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)

        return x

class MixHop_Model(nn.Module):
    """ 
    implemetation of MixHop from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/models.py with minor changes
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):

        super(MixHop_Model, self).__init__()

        self.args = args
        
        num_layers = 2
        hops = self.args.iterations

        self.convs = nn.ModuleList()
        self.convs.append(MixHopConv(in_channels, hid_channels, hops=hops))

        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopConv(hid_channels*(hops+1), hid_channels, hops=hops))

        self.convs.append(
            MixHopConv(hid_channels*(hops+1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(out_channels*(hops+1), out_channels)

        self.dropout = self.args.dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, x, edge_index):
        n = x.size(0)
        edge_weight = None
        
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index

            adj_t = torch.sparse_coo_tensor(indices=torch.stack([col, row], dim=0), values=edge_weight, size=(n, n))
        
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm( edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        x = self.final_project(x)
        return x

class ADGN_Model(MessagePassing):
    
    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super().__init__()

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_nodes = graph.x.size(0)

        self.setup_hyperparameters()
        self.setup_layers()

    def setup_hyperparameters(self):
        self.alpha = self.args.alpha
        self.gamma = self.args.lambd
        self.K = self.args.iterations

    def setup_layers(self):

        self.dropout = nn.Dropout(self.args.dropout)

        self.lins = nn.ModuleList()
        self.lins.append(Linear(self.in_channels, self.hid_channels))
        self.lins.append(Linear(self.hid_channels, self.out_channels))
        
        self.conv = AntiSymmetricConv(in_channels = self.hid_channels,
                                phi = None,
                                num_iters = self.K,
                                epsilon = self.alpha,
                                gamma = self.gamma,
                                act = 'tanh',
                                )
        
        
    def forward(self, x, edge_index):
        
        x = self.dropout(x)
        x = self.lins[0](x)
        x = self.dropout(x)
        x = self.conv(x, edge_index)
        x = self.dropout(x)

        return x


