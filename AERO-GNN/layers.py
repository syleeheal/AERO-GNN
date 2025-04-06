import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import sparse as sp
from torch import Tensor

from torch_sparse import SparseTensor, set_diag, matmul
from torch_sparse import spmm

from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, kaiming_uniform, ones, zeros
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.typing import (Adj, NoneType, OptPairTensor, OptTensor, Size, PairTensor)


class FA_Conv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]
    _alpha: OptTensor

    def __init__(self, channels: int, eps: float = 0.1, dropout: float = 0.0,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(FA_Conv, self).__init__(**kwargs)

        self.channels = channels
        self.eps = eps
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._alpha = None

        self.att_l = Linear(channels, 1, bias=False)
        self.att_r = Linear(channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.att_l.reset_parameters()
        self.att_r.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, return_attention_weights=True):
        # type: (Tensor, Tensor, Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Tensor, Tensor, SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Tensor, Tensor, Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Tensor, Tensor, SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        if self.normalize:
            if isinstance(edge_index, Tensor):
                assert edge_weight is None
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, None, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                assert not edge_index.has_value()
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, None, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        else:
            if isinstance(edge_index, Tensor):
                assert edge_weight is not None
            elif isinstance(edge_index, SparseTensor):
                assert edge_index.has_value()

        alpha_l = self.att_l(x)
        alpha_r = self.att_r(x)

        # propagate_type: (x: Tensor, alpha: PairTensor, edge_weight: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r),
                             edge_weight=edge_weight, size=None)

        alpha_unnorm = self._alpha_unnorm
        alpha_norm = self._alpha_norm
        self._alpha_unnorm = None
        self._alpha_norm = None

        if self.eps != 0.0:
            out = out + self.eps * x_0
        
        if isinstance(return_attention_weights, bool):
            return out, alpha_unnorm, alpha_norm
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor,
                edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        alpha = (alpha_j + alpha_i).tanh().squeeze(-1)
        self._alpha_unnorm = (alpha_j + alpha_i)
        self._alpha_norm = (alpha * edge_weight)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * (alpha * edge_weight).view(-1, 1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, eps={self.eps})'

    
class MixHopConv(nn.Module):
    """ Our MixHop layer """
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopConv, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops+1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x) ]
        for j in range(1,self.hops+1):
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = torch.sparse.mm(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)



class AntiSymmetricConv(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        phi: Optional[MessagePassing] = None,
        num_iters: int = 1,
        epsilon: float = 0.1,
        gamma: float = 0.1,
        act: Union[str, Callable, None] = 'tanh',
        act_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.act = activation_resolver(act, **(act_kwargs or {}))

        #if phi is None:
        phi = GCNConv(in_channels, in_channels, 
                        bias=False, 
                        cached=True, 
                        normalize=True, 
                        add_self_loops=False)

        self.W = Parameter(torch.Tensor(in_channels, in_channels))
        self.register_buffer('eye', torch.eye(in_channels))
        self.phi = phi
        self.bias = Parameter(torch.Tensor(in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.phi.reset_parameters()
        zeros(self.bias)


    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:

        antisymmetric_W = self.W - self.W.t() - self.gamma * self.eye

        for _ in range(self.num_iters):
            h = self.phi(x, edge_index)
            h = x @ antisymmetric_W.t() + h
            h += self.bias
            h = self.act(h)

            x = x + self.epsilon * h

        return x


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.in_channels}, '
                f'phi={self.phi}, '
                f'num_iters={self.num_iters}, '
                f'epsilon={self.epsilon}, '
                f'gamma={self.gamma})')

class GATv2_Conv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError()
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=None)

        alpha = self._alpha
        alpha_norm = self.alpha_norm
        self._alpha = None
        self.alpha_norm = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, alpha, alpha_norm, x_l

        else:
            return out,

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        self._alpha = alpha
        alpha = softmax(alpha, index, ptr, size_i)

        self.alpha_norm = alpha
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

