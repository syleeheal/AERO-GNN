import os
import numpy as np
import torch
from torch.nn import functional as F
import dgl
from dgl import ops
import pdb


class Filtered_Dataset:
    """
    Code adapted from repository of the paper,
    A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress? (ICLR, 2023),
    https://github.com/yandex-research/heterophilous-graphs
    """

    def __init__(self, name, add_self_loops=False, device='cpu', use_sgc_features=False, use_identity_features=False,
                 use_adjacency_features=False, do_not_use_original_features=False):

        if do_not_use_original_features and not any([use_sgc_features, use_identity_features, use_adjacency_features]):
            raise ValueError('If original node features are not used, at least one of the arguments '
                             'use_sgc_features, use_identity_features, use_adjacency_features should be used.')

        print('Preparing data...')
        data = np.load(os.path.join('./graph-data', f'{name.replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])

        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int)
        graph = dgl.to_bidirected(graph)
        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        node_features = self.augment_node_features(graph=graph,
                                                   node_features=node_features,
                                                   use_sgc_features=use_sgc_features,
                                                   use_identity_features=use_identity_features,
                                                   use_adjacency_features=use_adjacency_features,
                                                   do_not_use_original_features=do_not_use_original_features)

        self.name = name
        self.device = device

        self.graph = graph.to(device)
        self.node_features = node_features.to(device)
        self.labels = labels.to(device)

        self.train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        self.val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        self.test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1]
        self.num_targets = num_targets

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
        self.metric = 'ROC AUC' if num_targets == 1 else 'accuracy'

        # convert to pyg graph
        import torch_geometric
        #import torch
        src_edge = self.graph.edges()[0]
        dst_edge = self.graph.edges()[1]
        self.edge_index = torch.stack([src_edge, dst_edge], dim=0).long()
        self.pyg_graph = torch_geometric.data.Data(x=self.node_features, edge_index=self.edge_index)

        self.pyg_graph.train_mask = torch.zeros((self.pyg_graph.x.shape[0], self.num_data_splits), dtype=torch.bool)
        self.pyg_graph.val_mask = torch.zeros((self.pyg_graph.x.shape[0], self.num_data_splits), dtype=torch.bool)
        self.pyg_graph.test_mask = torch.zeros((self.pyg_graph.x.shape[0], self.num_data_splits), dtype=torch.bool)
        for i in range(self.num_data_splits): self.pyg_graph.train_mask[self.train_idx_list[i], i] = True
        for i in range(self.num_data_splits): self.pyg_graph.val_mask[self.val_idx_list[i], i] = True
        for i in range(self.num_data_splits): self.pyg_graph.test_mask[self.test_idx_list[i], i] = True
        
        self.pyg_graph.y = self.labels.long()



    @property
    def train_idx(self):
        return self.train_idx_list[self.cur_data_split]

    @property
    def val_idx(self):
        return self.val_idx_list[self.cur_data_split]

    @property
    def test_idx(self):
        return self.test_idx_list[self.cur_data_split]

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits


    @staticmethod
    def augment_node_features(graph, node_features, use_sgc_features, use_identity_features, use_adjacency_features,
                              do_not_use_original_features):

        n = graph.num_nodes()
        original_node_features = node_features

        if do_not_use_original_features:
            node_features = torch.tensor([[] for _ in range(n)])

        if use_sgc_features:
            sgc_features = Filtered_Dataset.compute_sgc_features(graph, original_node_features)
            node_features = torch.cat([node_features, sgc_features], axis=1)

        if use_identity_features:
            node_features = torch.cat([node_features, torch.eye(n)], axis=1)

        if use_adjacency_features:
            graph_without_self_loops = dgl.remove_self_loop(graph)
            adj_matrix = graph_without_self_loops.adjacency_matrix().to_dense()
            node_features = torch.cat([node_features, adj_matrix], axis=1)

        return node_features

    @staticmethod
    def compute_sgc_features(graph, node_features, num_props=5):
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        for _ in range(num_props):
            node_features = ops.u_mul_e_sum(graph, node_features, norm_coefs)

        return node_features
