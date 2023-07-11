import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
import pdb
from tqdm import trange

from models import *
from utils import fixed_split, sparse_split, init_optimizer

class Trainer(object):

    def __init__(self, args, graph):
        
        self.args = args
        
        self.graph = graph
        self.target = self.graph.y

        self.device = torch.device(self.args.device)
        torch.cuda.set_device(self.device)
        
        self.in_channels = self.graph.x.size(1)
        self.hid_channels = self.args.hid_dim
        self.out_channels = int(torch.max(self.target).item() + 1) 
           

    def create_model(self):

        if self.args.model == 'aero': Model = AERO_GNN_Model

        if self.args.model == 'gcn': Model = GCN_Model
        if self.args.model == 'appnp': Model = APPNP_Model

        if self.args.model == 'gcn2': Model = GCNII_Model
        if self.args.model == 'adgn' : Model = ADGN_Model

        if self.args.model == 'gat': Model = GAT_Model
        if self.args.model == 'gatv2': Model = GAT_v2_Model
        if self.args.model == 'gt': Model = GT_Model
        if self.args.model == 'gat-res' : Model = GAT_v2_Res_Model
        if self.args.model == 'fagcn': Model = FAGCN_Model

        if self.args.model == 'gprgnn': Model = GPR_GNN_Model
        if self.args.model == 'dagnn': Model = DAGNN_Model
        if self.args.model == 'mixhop': Model = MixHop_Model

        self.model = Model(self.args,
                            self.in_channels,
                            self.hid_channels,
                            self.out_channels,
                            self.graph,
                            )
                
    def data_split(self):

        """
        train/val/test split
        """
        if self.args.split == 'fixed': split = fixed_split(self.args, self.graph, self.exp)
        if self.args.split == 'sparse': split = sparse_split(self.graph, 0.025, 0.025)
        self.train_nodes, self.validation_nodes, self.test_nodes = split
        

    def transfer_to_gpu(self):

        if self.exp == 0: self.graph = self.graph.to(self.device)
        self.target = self.target.long().squeeze().to(self.device)
        self.model = self.model.to(self.device)
        
        
    def eval(self, index_set):

        self.model.eval()
        
        with torch.no_grad():
            
            prediction = self.model(self.graph.x, self.graph.edge_index)
            logits = F.log_softmax(prediction, dim=1)
            loss = F.nll_loss(logits[index_set], self.target[index_set])
            
            _, pred = logits.max(dim=1)
            correct = pred[index_set].eq(self.target[index_set]).sum().item()
            acc = correct / len(index_set)

            return acc, loss

        
    def do_a_step(self):

        self.model.train()
        self.optimizer.zero_grad()
        prediction  = self.model(self.graph.x, self.graph.edge_index)
        prediction = F.log_softmax(prediction, dim=1)
        self.loss = F.nll_loss(prediction[self.train_nodes], self.target[self.train_nodes])

        if self.args.lambd_l2 > 0: 
            self.loss += sum([p.pow(2).sum() for p in self.model.parameters()]) * self.args.lambd_l2

        self.loss.backward()
        self.optimizer.step()

    def train_neural_network(self):

        self.optimizer = init_optimizer(self.args, self.model)
        self.iterator = trange(self.args.epochs, desc='Validation accuracy: ', leave=False)

        self.step_counter = 0
        self.best_val_acc = 0
        self.best_val_loss = np.inf
        
        for _ in self.iterator:
            
            self.do_a_step()

            val_acc, val_loss = self.eval(self.validation_nodes)
            self.iterator.set_description("Validation accuracy: {:.4f}".format(val_acc))
            
            close = self.acc_step_counter(val_acc)
            if close: break
                

    def acc_step_counter(self, val_acc):
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            self.test_accuracy, _ = self.eval(self.test_nodes)
            self.step_counter = 0
            return False

        else:
            self.step_counter = self.step_counter + 1
            if self.step_counter > self.args.early_stopping_rounds:    
                self.iterator.close()                
                return True
            return False
            
            
    def fit(self):
        
        acc = []
        seeds = torch.load('./seeds_100.pt')

        for _ in range(self.args.exp_num):

            self.exp = _
            self.seed = seeds[_]

            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            self.model = None
            self.optimizer = None
            
            self.create_model()
            self.data_split()
            self.transfer_to_gpu()
            self.train_neural_network()

            acc.append(self.test_accuracy)
            print("Trial {:} Test Accuracy: {:.4f}".format(self.exp, self.test_accuracy))

        self.avg_acc = sum(acc)/len(acc)
        self.std_acc = torch.std(torch.tensor(acc)).item()
        
        print("Model: {}".format(self.args.model))
        print('n trials: {}'.format(self.args.exp_num))
        print('dataset: {}'.format(self.args.dataset))
        print("Mean test accuracy: {:.4f}".format(self.avg_acc), "±", '{:.3f}'.format(self.std_acc))

