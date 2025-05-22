import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np

class MultimodalDataset(Dataset):
    """
    Dataset for multimodal data with two features x1 and x2, and their corresponding probabilities px1 and px2.
    The dataset can also include labels, conditional probabilities px1_x2, and inverse propensity weights (ipw).
    """
    def __init__(self, x1, x2, px1, px2, label = None, px1_x2 = None, ipw = None):
        # TODO: Update this dataset as it implicitly assume 2D Y
        self.x1 = x1
        self.x2 = x2
        self.px1 = px1
        self.px2 = px2
        self.label = label
        self.px1_x2 = px1_x2
        self.ipw = ipw

    def __len__(self):
        return len(self.x1)
        
    def __getitem__(self, idx):
        results = {
                    'x1': self.x1[idx].astype(np.float64), 
                    'x2': self.x2[idx].astype(np.float64), 
                    'px1': np.array([1-self.px1[idx], self.px1[idx]]).astype(np.float64),
                    'px2': np.array([1-self.px2[idx], self.px2[idx]]).astype(np.float64),
                    'label': np.array([1-self.label[idx], self.label[idx]]).astype(np.float64) if not(self.label is None) else np.array([0., 0.]).astype(np.float64),
                    'px1_x2': np.array([1-self.px1_x2[idx], self.px1_x2[idx]]).astype(np.float64) if not(self.px1_x2 is None) else np.array([0., 0.]).astype(np.float64),
                    'idx': idx
        }
        if self.ipw is None:
            return results
        else:
            return {**results, 
                    'ipw': self.ipw[idx].astype(np.float64)}
    
def create_mlp(in_dims, layers, dropout = 0.):
    """
    Create a multi-layer perceptron (MLP) with the specified input dimensions, hidden layers, and dropout probability.
    Args:
        in_dims (int): Number of input dimensions.
        layers (list): List of integers representing the number of neurons in each hidden layer.
        dropout (float): Dropout probability for regularization. Default is 0 (no dropout).
    Returns:
        nn.Sequential: A sequential model containing the MLP layers.
    """
    modules = []
    prevdim = in_dims

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias = True))
        if dropout > 0:
            modules.append(nn.Dropout(p = dropout))
        modules.append(nn.ReLU())
        prevdim = hidden

    # Critical to not have a last ReLU to allow negative values
    return nn.Sequential(*modules[:-1])

class MLP(nn.Module):
    """
    MLP Module wrapper to match the scikit-learn API.
    """
    def __init__(self, in_dims, layers = [], dropout_prob = 0, n_classes = 1):
        super().__init__()
        self.network = create_mlp(in_dims, layers + [n_classes], dropout = dropout_prob)

    def forward(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        return nn.Softmax(dim = -1)(self.network(x))