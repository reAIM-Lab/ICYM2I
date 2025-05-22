import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils import *
import numpy as np

####### ENTROPY-BASED METHODS #######
def entropy_from_joint(p, eps = 1e-10):
    return - (p * torch.log2(torch.clamp(p, eps))).sum()

# I(y;x) = H(x) + H(y) - H(y,x)
def two_way_mutual_information_entropy(p_x_y, eps = 1e-10):
    # get marginals
    p_x = torch.sum(p_x_y, dim = 1)
    p_y = torch.sum(p_x_y, dim = 0)

    # get entropy quantities
    H_x = entropy_from_joint(p_x, eps = eps)
    H_y = entropy_from_joint(p_y, eps = eps)
    H_x_y = entropy_from_joint(p_x_y, eps = eps)

    return  H_x + H_y - H_x_y

# I(y:(x1,x2)) = I(y:x1) + I(y:x2|x1)
def three_way_mutual_information_entropy(q_x1_x2_y, eps = 1e-10):

    # get probabilities
    q_y = torch.sum(q_x1_x2_y, dim = (0,1))
    q_x1_x2 = torch.sum(q_x1_x2_y, dim = (2))

    # get entropy quantities
    H_y = entropy_from_joint(q_y, eps = eps)
    H_x1_x2 = entropy_from_joint(q_x1_x2, eps = eps)
    H_y_x_1_x2 = entropy_from_joint(q_x1_x2_y, eps = eps)

    return H_y + H_x1_x2 - H_y_x_1_x2

####### WRAPPER FUNCTIONS #######
def two_way_mutual_information(q_x_y,  eps = 1e-10, return_float = True):
    mi = two_way_mutual_information_entropy(q_x_y, eps = eps)

    if return_float:
        return mi.item()
    else:
        return mi
    
def three_way_mutual_information(q_x1_x2_y, eps = 1e-10, return_float = True):
    mi = three_way_mutual_information_entropy(q_x1_x2_y, eps = eps)

    if return_float:
        return mi.item()
    else:
        return mi

def pid_decomposition(q_x1_x2_y, p_y_given_x1_x2, p_x, p_y, eps = 1e-10):
    mi_p_y_x1_x2 = (p_y_given_x1_x2 * p_x * (torch.log2(torch.clamp(p_y_given_x1_x2, eps)) - torch.log2(p_y))).sum().item()
    
    mi_q_y_x1_x2 = three_way_mutual_information(q_x1_x2_y, eps = eps)
    mi_q_y_x1 = two_way_mutual_information(q_x1_x2_y.sum(1), eps = eps)
    mi_q_y_x2 = two_way_mutual_information(q_x1_x2_y.sum(0), eps = eps)

    # PID decomposition
    # unique1 = Iq(y:x1) - Iq(y:(x1,x2))
    # unique2 = Iq(y:x2) - Iq(y:(x1,x2))
    # shared = Iq(y:(x1,x2)) - unique1 - unique2
    # complementary = Ip(y:(x1,x2)) - Iq(y:x1,x2)
    pids = {}
    pids['unique1'] = max(mi_q_y_x1_x2 - mi_q_y_x2, 0)
    pids['unique2'] = max(mi_q_y_x1_x2 - mi_q_y_x1, 0)
    pids['shared'] = max(mi_q_y_x1_x2 - pids['unique1'] - pids['unique2'], 0)
    pids['complementary'] = max(mi_p_y_x1_x2 - mi_q_y_x1_x2, 0)

    return pids

def pid_decomposition_batched(estimator, x1, x2, p_y_given_x1, p_y_given_x2, p_y_given_x1_x2, p_y, ipw = None, batch_size = 256, eps = 1e-10):
    test_dataset = MultimodalDataset(x1 = x1, x2 = x2, px1 = p_y_given_x1, px2 = p_y_given_x2, 
                                     px1_x2 = p_y_given_x1_x2, label = p_y, ipw = ipw)
    test_dataloader = DataLoader(test_dataset, batch_size = min(batch_size, len(x1)), shuffle = False, drop_last=True)

    results = {}
    for i, batch in enumerate(test_dataloader):
        x1_batch = batch['x1']
        x2_batch = batch['x2']

        # IPW estimate of p_x
        p_x = batch['ipw'].unsqueeze(-1) / len(batch['ipw']) if 'ipw' in batch else torch.ones(batch['x1'].shape[0], 1) / len(batch['x1'])

        p_y_given_x1_batch = batch['px1']
        p_y_given_x2_batch = batch['px2']
        p_y_given_x1_x2_batch = batch['px1_x2']

        # Estimate of p_y in true distribution
        y_batch = batch['label'] * p_x # needs true labels propensity
        p_y = torch.sum(y_batch, axis = 0, keepdim = True)

        # Estimate Q
        q_x1_x2_y = estimator.forward(x1_batch, x2_batch, p_y_given_x1_batch, p_y_given_x2_batch, p_x).to('cpu')
        
        # Decompose PID
        pids = pid_decomposition(q_x1_x2_y, p_y_given_x1_x2_batch, p_x, p_y, eps = eps)
        for key in pids.keys():
            if key not in results:
                results[key] = []
            results[key].append(pids[key])

    # Average PID across batches
    for key in pids.keys():
        results[key + "_std"] = np.std(results[key])
        results[key] = np.mean(results[key])
    return results