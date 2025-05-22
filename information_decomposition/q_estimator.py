import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from .utils import MultimodalDataset, MLP
from sklearn.model_selection import ParameterSampler
from copy import deepcopy

def normalize(tensor, eps = 0):
    """
    Normalize a tensor to sum to 1.
    Args:
        tensor (torch.Tensor): The tensor to normalize.
        eps (float): A small value to avoid division by zero.
    Returns:
        torch.Tensor: The normalized tensor.
    """
    return tensor / torch.clamp(tensor, eps).sum()

def minimization_objective(q_x1_x2_y, eps = 1e-8):
    """
    Compute the three way information.
    Args:
        q_x1_x2_y (torch.Tensor): The joint distribution of x1, x2, and y.
        eps (float): A small value to avoid division by zero.
    Returns:
        torch.Tensor: The minimization objective.
    """
    q_x1_x2 = torch.sum(q_x1_x2_y, dim = (-1), keepdim=True)
    q_y = torch.sum(q_x1_x2_y, dim = (0,1), keepdim=True)

    numerator = torch.log2(torch.clamp(q_x1_x2_y, eps))
    denominator = torch.log2(torch.clamp(q_x1_x2, eps)) + torch.log2(torch.clamp(q_y, eps))

    return (q_x1_x2_y * (numerator - denominator)).sum().clip(0)

def sinkhorn_probs(q_x1_x2, p_x1, p_x2, atol = 1e-4, eps = 1e-8):
    """
    Sinkhorn-Knopp algorithm to compute the joint distribution q_x1_x2 given the marginals p_x1 and p_x2.
    Args:
        q_x1_x2 (torch.Tensor): The joint distribution of x1 and x2.
        p_x1 (torch.Tensor): The true marginal distribution of x1.
        p_x2 (torch.Tensor): The true marginal distribution of x2.
        atol (float): Absolute tolerance for convergence.
        eps (float): A small value to avoid division by zero.
    Returns:
        torch.Tensor: The updated joint distribution q_x1_x2.
        bool: Whether the update converged.
    """
    q_x1 = torch.sum(q_x1_x2, dim = 1)
    q_x2 = torch.sum(q_x1_x2, dim = 0)
    if (torch.all((torch.abs(q_x1 - p_x1) / p_x1) <= atol)) and (torch.all((torch.abs(q_x2 - p_x2) / p_x2) <= atol)):
        # Avoid update if all good
        return q_x1_x2, True

    q_x1_x2 = q_x1_x2 / (torch.clamp(q_x1_x2.sum(dim = 0, keepdim = True), eps))
    q_x1_x2 *= p_x2.unsqueeze(0)
    q_x1 = torch.sum(q_x1_x2, dim = 1)
    if (torch.all((torch.abs(q_x1 - p_x1) / p_x1) <= atol)):
        # Stop if update did not impact the other marginal
        return q_x1_x2, True
    
    q_x1_x2 = q_x1_x2 / (torch.clamp(q_x1_x2.sum(dim = 1, keepdim = True), eps))
    q_x1_x2 *= p_x1.unsqueeze(1)
    q_x2 = torch.sum(q_x1_x2, dim = 0)
    if (torch.all((torch.abs(q_x2 - p_x2) / p_x2) <= atol)):
        return q_x1_x2, True
    
    return q_x1_x2, False

class QBuilder(torch.nn.Module):
    """
    QBuilder is the full model that learns to estimate the joint distribution of two unimodal distributions.
    It uses two unimodal models to estimate the marginal distributions and then applies the Sinkhorn-Knopp algorithm
    to compute the joint distribution.
    """

    def __init__(self, in_dims, layers = [], n_classes = 2, atol=1e-6, eps=1e-8, sinkhorn_iterations=200, device = 'cpu'):
        super(QBuilder, self).__init__()

        self.in_dims = in_dims
        self.atol = atol
        self.eps = eps
        self.sinkhorn_iterations = sinkhorn_iterations
        self.device = device
        
        self.x1_unimodal_model = MLP(in_dims=self.in_dims[0], layers=layers, dropout_prob=0, n_classes=n_classes).double()
        self.x2_unimodal_model = MLP(in_dims=self.in_dims[1], layers=layers, dropout_prob=0, n_classes=n_classes).double()
        self.x1_unimodal_model.to(self.device)
        self.x2_unimodal_model.to(self.device)

    def forward(self, x1, x2, p_y_given_x1, p_y_given_x2, p_x = None):
        # Estimate Q
        q_y_given_x1 = self.x1_unimodal_model(x1)
        q_y_given_x2 = self.x2_unimodal_model(x2)

        # Normalize in log space to avoid vanishing gradients
        q_y_given_x1 = (q_y_given_x1 - q_y_given_x1.mean(dim=0, keepdim=True)) / (q_y_given_x1.std(dim=0, keepdim=True) + self.eps)
        q_y_given_x2 = (q_y_given_x2 - q_y_given_x2.mean(dim=0, keepdim=True)) / (q_y_given_x2.std(dim=0, keepdim=True) + self.eps)
        
        # Efficient outer produce on each dimensions
        Q = normalize(torch.exp(torch.einsum('ah, bh -> abh', q_y_given_x1, q_y_given_x2)), eps = self.eps)

        # If no p_x is provided, use uniform distribution
        if p_x is None:
            p_x = 1. / len(p_y_given_x1)
        
        # Ensure Q is in set Delta
        stop_met = [False] * p_y_given_x1.shape[-1]
        for si in range(self.sinkhorn_iterations):
            # Match joints of the true p
            q_x1_y = p_y_given_x1 * p_x
            q_x2_y = p_y_given_x2 * p_x

            new_Q = []
            for i in range(Q.size(-1)):
                # Unrolled sinkhorn for preserving gradient
                current, stop_met[i] = sinkhorn_probs(Q[:, :, i], q_x1_y[:, i], q_x2_y[:, i], atol=self.atol, eps=self.eps)
                new_Q.append(current)

            new_Q = torch.stack(new_Q, dim = -1).clip(min = self.eps)

            # Enforce normalization over y dimensions
            q_x1_x2 = new_Q.sum(dim=2, keepdims=True)
            q_y_given_x1_x2 = torch.clip(new_Q /q_x1_x2, self.eps, 1)
            new_Q = q_y_given_x1_x2 / q_y_given_x1_x2.sum(dim=2, keepdims=True) * q_x1_x2

            if torch.any(torch.isnan(new_Q)):
                print('WARNING: Q contains nan - Return last best')
                return Q
            
            Q = new_Q
            if all(stop_met):
                print(f'All sinkhorn conditions met at {si}', end='\r')
                return Q

        return Q

class QEstimator():
    """
    Wrapper class for the QBuilder model.
    It handles the training and validation of the model.
    And allows the user to use the model for inference.
    """

    def __init__(self, x1_train, x2_train, x1_val, x2_val,  
                 p_y_given_x1_train, p_y_given_x2_train, p_y_given_x1_val, p_y_given_x2_val, 
                 ipw_train = None, ipw_val= None,
                 grid_search = {}, n_iter = 5, bs=1024, atol=0.001, epochs = 10,
                 sinkhorn_iterations=100, eps = 1e-10, device='cuda'):
        train_dataset = MultimodalDataset(x1=x1_train, x2=x2_train, 
                                          px1=p_y_given_x1_train, px2=p_y_given_x2_train, 
                                          ipw=ipw_train)
        self.val_dataset = MultimodalDataset(x1=x1_val, x2=x2_val, 
                                             px1=p_y_given_x1_val, px2=p_y_given_x2_val, 
                                             ipw=ipw_val)
        self.train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=min(bs, len(self.val_dataset)), shuffle=True, drop_last=True)

        self.in_dims = (x1_train.shape[1], x2_train.shape[1])
        self.epochs = epochs
        self.device = device
        self.eps = eps
        self.atol = atol
        self.sinkhorn_iterations = sinkhorn_iterations
        self.param = ParameterSampler(grid_search, n_iter=n_iter, random_state=42)
        self.train()

    def forward(self, x1, x2, p_y_given_x1, p_y_given_x2, p_x = None):
        return self.q_builder.forward(x1 = x1.to(self.device), x2 = x2.to(self.device), 
                                      p_y_given_x1 = p_y_given_x1.to(self.device), p_y_given_x2 = p_y_given_x2.to(self.device), 
                                      p_x = p_x.to(self.device) if p_x is not None else None)

    def train(self):
        """
        Train the QBuilder model using the provided training and validation datasets.
        The training process includes hyperparameter tuning using grid search.
        """
        # For reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        best_loss_overall = float('inf')
        for param in self.param:
            best_loss, patience = float('inf'), 0

            layers = param.pop('layers', [])
            lr = param.pop('lr', 0.001)

            q_builder = QBuilder(in_dims = self.in_dims,
                                atol = self.atol,
                                eps = self.eps,
                                sinkhorn_iterations = self.sinkhorn_iterations,
                                layers = layers,
                                device = self.device).to(self.device)
            optimizer = Adam(q_builder.parameters(), lr = lr)

            for epoch in range(self.epochs):
                train_losses, val_losses = [], []
                q_builder.train()
                for batch in self.train_dataloader:
                    optimizer.zero_grad()
                    
                    x1_batch = batch['x1'].to(self.device)
                    x2_batch = batch['x2'].to(self.device)
                    
                    p_y_given_x1_batch = batch['px1'].to(self.device)
                    p_y_given_x2_batch = batch['px2'].to(self.device)

                    # Correction
                    p_x_batch = batch['ipw'].to(self.device).unsqueeze(-1) / len(batch['ipw']) if 'ipw' in batch else None

                    Q = q_builder.forward(x1 = x1_batch, x2 = x2_batch, 
                                          p_y_given_x1 = p_y_given_x1_batch, p_y_given_x2 = p_y_given_x2_batch, 
                                          p_x = p_x_batch)
                    loss = minimization_objective(Q, eps = self.eps)

                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())


                q_builder.eval()
                for batch in self.val_dataloader:
                    x1_val = batch['x1'].to(self.device)
                    x2_val = batch['x2'].to(self.device)
                    
                    p_y_given_x1_val = batch['px1'].to(self.device)
                    p_y_given_x2_val = batch['px2'].to(self.device)

                    p_x_batch = batch['ipw'].to(self.device).unsqueeze(-1) / len(batch['ipw']) if 'ipw' in batch else None

                    Q_val = q_builder.forward(x1 = x1_val, x2 = x2_val, 
                                              p_y_given_x1 = p_y_given_x1_val, p_y_given_x2 = p_y_given_x2_val, 
                                              p_x = p_x_batch)
                    loss_val = minimization_objective(Q_val, eps = self.eps)
                    val_losses.append(loss_val.item())

                loss_val = np.mean(val_losses)
                print(f'EPOCH {epoch + 1}/{self.epochs}, Training Loss: {np.mean(train_losses):.4f}, Validation Loss: {loss_val:.4f}')

                minimal_change = (abs(loss_val - best_loss) / best_loss) < 0.01 if not np.isinf(best_loss) else False
                # Early stopping
                if loss_val < best_loss:
                    best_loss = loss_val
                    if loss_val < best_loss_overall:
                        best_loss_overall = loss_val
                        best_builder = deepcopy(q_builder)
                    patience = 0
                
                if patience >= 3:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
                elif loss_val > best_loss or minimal_change: # Less than 1% improvement
                    patience += 1
        self.q_builder = best_builder