import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import BCELoss
from torch.optim import Adam

from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from information_decomposition.utils import MLP
from copy import deepcopy

import numpy as np
import pandas as pd

class ClassificationDataset(Dataset):
    """Custom dataset for classification tasks."""
    def __init__(self, x, label, ipw):
        self.x = x
        self.label = label
        self.ipw = ipw

    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return {'x': self.x.values[idx].astype(np.float64), 
                'label': np.array([1-self.label.values[idx], self.label.values[idx]]).astype(np.float64), 
                'ipw': self.ipw.values[idx].astype(np.float64) if not(self.ipw is None) else np.float64(1.),
                'idx': idx}
    
def train_mlp_and_get_prediction_probabilities(X_train, y_train, X_val, y_val, X, sample_weight = None, weight_val = None,
                                              grid_search = {}, clip = False):
    """
    Train a multi-layer perceptron (MLP) and return prediction probabilities.
    Parameters:
    - X_train: Training features
    - y_train: Training labels 
    - X_val: Validation features
    - y_val: Validation labels
    - X: Features for prediction
    - sample_weight: Sample weights for training
    - weight_val: Sample weights for validation
    - grid_search: Hyperparameters for grid search
    - clip: Whether to clip the probabilities between 0.1 and 0.9
    Returns:
    - pd.Series: Prediction probabilities
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = ClassificationDataset(x=X_train, label=y_train, ipw=sample_weight)
    val_dataset = ClassificationDataset(x=X_val, label=y_val, ipw=weight_val)

    best_loss_overall = float('inf')
    for param in ParameterSampler(grid_search, n_iter=5, random_state=42):
        best_loss, patience = float('inf'), 0

        layers = param.pop('layers', [])
        lr = param.pop('lr', 0.001)
        epochs = param.pop('epochs', 100)
        bs = param.pop('batch_size', 1024)

        train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, drop_last=True)

        mlp = MLP(in_dims = X_train.shape[1], n_classes = 2, layers = layers).double().to(device)
        optimizer = Adam(mlp.parameters(), lr = lr)

        for epoch in range(epochs):
            train_losses, val_losses = [], []
            mlp.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                x_batch = batch['x'].to(device)
                label_batch = batch['label'].to(device)
                ipw_batch = batch['ipw'].unsqueeze(-1).to(device)

                output = mlp.predict_proba(x_batch)
                loss = BCELoss(weight=ipw_batch)(output, label_batch)

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            mlp.eval()
            for batch in val_dataloader:
                x_batch = batch['x'].to(device)
                label_batch = batch['label'].to(device)
                ipw_batch = batch['ipw'].unsqueeze(-1).to(device)

                output = mlp.predict_proba(x_batch)
                loss_val = BCELoss(weight=ipw_batch)(output, label_batch)
                val_losses.append(loss_val.item())

            loss_val = np.mean(val_losses)
            print(f'EPOCH {epoch + 1}/{epochs}, Training Loss: {np.mean(train_losses):.4f}, Validation Loss: {loss_val:.4f}', end='\r')

            # Early stopping
            if loss_val < best_loss:
                best_loss = loss_val
                if loss_val < best_loss_overall:
                    best_loss_overall = loss_val
                    best_mlp = deepcopy(mlp)
                patience = 0
            elif patience >= 3:
                print(f'Early stopping at epoch {epoch + 1}')
                break
            else:
                patience += 1
    
    if clip:
        return pd.Series(np.clip(best_mlp.predict_proba(torch.tensor(X.values).to(device))[:, 1].detach().cpu().numpy(), 0.1, 0.9), index = X.index)
    else:
        return pd.Series(best_mlp.predict_proba(torch.tensor(X.values).to(device))[:, 1].detach().cpu().numpy(), index = X.index)

def train_logistic_regression_and_get_prediction_probabilities(X_train, y_train, X_val, y_val, X, sample_weight = None, weight_val = None, clip = False) :
    """
    Train a logistic regression model and return prediction probabilities.
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - X: Features for prediction
    - sample_weight: Sample weights for training
    - weight_val: Sample weights for validation
    - clip: Whether to clip the probabilities between 0.1 and 0.9
    Returns:
    - pd.Series: Prediction probabilities
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train.copy())

    best_model, best_score = None, 0
    # Iterate over different values of C
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(C = C, max_iter = 10000).fit(X_norm, y_train, sample_weight = sample_weight)
        score = model.score(scaler.transform(X_val), y_val, sample_weight = weight_val)
        
        if score > best_score:
            print(f'New best model found with C = {C} and score = {score}')
            best_score = score
            best_model = deepcopy(model)

    if clip:
        return pd.Series(np.clip(best_model.predict_proba(scaler.transform(X))[:, 1], 0.1, 0.9), index = X.index)
    else:
        return pd.Series(best_model.predict_proba(scaler.transform(X))[:, 1], index = X.index)

def get_classification_metric_dict(y_true, y_pred, ipw_weights = None):
    """
    Get classification boostrapped metrics for a binary classification task
    Args:
        y_true (pd.Series): True labels
        y_pred (pd.Series): Predicted probabilities
        ipw_weights (pd.Series, optional): Inverse probability weights. Defaults to None.
    Returns:
        dict: Dictionary containing AUROC, AUPRC, and BCE loss
    """
    def uncertainty(metric):
        results = []
        for _ in range(100):
            y_sample = y_true.sample(frac = 1, replace = True)
            results.append(metric(y_sample, y_pred.loc[y_sample.index], sample_weight = None if ipw_weights is None else ipw_weights.loc[y_sample.index]))
        return np.mean(results), np.std(results)
    
    auroc = uncertainty(roc_auc_score)
    auprc = uncertainty(average_precision_score)
    bce_loss = uncertainty(log_loss)
    current_metric_dict = {
                            'auroc': auroc[0],
                            'auprc': auprc[0],
                            'bce_loss': bce_loss[0],
                            'auroc_std': auroc[1],
                            'auprc_std': auprc[1],
                            'bce_loss_std': bce_loss[1]
                          }

    return current_metric_dict