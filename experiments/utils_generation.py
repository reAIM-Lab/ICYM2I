
from scipy.special import expit
from scipy.optimize import minimize

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import os

def create_path_and_all_parents(path_to_create):
    """
    Create a path and all its parents if they do not exist
    """
    path_to_create = path_to_create.strip(os.sep)
    path_to_create_split = path_to_create.split(os.sep)

    for i,_ in enumerate(path_to_create_split):
        
        current_path = os.path.join('', *path_to_create_split[:i+1])
        if not os.path.isdir(current_path):
            os.mkdir(current_path)
            print(f'{current_path} created.')

    return None


# Create data
def get_latents(dims = 10, n_samples = 10000):
    """
    Generate latent variables
    Args:
        dims (int): Number of dimensions for each latent variable
        n_samples (int): Number of samples to generate
    Returns:
        dict: Dictionary containing the latent variables
        np.ndarray: Assigned cluster labels
    """
    data, cluster = make_blobs(n_samples, 3 * dims, centers = 10, random_state = 0, center_box=(-1.0, 1.0))
    data = StandardScaler().fit_transform(data)
    latent_dict = {z: data[:, dims*i:dims*(i+1)] for i, z in enumerate(['z1', 'z2', 'zc'])} 

    return latent_dict, cluster

def randomly_flip_elements_in_array(arr, flip_prob = 0.2):
    """
    Randomly flip elements in a binary array with a given probability
    Args:
        arr (np.ndarray): Input binary array
        flip_prob (float): Probability of flipping each element
    Returns:
        np.ndarray: Array with randomly flipped elements
    """
    flip_mask = np.random.binomial(1, flip_prob, size=arr.shape)
    return np.where(flip_mask == 1, 1 - arr, arr)

def get_input_features(latent_dict):
    """
    Get input modalites by concatenating latent variables
    Args:
        latent_dict (dict): Dictionary containing the latent variables
    Returns:
        np.ndarray: Concatenated input features
    """
    x1 = np.concatenate([latent_dict['z1'], latent_dict['zc']], axis = 1)
    x2 = np.concatenate([latent_dict['z2'], latent_dict['zc']], axis = 1)
    return x1, x2

def get_label(latent_dict, cluster, mixture_ratio, flip_prob = 0.2):
    """
    Generated labels based on the mixture ratio of latent variables
    Args:
        latent_dict (dict): Dictionary containing the latent variables
        cluster (np.ndarray): Assigned cluster labels
        mixture_ratio (dict): Mixture ratio for each latent variable
        flip_prob (float): Probability of flipping each element
    Returns:
        np.ndarray: Generated labels
    """
    transform_dim = latent_dict['z1'].shape[1]
    z1_proportion = np.round(transform_dim * mixture_ratio['z1']).astype(int)
    z2_proportion = np.round(transform_dim * mixture_ratio['z2']).astype(int)

    x3 = np.concatenate([latent_dict['z1'][:, :z1_proportion],
                         latent_dict['z2'][:, :z2_proportion],
                         latent_dict['zc'][:, z1_proportion + z2_proportion:]], axis = 1)
    
    coeffs = np.random.rand(10, x3.shape[1])
    y = (expit((x3 * coeffs[cluster] * 2).sum(axis = 1)) >= 0.5).astype(int)
    y = randomly_flip_elements_in_array(y, flip_prob = flip_prob)
    
    return y

def get_train_val_test_indices(n_samples, val_ratio = 0.1, test_ratio = 0.2):
    """
    Get train, validation, and test indices for a dataset
    Args:
        n_samples (int): Number of samples in the dataset
        val_ratio (float): Ratio of samples to use for validation
        test_ratio (float): Ratio of samples to use for testing
    Returns:
        list: List of strings indicating the split for each sample
    """
    all_indices = np.arange(n_samples)
    test_indices = set(np.random.choice(all_indices, size = np.round(n_samples * test_ratio).astype(int), replace = False))
    train_indices = sorted(set(all_indices) - test_indices)
    val_indices = set(np.random.choice(train_indices, size = np.round(len(train_indices) * val_ratio).astype(int), replace = False))
    train_indices = set(train_indices) - val_indices

    assert train_indices & val_indices == set()
    assert train_indices & test_indices == set()
    assert test_indices & val_indices == set()

    train_indices, val_indices, test_indices = sorted(train_indices), sorted(val_indices), sorted(test_indices)

    return ['train' if i in train_indices else 'val' if i in val_indices else 'test' for i in np.arange(n_samples)]

def get_labels_by_ratio(latent_dict, cluster, ratio_dict = {}, flip_prob = 0.2):
    """
    Generate labels based on the mixture ratio of latent variables
    Args:
        latent_dict (dict): Dictionary containing the latent variables
        cluster (np.ndarray): Assigned cluster labels
        ratio_dict (dict): Mixture ratio for each latent variable
        flip_prob (float): Probability of flipping each element
    Returns:
        pd.DataFrame: DataFrame containing the generated labels
    """
    # Data
    x1, x2 = get_input_features(latent_dict)

    # Scenarios
    values = np.arange(0., 1 + 0.1, 0.1)
    ratios = np.round(np.array([(a, b, 1 - a - b) for a in values for b in values if 0 <= 1 - a - b <= 1]), 1)
    ratio_dict = {f'causal_{z1}_{z2}_{zc}':{'z1':z1, 'z2':z2, 'zc':zc}
                        for (z1,z2,zc) in ratios}

    # Create labels
    labels = {}
    for current_setting_name, mixture_ratio in ratio_dict.items():
        labels[current_setting_name] = get_label(latent_dict, cluster, mixture_ratio, flip_prob = flip_prob)

    labels = pd.DataFrame(labels)
    labels['data_split'] = get_train_val_test_indices(x1.shape[0])

    return pd.DataFrame(x1), pd.DataFrame(x2), labels

def generate_missingness(X, other, label, p_m_1_list): 
    """
    Generate missingness based on the given data and labels
    Args:
        X (pd.DataFrame): Input features
        other (pd.DataFrame): Second set of input features
        label (pd.Series): Labels
        p_m_1_list (list): List of missingness rates
    Returns:
        pd.DataFrame: DataFrame containing the generated missingness
    """
    # Set random seed for reproducibility
    np.random.seed(0)

    # To create a shift due to missingness we aim to learn structure in the second modality and outcome given X
    kmeans_assignment = KMeans(n_clusters=100, random_state=0).fit_predict(other)
    target = (kmeans_assignment * label)

    # Train non linear model to predict the target
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, n_jobs=20)
    model.fit(X, target)

    # Use the previous prediction and a linear shift to match a given missingness rate
    results = {}
    for p_m_1 in p_m_1_list:
        def missingness_rate(beta0):
            probs = np.clip(expit((model.predict(X) + model.predict_proba(X).max(axis = 1)) * 5 + beta0), 0.05, 0.95)
            return np.square(np.mean(probs) - p_m_1)

        beta0 = minimize(missingness_rate, np.array([0])).x
        results[p_m_1] = np.clip(expit((model.predict(X) + model.predict_proba(X).max(axis = 1)) * 5 + beta0), 0.05, 0.95)

    return pd.DataFrame(results, index = X.index)