import logging

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import os
import numpy as np
import pandas as pd
from tsl.ops.similarities import geographical_distance

def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return {"r2_score": score, "mae": mae, "max_error": me}


import numpy as np
import xarray as xr
from typing import List, Tuple
from typing import Optional
import numpy as np

def extract_target_variable(
    ds: xr.Dataset,
    target_vars: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts specified target variables from an xarray Dataset, reshapes them for modeling, 
    and returns the resulting array along with a mask indicating non-NaN values.

    Args:
        ds (xr.Dataset): The input xarray Dataset.
        target_vars (List[str]): List of target variable names to extract from the Dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - target (np.ndarray): A numpy array with shape (time, nodes, channels), where:
                - time: The temporal dimension.
                - nodes: Flattened latitude-longitude pairs.
                - channels: The number of target variables.
            - mask (np.ndarray): A numpy array of the same shape as `target`, where:
                - 1 indicates a non-NaN value.
                - 0 indicates a NaN value.
    """
    # Extract the target variables as DataArrays
    data_arrays = [ds[var] for var in target_vars]

    # Flatten latitude and longitude into a single "node" dimension
    data_arrays_flattened = [da.stack(node=('latitude', 'longitude')) for da in data_arrays]

    # Convert each DataArray to a numpy array and add a channel dimension
    target = np.stack([da.to_numpy() for da in data_arrays_flattened], axis=-1)

    # Create a mask where non-NaN values are 1, and NaN values are 0
    mask = np.where(~np.isnan(target), 1, 0)

    return target, mask


import numpy as np
import xarray as xr
from typing import List

def extract_covariates(
    ds: xr.Dataset,
    variables: List[str],
) -> np.ndarray:
    """
    Extracts specified covariates from an xarray Dataset, reshapes them for modeling,
    and returns the resulting array with NaN values replaced by 0.

    Args:
        ds (xr.Dataset): The input xarray Dataset.
        variables (List[str]): List of variable names to extract from the Dataset as covariates.

    Returns:
        np.ndarray: A numpy array with shape (time, nodes, channels), where:
                    - time: The temporal dimension.
                    - nodes: Flattened latitude-longitude pairs.
                    - channels: The number of covariate variables.
    """
    # Extract the covariate variables as DataArrays
    data_arrays = [ds[var] for var in variables]

    # Flatten latitude and longitude into a single "node" dimension
    data_arrays_flattened = [da.stack(node=('latitude', 'longitude')) for da in data_arrays]

    # Convert each DataArray to a numpy array and add a channel dimension
    covariates = np.stack([da.to_numpy() for da in data_arrays_flattened], axis=-1)

    # Replace NaN values with 0 to handle missing data
    covariates = np.nan_to_num(covariates, nan=0.0)

    return covariates


import xarray as xr
import numpy as np
import pandas as pd

def generate_metadata_array(
    ds: xr.Dataset
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generates a metadata DataFrame and numpy array from a given xarray Dataset 
    by extracting unique latitude and longitude combinations and assigning a 
    unique node ID to each combination.

    Args:
        ds (xr.Dataset): Preloaded xarray Dataset containing spatial coordinates.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: 
            - A DataFrame with 'lat', 'lon', and 'node_id' set as the index.
            - A numpy array containing latitude, longitude, and node IDs, with shape (nodes, 3).
    
    Example usage:
        metadata_df, metadata_array = generate_metadata_array(ds=ds)
    """
    # Extract latitude and longitude values
    latitudes = ds.coords['latitude'].values
    longitudes = ds.coords['longitude'].values

    # Create a DataFrame with all combinations of latitude and longitude
    metadata_df = pd.DataFrame({
        'lat': np.repeat(latitudes, len(longitudes)),
        'lon': np.tile(longitudes, len(latitudes))
    })

    # Drop duplicates and reset index
    metadata_df = metadata_df.drop_duplicates().reset_index(drop=True)

    # Add a unique node ID
    metadata_df['node_id'] = metadata_df.index

    # Set node_id as the index
    metadata_df = metadata_df.set_index('node_id')

    # Convert metadata to a numpy array
    metadata_array = metadata_df.reset_index().to_numpy()

    return metadata_df, metadata_array


def generate_metadata(ds: xr.Dataset):
    """
    Generates a metadata numpy array from an xarray Dataset by extracting unique latitude and longitude
    combinations and assigning a unique node ID to each combination.

    Args:
        ds (xr.Dataset): Preloaded xarray Dataset containing spatial coordinates.
        save_directory (str, optional): Directory where the metadata files will be saved. Defaults to 'data/05_model_input/'.

    Returns:
        pd.DataFrame: A metadata DataFrame containing latitude, longitude, and node ID.
        np.ndarray: A numpy array containing metadata with shape (nodes, 3).
    """
    # Ensure the dataset has required coordinates
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError("Dataset must contain 'latitude' and 'longitude' coordinates.")
    
    # Extract unique latitude and longitude values
    latitudes = ds.coords['latitude'].values
    longitudes = ds.coords['longitude'].values
    
    # Create a DataFrame with all unique combinations of latitude and longitude
    metadata_df = pd.DataFrame({
        'lat': np.repeat(latitudes, len(longitudes)),
        'lon': np.tile(longitudes, len(latitudes))
    })
    
    # Drop duplicates and assign unique node IDs
    metadata_df = metadata_df.drop_duplicates().reset_index(drop=True)
    metadata_df['node_id'] = metadata_df.index
    
    # Convert metadata to a NumPy array
    metadata_array = metadata_df.to_numpy()
    
    return metadata_df, metadata_array



def create_distance_matrix(metadata: pd.DataFrame) -> np.ndarray:
    """
    Computes the pairwise geographical distance matrix based on provided metadata.
    
    This function calculates the great-circle distances between locations
    using their latitude and longitude coordinates.
    
    Args:
        metadata (pd.DataFrame):
            A DataFrame containing latitude and longitude columns
            for the locations of interest.
    
    Returns:
        np.ndarray:
            A 2D NumPy array where each entry (i, j) represents the geographical 
            distance between location i and location j in kilometers.
    """
    
    # Compute the geographical distance matrix
    distance_matrix = geographical_distance(metadata, to_rad=True).values
    
    return distance_matrix


from typing import Optional, Union, List
import numpy as np
import pandas as pd 

from tsl.datasets.prototypes import TabularDataset
from tsl.ops.similarities import gaussian_kernel

class Dataset(TabularDataset):

    similarity_options = {'distance', 'grid'}

    def __init__(self,
                 target,
                 mask, 
                 distances,
                 u,
                 metadata,
                 method
                 ):

        covariates = {
            'u': (u),
            'metadata' : (metadata),
            'distances': (distances)
        }

        super().__init__(target=target,
                         mask=mask,
                         covariates=covariates,
                         similarity_score=method,
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         name='Dataset')


    def compute_similarity(self, method: str, **kwargs):
        """
        Compute similarity matrix based on the specified method.

        Args:
            method (str): The similarity computation method ('distance' or 'correlation').
            **kwargs: Additional keyword arguments for similarity computation.

        Returns:
            numpy.ndarray: Computed similarity matrix.

        Raises:
            ValueError: If an unknown similarity method is provided.
        """
        if method == "distance":
            # Calculate a Gaussian kernel similarity from the distance matrix, using a default or provided 'theta'
            theta = kwargs.get('theta', np.std(self.distances))
            return self.gaussian_kernel(self.distances, theta=theta)
        elif method == "correlation":
            # Compute the average correlation between nodes over the target features
            # Reshape target data to have nodes as columns
            target_values = self.target.values.reshape(len(self.target), -1, len(self.target_node_feature))
            # Average over the target features
            target_mean = target_values.mean(axis=2)
            # Compute correlation between nodes
            corr = np.corrcoef(target_mean, rowvar=False)
            return (corr + 1) / 2  # Normalize to [0, 1]
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    @staticmethod
    def gaussian_kernel(distances, theta):
        """
        Compute Gaussian kernel similarity from distances.

        Args:
            distances (numpy.ndarray): Distance matrix.
            theta (float): Kernel bandwidth parameter.

        Returns:
            numpy.ndarray: Gaussian kernel similarity matrix.
        """
        return np.exp(-(distances ** 2) / (2 * (theta ** 2)))


import os
import torch
import numpy as np
from typing import Optional

def get_connectivity_matrix(
    target: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    distances: Optional[np.ndarray] = None,
    covariates: Optional[np.ndarray] = None,
    metadata: Optional[np.ndarray] = None,
    method: str = 'distance',
    threshold: float = 0.1,
    knn: int = 8,
    binary_weights: bool = False,
    include_self: bool = False,
    force_symmetric: bool = True,
    layout: str = "csr"
):
    """
    Returns the weighted adjacency matrix A ∈ ℝ^(N × N), where N = self.n_nodes.
    The element a_{i,j} ∈ A is 0 if there does not exist an edge connecting node i to node j.
    The return type depends on the specified layout (default: 'edge_index').

    Args:
        target (Optional[np.ndarray]): Target data.
        mask (Optional[np.ndarray]): Mask data.
        distances (Optional[np.ndarray]): Precomputed distance matrix.
        covariates (Optional[np.ndarray]): Covariates data.
        metadata (Optional[np.ndarray]): Metadata.
        method (str): Method to compute connectivity ('distance' or 'grid').
        threshold (float): Threshold for edge creation based on similarity scores.
        knn (int): Number of nearest neighbors to include in the graph.
        binary_weights (bool): If True, use binary weights in the adjacency matrix.
        include_self (bool): Whether to include self-loops.
        force_symmetric (bool): Force the connectivity matrix to be symmetric.
        layout (str): Desired layout of the connectivity matrix ('csr', 'coo', etc.).

    Returns:
        scipy.sparse._csr.csr_matrix: Connectivity matrix in the specified layout.
    """
    dataset = Dataset(target=target,
                         mask=mask,
                         distances=distances,
                         u=covariates,
                         metadata=metadata,
                         method=method)
                         
    # Compute the connectivity matrix
    connectivity = dataset.get_connectivity(
        method=method,
        threshold=threshold,
        knn=knn,
        binary_weights=binary_weights,
        include_self=include_self,
        force_symmetric=force_symmetric,
        layout=layout
    )

    return connectivity


from typing import Optional
import numpy as np
from tsl.data import Dataset
from tsl.datasets import SpatioTemporalDataset

def get_torch_dataset(
    target: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    distances: Optional[np.ndarray] = None,
    covariates: Optional[np.ndarray] = None,
    metadata: Optional[np.ndarray] = None,
    method: str = 'distance',
    connectivity = None,
    horizon: int = 6,
    window: int = 12,
    stride: int = 1
) -> tsl.data.spatiotemporal_dataset.SpatioTemporalDataset:
    """
    Creates a spatio-temporal dataset for PyTorch-based processing.

    Args:
        target (Optional[np.ndarray]): Target data.
        mask (Optional[np.ndarray]): Mask data.
        distances (Optional[np.ndarray]): Precomputed distance matrix.
        covariates (Optional[np.ndarray]): Covariates data.
        metadata (Optional[np.ndarray]): Metadata.
        method (str): Method to compute connectivity ('distance' or 'grid').
        connectivity: Predefined connectivity structure for the dataset.
        horizon (int): Forecasting horizon (number of future steps to predict).
        window (int): Size of the historical window used for input features.
        stride (int): Step size between consecutive windows.

    Returns:
        tsl.data.spatiotemporal_dataset.SpatioTemporalDataset: 
        A PyTorch-ready spatio-temporal dataset object.
    """
    dataset = Dataset(
        target=target,
        mask=mask,
        distances=distances,
        u=covariates,
        metadata=metadata,
        method=method
    )
    
    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        mask=mask,
        covariates=covariates,
        connectivity=connectivity,
        horizon=horizon, 
        window=window, 
        stride=stride 
    )
    
    return torch_dataset



# def get_datamodule(
#     from tsl.data.preprocessing import StandardScaler, MinMaxScaler

#     scalers = {
#         'target': StandardScaler(axis=(0, 1)),
#         'u': StandardScaler(axis=(0, 1))
#     }

#     from tsl.data.datamodule import (SpatioTemporalDataModule,
#                                  TemporalSplitter)
                                 
#     # Split data sequentially:
#     #   |------------ dataset -----------|
#     #   |--- train ---|- val -|-- test --|
#     splitter = TemporalSplitter(val_len=0.1, test_len=0.2)

#     # Create a SpatioTemporalDataModule
#     datamodule = SpatioTemporalDataModule(
#         dataset=torch_dataset,
#         scalers=scalers,
#         mask_scaling=True,
#         splitter=splitter,
#         batch_size=4, 
#         workers=15
#         )

#     datamodule.setup()

# return datamodule