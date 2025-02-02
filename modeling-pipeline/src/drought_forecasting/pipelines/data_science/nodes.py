import logging

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


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


import pandas as pd
import numpy as np
import xarray as xr

def generate_metadata_array(
    ds: xr.Dataset
) -> np.ndarray:
    """
    Generates a metadata numpy array from a given xarray Dataset by extracting unique
    latitude and longitude combinations and assigning a unique node ID to each combination.

    Args:
        ds (xr.Dataset): Preloaded xarray Dataset containing spatial coordinates.

    Returns:
        np.ndarray: A numpy array containing latitude, longitude, and node IDs, with shape (nodes, 3).
    
    Example usage:
        metadata_array = generate_metadata_array(ds=ds)
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

    # Convert metadata to a numpy array
    metadata_array = metadata_df.to_numpy()

    return metadata_array


import pandas as pd
from tsl.ops.similarities import geographical_distance

import os
import numpy as np
import pandas as pd

def create_distance_matrix(parquet_path: str, to_rad: bool = True):
    """
    Create a geographical distance matrix from metadata stored in a parquet file.
    If the distance matrix already exists, load it instead of recomputing.

    Args:
        parquet_path (str): Path to the parquet file containing metadata.
        to_rad (bool): Whether to convert coordinates to radians before calculation.

    Returns:
        numpy.ndarray: A NumPy array containing the distance matrix.

    Example usage:
        distance_matrix = create_distance_matrix(parquet_file_path)
    """
    distance_matrix_path = "drought-forecasting/modeling-pipeline/data/05_model_input/distance_matrix.npy"

    # Check if the file already exists
    if os.path.exists(distance_matrix_path):
        print("Loading existing distance matrix from file.")
        return np.load(distance_matrix_path)

    print("Generating new distance matrix.")

    # Load metadata from the parquet file
    metadata = pd.read_parquet(parquet_path)

    # Validate the presence of required columns
    required_columns = {'latitude', 'longitude'}
    if not required_columns.issubset(metadata.columns):
        raise ValueError(f"The metadata file must contain the columns: {required_columns}")

    # Calculate geographical distances
    distance_matrix = geographical_distance(metadata, to_rad=to_rad).values

    # Save the computed distance matrix for future use
    np.save(distance_matrix_path, distance_matrix)

    return distance_matrix


import numpy as np
import pandas as pd 

from tsl.datasets.prototypes import TabularDataset
from tsl.ops.similarities import gaussian_kernel

class SoilWaterDataset(TabularDataset):

    similarity_options = {'distance', 'grid'}

    def __init__(self,
                 mode: str = 'connectivity',
                 target: Optional[np.ndarray] = None,
                 mask: Optional[np.ndarray] = None,
                 distances: Optional[np.ndarray] = None,
                 covariates: Optional[np.ndarray] = None,
                 metadata: Optional[np.ndarray] = None):
        """
        Initialize the SoilWaterDataset.

        Args:
            mode (str): Mode of the dataset. Options are 'connectivity' or 'training'.
            target (Optional[np.ndarray]): Target data.
            mask (Optional[np.ndarray]): Mask data.
            distances (Optional[np.ndarray]): Precomputed distance matrix (required in connectivity mode).
            covariates (Optional[np.ndarray]): Covariates data.
            metadata (Optional[np.ndarray]): Metadata (required in connectivity mode).
        """
        self.mode = mode
        self.target = target
        self.mask = mask
        self.distances = distances
        self.covariates = covariates
        self.metadata = metadata

        if self.mode not in ['connectivity', 'training']:
            raise ValueError("Mode must be either 'connectivity' or 'training'")

        if self.mode == 'connectivity':
            if self.distances is None or self.metadata is None:
                raise ValueError("'distances' and 'metadata' are required in 'connectivity' mode")

        covariates_dict = {
            'u': self.covariates
        }

        if self.mode == 'connectivity':
            covariates_dict.update({
                'metadata': self.metadata,
                'distances': self.distances
            })

        super().__init__(target=self.target,
                         mask=self.mask,
                         covariates=covariates_dict,
                         similarity_score='distance',
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         name='SoilWaterDataset')

    def compute_similarity(self, method: str, **kwargs):
        """
        Compute similarity matrix based on the specified method.

        Args:
            method (str): The similarity computation method ('distance' or 'grid').
            **kwargs: Additional keyword arguments for similarity computation.

        Returns:
            numpy.ndarray: Computed similarity matrix.

        Raises:
            ValueError: If an unknown similarity method is provided.
        """
        if method == "distance":
            # Calculate a Gaussian kernel similarity from the distance matrix, using a default or provided 'theta'
            theta = kwargs.get('theta', np.std(self.distances))
            return gaussian_kernel(self.distances, theta=theta)
        elif method == "grid":
            dist = self.distances.copy()
            dist[dist > 16] = np.inf  # keep only grid edges
            theta = kwargs.get('theta', 20)
            return gaussian_kernel(dist, theta=theta)
        else:
            raise ValueError(f"Unknown similarity method: {method}")


import os
import torch
import numpy as np
from typing import Optional

def get_connectivity_matrix(
    mode: str = 'connectivity',
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
    Compute the connectivity matrix for the SoilWaterDataset.
    If a precomputed connectivity matrix exists, load it instead of recomputing.

    Args:
        mode (str): Mode of the dataset. Options are 'connectivity' or 'training'.
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
        scipy.sparse.spmatrix: Connectivity matrix in the specified layout.
    """
    connectivity_matrix_path = "drought-forecasting/modeling-pipeline/data/05_model_input/connectivity.pt"

    # Check if the precomputed connectivity matrix exists
    if os.path.exists(connectivity_matrix_path):
        print("Loading existing connectivity matrix from file.")
        return torch.load(connectivity_matrix_path)

    print("Generating new connectivity matrix.")

    # Create an instance of SoilWaterDataset in the specified mode
    dataset = SoilWaterDataset(mode=mode,
                               target=target,
                               mask=mask,
                               distances=distances,
                               covariates=covariates,
                               metadata=metadata)

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

    # Save the computed connectivity matrix for future use
    torch.save(connectivity, connectivity_matrix_path)

    return connectivity



# Create an instance of SoilWaterDataset in the specified mode
# dataset = SoilWaterDataset(mode='training',
#                             target=target,
#                             mask=mask,
#                             covariates=covariates)

# import torch

# connectivity = torch.load("soil-water-forecasting/modeling-pipeline/data/05_model_input/connectivity.pt")

# from tsl.data import SpatioTemporalDataset

# # covariates=dict(u=dataset.covariates['u'])
# covariates=dataset.covariates
# mask = dataset.mask

# horizon=6
# window=12
# stride=1

# torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
#                                       mask=mask,
#                                       covariates=covariates,
#                                       connectivity=connectivity,
#                                       horizon=horizon, 
#                                       window=window, 
#                                       stride=stride 
#                                       )

# czy ja moge to jakos zapisac?