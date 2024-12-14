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

def extract_target_variable(
    ds: xr.Dataset,
    target_vars: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts specified target variables from an xarray Dataset, reshapes them for modeling, 
    and returns the resulting array along with a mask indicating non-NaN values.

    Parameters:
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
    
    Example usage:
        target, mask = extract_target_variable(ds, ['swvl1'])
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

# Example usage:
# target, mask = extract_target_variable(ds, ['swvl1'])

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

    Parameters:
        ds (xr.Dataset): The input xarray Dataset.
        variables (List[str]): List of variable names to extract from the Dataset as covariates.

    Returns:
        np.ndarray: A numpy array with shape (time, nodes, channels), where:
                    - time: The temporal dimension.
                    - nodes: Flattened latitude-longitude pairs.
                    - channels: The number of covariate variables.
    Example usage:
        covariates = extract_covariates(ds, [
            'u10', 'v10', 't2m', 'sst', 'sp', 'tp', 
            'ssr', 'ssrd', 'tcc', 'cl', 'e', 'pev', 
            'ro', 'asn', 'slt', 'cvh', 'lai_hv', 
            'tvh', 'z', 'season'
        ])
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

# Example usage:
# covariates = extract_covariates(ds, [
#     'u10', 'v10', 't2m', 'sst', 'sp', 'tp', 
#     'ssr', 'ssrd', 'tcc', 'cl', 'e', 'pev', 
#     'ro', 'asn', 'slt', 'cvh', 'lai_hv', 
#     'tvh', 'z', 'season'
# ])

import pandas as pd
import numpy as np
import xarray as xr

def generate_metadata_array(
    ds: xr.Dataset
) -> np.ndarray:
    """
    Generates a metadata numpy array from a given xarray Dataset by extracting unique
    latitude and longitude combinations and assigning a unique node ID to each combination.

    Parameters:
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

# Example usage:
# metadata_array = generate_metadata_array(ds=ds)
