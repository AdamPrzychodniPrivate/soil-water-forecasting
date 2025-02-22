
import pandas as pd
import xarray as xr
from typing import List

from typing import List
import xarray as xr

import xarray as xr

def merge_datasets(
    datasets
):
    """
    Merges multiple xarray datasets, ensuring all timestamps in the 'valid_time' coordinate
    are standardized to '00:00:00.000000000' by truncating to date only.
    
    Args:
        datasets (list): A list of xarray.Dataset objects to be merged.
    
    Returns:
        xr.Dataset: A merged dataset with standardized timestamps.
    """
    standardized_datasets = []
    
    for ds in datasets:
        if "valid_time" in ds.coords:
            ds["valid_time"] = ds["valid_time"].dt.floor("D")  # Truncate to date only
        else:
            raise ValueError("Dataset does not contain 'valid_time' coordinate.")
        
        standardized_datasets.append(ds)
    
    # Merge the standardized datasets
    merged_ds = xr.merge(standardized_datasets)
    
    return merged_ds


import xarray as xr
from typing import List, Union

def fillna_in_variables(
    ds: xr.Dataset,
    variables: List[str],
    fill_value: Union[int, float]
) -> xr.Dataset:
    """
    Fills NaN values in specified variables of an xarray.Dataset with a provided value.

    Args:
        ds (xr.Dataset): The input dataset containing the data variables.
        variables (List[str]): A list of variable names in the dataset for which to fill NaN values.
        fill_value (Union[int, float]): The value to fill NaN values with (e.g., 0).

    Returns:
        xr.Dataset: A new dataset with NaN values filled in the specified variables.
    
    """
    # Check if each variable exists in the dataset
    for var in variables:
        if var not in ds:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
    
    # Fill NaN values for the specified variables
    filled_ds = ds.copy()
    for var in variables:
        filled_ds[var] = filled_ds[var].fillna(fill_value)
    
    return filled_ds


from typing import List, Union, Optional, Literal
import numpy as np
import xarray as xr

def interpolate_to_target_grid(
    ds: xr.Dataset, 
    target_lat: np.ndarray, 
    target_lon: np.ndarray, 
    method: Literal["linear", "nearest", "zero", "slinear", "quadratic", "cubic", "quintic"] = "linear"
) -> xr.Dataset:
    """
    Interpolates an xarray Dataset to a specified target latitude and longitude grid.
    
    Args:
        ds (xr.Dataset): The input dataset to be interpolated.
        target_lat (np.ndarray): Target latitude values as a NumPy array.
        target_lon (np.ndarray): Target longitude values as a NumPy array.
        method (Literal): Interpolation method, restricted to allowed values.

    Returns:
        xr.Dataset: The interpolated dataset on the specified target grid.
    """
    if ds.latitude[0] > ds.latitude[-1]:
        ds = ds.sortby("latitude")

    interpolated_ds = ds.interp(latitude=target_lat, longitude=target_lon, method=method)
    return interpolated_ds


from typing import List, Union, Optional


def preprocess(
    ds_0: xr.Dataset,
    ds_1: xr.Dataset,
    ds_2: xr.Dataset,
    target_lat: List[float],
    target_lon: List[float],
    method: Literal["linear", "nearest", "zero", "slinear", "quadratic", "cubic", "quintic"] = "linear",
    fill_variables: Optional[List[str]] = None,
    fill_value: Union[int, float] = 0
) -> xr.Dataset:
    """
    Preprocesses and merges multiple xarray datasets:
      1. Merges datasets.
      2. Fills NaN values in specified variables.
      3. Drops unnecessary variables.
      4. Interpolates to target grid.

    Args:
        ds_0 (xr.Dataset): First dataset.
        ds_1 (xr.Dataset): Second dataset.
        ds_2 (xr.Dataset): Third dataset.
        target_lat (List[float]): Target latitude grid as list from YAML.
        target_lon (List[float]): Target longitude grid as list from YAML.
        method (str): Interpolation method.
        fill_variables (List[str]): Variables to fill NaN values.
        fill_value (Union[int, float]): Fill value.

    Returns:
        xr.Dataset: Processed dataset.
    """
    datasets = [ds_0, ds_1, ds_2]
    ds = merge_datasets(datasets)

    if fill_variables:
        for var in fill_variables:
            if var in ds:
                ds[var] = ds[var].fillna(fill_value)

    drop_vars = ["number", "expver"]
    ds = ds.drop_vars([var for var in drop_vars if var in ds])

    # Convert YAML list to NumPy array before interpolation
    target_lat_np = np.arange(*target_lat)  # Converts [-90, 90, 1] → np.arange(-90, 90, 1)
    target_lon_np = np.arange(*target_lon)  # Converts [0, 360, 1] → np.arange(0, 360, 1)

    # Interpolation step
    ds = interpolate_to_target_grid(ds, target_lat_np, target_lon_np, method)

    return ds



import xarray as xr

def add_season_to_dataset(ds: xr.Dataset, time_var: str = "valid_time", lat_var: str = "latitude", lon_var: str = "longitude") -> xr.Dataset:
    """
    Adds a 'season' variable to an xarray.Dataset based on the time coordinate, 
    with values mapped to numerical representations of seasons.

    Args:
        ds (xr.Dataset): The input dataset with a time coordinate.
        time_var (str): Name of the time variable in the dataset. Default is 'valid_time'.
        lat_var (str): Name of the latitude variable in the dataset. Default is 'latitude'.
        lon_var (str): Name of the longitude variable in the dataset. Default is 'longitude'.

    Returns:
        xr.Dataset: The modified dataset with an added 'season' variable.
    """
    # Ensure the time variable is available in the dataset
    if time_var not in ds:
        raise ValueError(f"The dataset does not contain the specified time variable '{time_var}'.")

    # Extract the season from the time coordinate
    ds["season"] = ds[time_var].dt.season

    # Define a dictionary to map season strings to numerical values
    season_mapping = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}

    # Convert the DataArray to a pandas Series, map the season names, and convert back to a DataArray
    ds["season"] = ds["season"].to_series().map(season_mapping).to_xarray()

    # Broadcast 'season' to match dimensions (time_var, lat_var, lon_var)
    season_broadcasted = ds["season"].expand_dims({lat_var: ds[lat_var], lon_var: ds[lon_var]})

    # Add this expanded 'season' variable back to the dataset
    ds["season"] = season_broadcasted

    # Transpose 'season' to match the desired dimensions (time_var, lat_var, lon_var)
    ds["season"] = ds["season"].transpose(time_var, lat_var, lon_var)

    # Convert season to int32
    ds["season"] = ds["season"].astype("int32")

    return ds


import xarray as xr
from typing import Any

def update_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Update dataset variables for water masking and unit conversion.
    
    This function applies several updates to the input dataset:
      - For 'swvl1': Sets values to 1 wherever 'lsm' indicates water 
        (assumed to be where ds['lsm'] == 0).
      - For 'pev': Sets values to 0 wherever 'lsm' indicates water.
      - For 'tp', 'pev', and 'e': Multiplies the values by 1000 and updates 
        their 'units' attribute to 'mm'.

    Args:
        ds (xr.Dataset): The input dataset containing the variables:
            - 'swvl1': Soil water variable.
            - 'lsm': Land-sea mask (assumes 0 indicates water).
            - 'pev': Potential evapotranspiration variable.
            - 'tp': Total precipitation variable.
            - 'e': Evaporation variable.

    Returns:
        xr.Dataset: The updated dataset with modified variables.
    """
    # Check that all required variables exist in the dataset.
    required_vars = ['swvl1', 'lsm', 'pev', 'tp', 'e']
    for var in required_vars:
        if var not in ds:
            raise KeyError(f"Dataset must contain variable '{var}'")
    
    # Update swvl1: Set to 1 wherever lsm indicates water (assuming water is represented by 0).
    ds['swvl1'] = ds['swvl1'].where(ds['lsm'] != 0, other=1)
    
    # Update pev: Set to 0 wherever lsm indicates water.
    ds['pev'] = ds['pev'].where(ds['lsm'] != 0, other=0)
    
    # Convert 'tp' values by multiplying by 1000 and update attributes.
    tp_attrs = ds['tp'].attrs.copy()  # Copy original attributes.
    ds['tp'] = ds['tp'] * 1000
    tp_attrs['units'] = 'mm'
    ds['tp'].attrs = tp_attrs
    
    # Convert 'pev' values by multiplying by 1000 and update attributes.
    pev_attrs = ds['pev'].attrs.copy()
    ds['pev'] = ds['pev'] * 1000
    pev_attrs['units'] = 'mm'
    ds['pev'].attrs = pev_attrs
    
    # Convert 'e' values by multiplying by 1000 and update attributes.
    e_attrs = ds['e'].attrs.copy()
    ds['e'] = ds['e'] * 1000
    e_attrs['units'] = 'mm'
    ds['e'].attrs = e_attrs

    return ds


def feature_engineering(ds: xr.Dataset) -> xr.Dataset:
    """
    Performs feature engineering on the input xarray Dataset.
    
    The function executes the following steps:
      1. Adds seasonal variables based on time, latitude, and longitude.
      2. Applies dataset updates for water masking and unit conversion.

    Args:
        ds (xr.Dataset): The input xarray Dataset containing meteorological and geographical variables.

    Returns:
        xr.Dataset: The processed dataset with additional seasonal features and updated variables.
    """
    # Step 1: Add seasonal features based on time, latitude, and longitude
    ds = add_season_to_dataset(ds, time_var="valid_time", lat_var="latitude", lon_var="longitude")
    
    # Step 2: Apply dataset updates
    ds = update_dataset(ds)
    
    return ds