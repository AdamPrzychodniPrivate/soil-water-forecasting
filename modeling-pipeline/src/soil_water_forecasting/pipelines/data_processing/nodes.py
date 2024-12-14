
import pandas as pd
import xarray as xr
from typing import List


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_companies(companies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies, {"columns": companies.columns.tolist(), "data_type": "companies"}


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, 
    companies: pd.DataFrame, 
    reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table

from typing import List
import xarray as xr

import xarray as xr

def merge_datasets(
    datasets
):
    """
    Merges multiple xarray datasets, ensuring all timestamps in the 'valid_time' coordinate
    are standardized to '00:00:00.000000000' by truncating to date only.
    
    Parameters:
        datasets (list): A list of xarray.Dataset objects to be merged.
    
    Returns:
        xr.Dataset: A merged dataset with standardized timestamps.

    Example:
        datasets = [ds_0, ds_1, ds_2]
        ds = merge_datasets(datasets)
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
    
    Example:
        ds = fillna_in_variables(ds, ["sst", "t2m"], fill_value=0)
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


import numpy as np
import xarray as xr

def interpolate_to_target_grid(
    ds, 
    target_lat, 
    target_lon, 
    method="linear"
):
    """
    Interpolates an xarray Dataset to a specified target latitude and longitude grid.
    
    Parameters:
        ds (xr.Dataset): The input dataset to be interpolated.
        target_lat (array-like): Target latitude values (e.g., np.arange(-90, 90, 1)).
        target_lon (array-like): Target longitude values (e.g., np.arange(0, 360, 1)).
        method (str): Interpolation method. Defaults to 'linear'.
                      Options include 'linear', 'nearest', etc.
    
    Returns:
        xr.Dataset: The interpolated dataset on the specified target grid.

    Example:
        target_lat = np.arange(-90, 90, 1)   # 1° resolution latitude
        target_lon = np.arange(0, 360, 1)   # 1° resolution longitude
        method = "linear"

        ds = interpolate_to_target_grid(ds, target_lat, target_lon, method)
    """
    # Step 1: Ensure latitude is in ascending order
    if ds.latitude[0] > ds.latitude[-1]:
        ds = ds.sortby("latitude")
    
    # Step 2: Perform interpolation
    interpolated_ds = ds.interp(latitude=target_lat, longitude=target_lon, method=method)
    
    return interpolated_ds



# datasets = [ds_0, ds_1, ds_2]
# ds = merge_datasets(datasets)

# # Update swvl1: Set to 1 wherever lsm indicates water (lsm == 1)
# ds['swvl1'] = ds['swvl1'].where(ds['lsm'] != 0, other=1)

# ds = fillna_in_variables(ds, ["sst"], fill_value=0)

# ds = ds.drop_vars(['number', 'expver'])

# # Example usage
# target_lat = np.arange(-90, 90, 1)   # 1° resolution latitude
# target_lon = np.arange(0, 360, 1)   # 1° resolution longitude
# method = "linear"

# ds = interpolate_to_target_grid(ds, target_lat, target_lon, method)

from typing import List, Union
import numpy as np
import xarray as xr

def preprocess(
    ds_0: xr.Dataset,
    ds_1: xr.Dataset,
    ds_2: xr.Dataset,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: str = "linear",
    fill_variables: List[str] = None,
    fill_value: Union[int, float] = 0
) -> xr.Dataset:
    """
    Preprocesses and merges multiple xarray datasets with the following steps:
      1. Merges datasets.
      2. Updates 'swvl1': Sets it to 1 wherever 'lsm' indicates water (lsm == 1).
      3. Fills NaN values in specified variables with a given fill value.
      4. Drops unnecessary variables.
      5. Interpolates the dataset to a target latitude and longitude grid.

    Parameters:
        datasets (List[xr.Dataset]): List of datasets to merge and preprocess.
        target_lat (np.ndarray): Target latitude grid for interpolation.
        target_lon (np.ndarray): Target longitude grid for interpolation.
        method (str): Interpolation method. Defaults to 'linear'.
        fill_variables (List[str]): Variables to fill NaN values for. Defaults to None.
        fill_value (Union[int, float]): Value to fill NaN values with. Defaults to 0.

    Returns:
        xr.Dataset: The preprocessed and interpolated dataset.
    """
    # Step 1: Merge datasets
    datasets = [ds_0, ds_1, ds_2]
    ds = merge_datasets(datasets)

    # Step 2: Update 'swvl1': Set to 1 wherever 'lsm' indicates water (lsm == 1)
    if "swvl1" in ds and "lsm" in ds:
        ds["swvl1"] = ds["swvl1"].where(ds["lsm"] != 0, other=1)

    # Step 3: Fill NaN values in specified variables
    if fill_variables:
        ds = fillna_in_variables(ds, fill_variables, fill_value)

    # Step 4: Drop unnecessary variables
    drop_vars = ["number", "expver"]
    ds = ds.drop_vars([var for var in drop_vars if var in ds])

    # Step 5: Interpolate to the target grid
    ds = interpolate_to_target_grid(ds, target_lat, target_lon, method)

    return ds

# # Example usage
# datasets = [ds_0, ds_1, ds_2]
# target_lat = np.arange(-90, 90, 1)   # 1° resolution latitude
# target_lon = np.arange(0, 360, 1)   # 1° resolution longitude
# fill_variables = ["sst"]  # Variables to fill NaN values for

# ds = preprocess(
#     datasets=datasets,
#     target_lat=target_lat,
#     target_lon=target_lon,
#     method="linear",
#     fill_variables=fill_variables,
#     fill_value=0
# )

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
    Example: 
        ds_with_season = add_season_to_dataset(ds, time_var="valid_time", lat_var="latitude", lon_var="longitude")

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

# ds_with_season = add_season_to_dataset(ds, time_var="valid_time", lat_var="latitude", lon_var="longitude")


def feature_engineering(
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    Performs feature engineering on the given xarray Dataset with the following steps:
      1. Adds seasonal variables to the dataset based on time, latitude, and longitude.
    
    Parameters:
        ds (xr.Dataset): The input xarray Dataset to preprocess.

    Returns:
        xr.Dataset: The processed xarray Dataset with additional seasonal variables.
    """
    # Step 1: Add seasons variables to dataset
    ds = add_season_to_dataset(ds, time_var="valid_time", lat_var="latitude", lon_var="longitude")

    return ds

# # Example usage
# ds = feature_engineering(ds)
