
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
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
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

import pandas as pd
import xarray as xr

def adjust_valid_time(dataset: xr.Dataset, hours_offset: int = 6) -> xr.Dataset:
    """
    Adjust the valid_time coordinate of an xarray.Dataset by subtracting a specified time delta.

    Args:
        dataset (xr.Dataset): The input dataset with a valid_time coordinate.
        hours_offset (int): Number of hours to subtract from the valid_time. Default is 6.

    Returns:
        xr.Dataset: The dataset with the adjusted valid_time coordinate.

    Raises:
        KeyError: If the dataset does not contain a 'valid_time' coordinate.
    """
    if 'valid_time' not in dataset.coords:
        raise KeyError("The dataset does not contain a 'valid_time' coordinate.")
    
    # Subtract the time delta from the underlying array
    timedelta = pd.Timedelta(hours=hours_offset)
    adjusted_valid_time = dataset['valid_time'].values - timedelta

    # Reassign the valid_time coordinate with the adjusted values
    dataset = dataset.assign_coords(valid_time=adjusted_valid_time)

    return dataset


def merge_datasets(datasets: List[xr.Dataset]) -> xr.Dataset:
    """
    Merge a list of xarray.Dataset objects into a single dataset.

    Args:
        datasets (List[xr.Dataset]): A list of datasets to merge.

    Returns:
        xr.Dataset: The merged dataset.

    Raises:
        ValueError: If the input list is empty.
    """
    if not datasets:
        raise ValueError("The list of datasets is empty. Provide at least one dataset to merge.")
    
    # Merge datasets
    merged_dataset = xr.merge(datasets)
    return merged_dataset



