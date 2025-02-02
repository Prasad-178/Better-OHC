import pandas as pd
import numpy as np
from typing import List, Union, Dict, Optional
import math
import json
import warnings


class BetterOHC:
    """
    A class for dynamic and flexible one-hot encoding with configurable thresholds and custom 'Other' class handling.
    """

    def __init__(self, csv_path: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None):
        """
        Initialize the BetterOHC class with either a CSV file path or an existing DataFrame.

        Args:
            csv_path (str, optional): Path to the CSV file.
            dataframe (pd.DataFrame, optional): Existing DataFrame.

        Raises:
            ValueError: If neither a CSV path nor a DataFrame is provided.
        """
        if csv_path is None and dataframe is None:
            raise ValueError(
                "You must provide either a CSV file path or an existing DataFrame."
            )

        self.df = pd.read_csv(csv_path) if csv_path else dataframe.copy()
        self.column_values: Dict[str, List[str]] = {}
        self.columns: List[str] = []
        self.new_class_name: Dict[str, str] = {}
        self.threshold_counts: Dict[str, int] = {}
        self.thresholds: Dict[str, float] = {}
        self.is_fitted = False


    def fit(
        self,
        columns: List[str],
        threshold: Union[float, Dict[str, float]] = 0.005,
        new_class_name: Union[str, Dict[str, str]] = "Other",
    ):
        """Fit the encoder by learning the value distributions and thresholds.

        Parameters
        ----------
        columns : list of str
            Names of the columns to be one-hot encoded.
        threshold : float or dict of str to float, optional
            Threshold as a percentage of the total rows. Values below 
            this threshold are grouped into an 'Other' class. Defaults to 0.005.
        new_class_name : str or dict of str to str, optional
            Name for the 'Other' class or a dictionary specifying names 
            for each column. Defaults to "Other".

        Returns
        -------
        self: The fitted encoder

        Notes
        -----
        The threshold can be provided as a single float value for all columns 
        or a dictionary for column-specific thresholds. Similarly, the 
        `new_class_name` can be customized per column.
        """

        self.columns = columns

        if isinstance(threshold, float):
            if not 0 < threshold < 1:
                raise ValueError(
                    "Threshold must be between 0 and 1."
                )
            self.thresholds = {column: threshold for column in columns}
        else:
            for column, value in threshold.items():
                if not 0 < value < 1:
                    raise ValueError(
                        f"Threshold for column '{column}' must be between 0 and 1."
                    )
            self.thresholds = threshold

        if isinstance(new_class_name, str):
            self.new_class_name = {column: new_class_name for column in columns}
        else:
            self.new_class_name = new_class_name

        self.threshold_counts = {
            column: math.floor(self.df.shape[0] * self.thresholds[column])
            for column in columns
        }

        for column in columns:
            value_counts = self.df[column].value_counts()
            self.column_values[column] = value_counts[
                value_counts >= self.threshold_counts[column]
            ].index.tolist()

            if len(value_counts[value_counts < self.threshold_counts[column]]) > 0:
                self.column_values[column].append(self.new_class_name[column])

        self.is_fitted = True
        return self


    def transform(
            self,
            X: Union[pd.DataFrame, str],
            columns: Optional[List[str]] = None,
            drop_original: bool = False
    ) -> pd.DataFrame:
        """Transform data using the fitted encoder.

        Args:
            X: DataFrame or path to CSV
            columns: Columns to encode (defaults to columns used in fit)
            drop_original: Whether to drop original columns

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transforming data.")
        
        if isinstance(X, str):
            df = pd.read_csv(X)
        else:
            df = X.copy()

        if columns is None:
            columns = self.columns

        missing_cols = [col for col in columns if col not in self.column_values]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} were not present during fitting")

        for column in columns:
            other_class = self.new_class_name[column]
            allowed_values = set(self.column_values[column])

            df[column] = df[column].apply(
                lambda x: x if x in allowed_values else other_class
            )

            one_hot = pd.get_dummies(df[column], prefix=column)

            for value in self.column_values[column]:
                col_name = f"{column}_{value}"
                if col_name not in one_hot.columns:
                    one_hot[col_name] = 0

            df = pd.concat([df, one_hot], axis=1)

            if drop_original:
                df.drop(column, axis=1, inplace=True)

        return df

        
    def fit_transform(
        self,
        columns: List[str],
        threshold: Union[float, Dict[str, float]] = 0.005,
        new_class_name: Union[str, Dict[str, str]] = "Other",
        drop_original: bool = False,
    ) -> pd.DataFrame:
        """
        Fit the encoder and transform the data in one step.

        Args:
            columns: Columns to encode
            threshold: Threshold value(s)
            new_class_name: Name(s) for Other class
            drop_original: Whether to drop original columns

        Returns:
            Transformed DataFrame
        """
        self.fit(columns, threshold, new_class_name)
        return self.transform(self.df, columns, drop_original)


    def get_feature_names(self) -> List[str]:
        """Get the names of the encoded features."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before getting feature names")
        
        feature_names = []
        for column in self.columns:
            feature_names.extend([f"{column}_{value}" for value in self.column_values[column]])
        return feature_names


    def save(self, path: str):
        """
        Save the fitted encoder to a file.

        Args:
            path: Path to save the encoder
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before saving")

        state = {
            'column_values': self.column_values,
            'columns': self.columns,
            'new_class_name': self.new_class_name,
            'threshold_counts': self.threshold_counts,
            'thresholds': self.thresholds,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'w') as f:
            json.dump(state, f)


    @classmethod
    def load(cls, path: str) -> 'BetterOHC':
        """
        Load a fitted encoder from a file.

        Args:
            path: Path to the saved encoder

        Returns:
            Loaded encoder
        """
        with open(path, 'r') as f:
            state = json.load(f)

        encoder = cls()
        encoder.column_values = state['column_values']
        encoder.columns = state['columns']
        encoder.new_class_name = state['new_class_name']
        encoder.threshold_counts = state['threshold_counts']
        encoder.thresholds = state['thresholds']
        encoder.is_fitted = state['is_fitted']

        return encoder
