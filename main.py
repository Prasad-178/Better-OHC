import pandas as pd
from typing import List, Union, Dict
import math


class BetterOHC:
    """
    A class for dynamic and flexible one-hot encoding with configurable thresholds and custom 'Other' class handling.
    """

    def __init__(self, csv_path: str = None, dataframe: pd.DataFrame = None):
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

    def encode(
        self,
        columns: List[str],
        threshold: Union[float, Dict[str, float]] = 0.005,
        new_class_name: Union[str, Dict[str, str]] = "Other",
        return_type: bool = False,
        drop_original: bool = False,
    ):
        """Perform one-hot encoding with dynamic threshold and 'Other' grouping.

        This method encodes the specified columns by creating binary 
        indicators for unique values, grouping rare occurrences into 
        an 'Other' class based on a threshold.

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
        return_type : bool, optional
            If True, returns the modified DataFrame. Defaults to False.
        drop_original : bool, optional
            If True, removes the original columns after encoding. 
            Defaults to False.

        Returns
        -------
        pd.DataFrame, optional
            Encoded DataFrame if `return_type` is True.

        Notes
        -----
        The threshold can be provided as a single float value for all columns 
        or a dictionary for column-specific thresholds. Similarly, the 
        `new_class_name` can be customized per column.
        """
        self.columns = columns
        if isinstance(threshold, float):
            threshold = {column: threshold for column in columns}
        else:
            for column, value in threshold.items():
                assert (
                    0 < value < 1
                ), f"Threshold for column '{column}' must be between 0 and 1."

        if isinstance(new_class_name, str):
            new_class_name = {column: new_class_name for column in columns}

        threshold_counts = {
            column: math.floor(self.df.shape[0] * threshold[column])
            for column in columns
        }
        self.threshold_counts = threshold_counts
        self.new_class_name = new_class_name

        for column in columns:
            value_counts = self.df[column].value_counts()
            keys_below_threshold = value_counts[
                value_counts < threshold_counts[column]
            ].index.tolist()

            self.df[column] = self.df[column].apply(
                lambda x: new_class_name[column] if x in keys_below_threshold else x
            )
            self.column_values[column] = self.df[column].value_counts().index.tolist()

            one_hot = pd.get_dummies(self.df[column], prefix=column)
            self.df = pd.concat([self.df, one_hot], axis=1)

            if drop_original:
                self.df.drop(column, axis=1, inplace=True)

        if return_type:
            return self.df

    def encode_test(
        self,
        test_csv_path: str = None,
        test_dataframe: pd.DataFrame = None,
        columns: Union[List[str], None] = None,
        drop_original: bool = False,
    ):
        """Apply one-hot encoding to a test DataFrame.

        Encodes the test dataset using the values learned during the 
        training set encoding. Rare values in the test data are grouped 
        into the 'Other' class.

        Parameters
        ----------
        test_csv_path : str, optional
            Path to the test CSV file. Either this or `test_dataframe` 
            must be provided.
        test_dataframe : pandas.DataFrame, optional
            Test DataFrame. Either this or `test_csv_path` must be provided.
        columns : list of str, optional
            Names of the columns to encode. Defaults to the columns encoded 
            during training.
        drop_original : bool, optional
            If True, removes the original columns after encoding. 
            Defaults to False.

        Returns
        -------
        pd.DataFrame
            Test DataFrame with one-hot encoded columns.

        Notes
        -----
        The function ensures that test data is encoded consistently with 
        the training data. Columns not seen during training are filled 
        with zeros for compatibility.
        """
        if test_csv_path is None and test_dataframe is None:
            raise ValueError(
                "You must provide either a test CSV file path or an existing test DataFrame."
            )

        test_df = pd.read_csv(test_csv_path) if test_csv_path else test_dataframe.copy()

        if columns is None:
            columns = self.columns

        for column in columns:
            assert (
                column in self.column_values
            ), f"Column '{column}' was not encoded in the training set."

            other_class = self.new_class_name[column]
            allowed_values = set(self.column_values[column])

            test_df[column] = test_df[column].apply(
                lambda x: x if x in allowed_values else other_class
            )

            one_hot = pd.get_dummies(test_df[column], prefix=column)
            test_df = pd.concat([test_df, one_hot], axis=1)

            for col in [f"{column}_{value}" for value in self.column_values[column]]:
                if col not in test_df.columns:
                    test_df[col] = 0

            if drop_original:
                test_df.drop(column, axis=1, inplace=True)

        return test_df
