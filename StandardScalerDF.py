import pandas as pd
import numpy as np

class StandardScalerDF:
    '''
    This class normalizes data using z = (x - u) / s, where x is the data, u is the mean, and s is the standard deviation.
    Applicable only to DataFrames.
    '''
    def fit(self, data: pd.DataFrame, columns_excluded=None, ddof:int=0):
        '''
        Fit the scaler to the data. 
        Parameters:
        - data: DataFrame to calculate statistics from.
        - columns_excluded: Iterable of strings for columns to exclude from normalization; if None, all numeric columns are used.
        - ddof: Degrees of freedom for standard deviation calculation.
        '''
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        if columns_excluded is None:
            self.columns_normalized = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            invalid_columns = set(columns_excluded) - set(data.columns)
            if invalid_columns:
                raise ValueError(f"Columns not found in DataFrame: {invalid_columns}")
            self.columns_normalized = data.select_dtypes(include=[np.number]).columns.tolist()
            self.columns_normalized = [col for col in self.columns_normalized if col not in columns_excluded]

        self.Scaler = {}
        for column_name in self.columns_normalized:
            mean = data[column_name].mean()
            std = data[column_name].std(ddof=ddof)
            self.Scaler[column_name] = (mean, std)
    
    def transform(self, data: pd.DataFrame):
        '''
        Transform the data using the fitted scaler.
        Parameters:
        - data: DataFrame to normalize.
        Returns:
        - Normalized DataFrame.
        '''
        for column_name, (mean, std) in self.Scaler.items():
            if column_name in data.columns:
                data[column_name] = (data[column_name] - mean) / std
            else:
                raise ValueError(f"Column '{column_name}' not found in input DataFrame.")
        return data
    
    def fit_transform(self, data: pd.DataFrame, columns_excluded=None, ddof:int=0):
        '''
        Fit to the data and then transform it.
        Parameters:
        - data: DataFrame to fit and transform.
        - columns_excluded: Columns to exclude from normalization.
        - ddof: Degrees of freedom for std calculation.
        Returns:
        - Normalized DataFrame.
        '''
        self.fit(data, columns_excluded, ddof)
        return self.transform(data)

    def inverse_transform(self, data: pd.DataFrame):
        '''
        Reverse the normalization process.
        Parameters:
        - data: DataFrame to inverse transform.
        Returns:
        - Denormalized DataFrame.
        '''
        for column_name, (mean, std) in self.Scaler.items():
            if column_name in data.columns:
                data[column_name] = data[column_name] * std + mean
            else:
                raise ValueError(f"Column '{column_name}' not found in input DataFrame.")
        return data
