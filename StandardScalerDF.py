import pandas as pd
class StandardScalerDF:
    '''
    This class has methods used for normalization of data using the method z=(x-u)/s where x is data,u is mean and s is std deviation
    This class is only applicable on dataframes as it uses the columnnames for keeping track of the means and standard deviations
    '''
    def fit(self,data:pd.DataFrame,columns_normalized=None,ddof:int=0):
        '''
        data: Dataframe whose statistics will be used for transforming data
        columns_normalized: iterable of strings containing the names of columns that are to be normalized, if None then all the numeric columns will be normalized
        ddof: decreased degrees of freedom , std dev is calculated using N-ddof degrees of freedom
        '''
        if columns_normalized is None:
            self.columns_normalized=set(data.columns)-set(data.select_dtypes(['object','timedelta64','bool','datetime64','category']).columns) ##getting the numeric data type column names
        else:
            self.columns_normalized=columns_normalized
        self.Scaler={} ##dictionary for storing corresponding means and 
        for column_name in self.columns_normalized:
            self.Scaler[column_name]=(data[column_name].mean(),data[column_name].std(ddof=ddof)) ## setting corresponding means and std in a tuple corresponding to the key of column name
    
    def transform(self,data:pd.DataFrame):
        '''
        data: Dataframe that needs to be normalized
        '''
        for column_name,stats_tuple in self.Scaler.items():
            data[column_name]=(data[column_name]-stats_tuple[0])/stats_tuple[1]
        return data
    
    def fit_transform(self,data:pd.DataFrame,columns_normalized:list=None,ddof:int=0):
        '''
        data: Dataframe whose statistics will be used for transforming data and which will be normalized using the same
        columns_normalized: iterable of strings containing the names of columns that are to be normalized, if None then all the numeric columns will be normalized
        ddof: decreased degrees of freedom , std dev is calculated using N-ddof degrees of freedom
        '''
        self.fit(data,columns_normalized,ddof)
        data_new=self.transform(data)
        return data_new