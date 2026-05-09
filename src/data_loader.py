import pandas as pd 
import logging 



def data_ingestion(file_path: str, encoding: str):
    """
    This function reads the data from the given file path and returns the dataframe object.
    """
    try: 
        df = pd.read_csv(file_path, encoding=encoding)

        return df 
    
    except Exception as e:
        print(f"Error occured while reading the data:{e}")
        return None
    