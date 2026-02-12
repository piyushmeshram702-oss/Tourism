import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and clean tourism data from Excel files."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir (str): Directory containing Excel files
        """
        self.data_dir = data_dir
        self.dataframes = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all Excel files into pandas DataFrames.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of loaded dataframes
        """
        files = {
            'transaction': 'Transaction.xlsx',
            'user': 'User.xlsx',
            'city': 'City.xlsx',
            'country': 'Country.xlsx',
            'region': 'Region.xlsx',
            'continent': 'Continent.xlsx',
            'item': 'Item.xlsx',
            'type': 'Type.xlsx',
            'mode': 'Mode.xlsx'
        }
        
        for key, filename in files.items():
            file_path = os.path.join(self.data_dir, filename)
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                self.dataframes[key] = df
                logger.info(f"Loaded {filename}: {df.shape}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                raise
                
        return self.dataframes
    
    def clean_data(self) -> Dict[str, pd.DataFrame]:
        """
        Clean all loaded dataframes.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of cleaned dataframes
        """
        if not self.dataframes:
            self.load_all_data()
            
        # Clean each dataframe
        self.dataframes['transaction'] = self._clean_transaction_data(self.dataframes['transaction'])
        self.dataframes['user'] = self._clean_user_data(self.dataframes['user'])
        self.dataframes['city'] = self._clean_city_data(self.dataframes['city'])
        self.dataframes['country'] = self._clean_country_data(self.dataframes['country'])
        self.dataframes['region'] = self._clean_region_data(self.dataframes['region'])
        self.dataframes['continent'] = self._clean_continent_data(self.dataframes['continent'])
        self.dataframes['item'] = self._clean_item_data(self.dataframes['item'])
        self.dataframes['type'] = self._clean_type_data(self.dataframes['type'])
        self.dataframes['mode'] = self._clean_mode_data(self.dataframes['mode'])
        
        return self.dataframes
    
    def _clean_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean transaction data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.dropna()
        
        # Validate rating values (should be 1-5)
        df = df[(df['Rating'] >= 1) & (df['Rating'] <= 5)]
        
        # Validate VisitMode values
        valid_modes = [1, 2, 3, 4, 5]  # Business, Couples, Family, Friends, Solo
        df = df[df['VisitMode'].isin(valid_modes)]
        
        # Convert data types
        df['VisitYear'] = df['VisitYear'].astype(int)
        df['VisitMonth'] = df['VisitMonth'].astype(int)
        df['Rating'] = df['Rating'].astype(int)
        
        logger.info(f"Cleaned transaction data: {df.shape}")
        return df
    
    def _clean_user_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean user data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values - fill with mode for categorical data
        for col in ['ContinentId', 'RegionId', 'CountryId', 'CityId']:
            if col in df.columns:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(mode_value)
        
        # Convert to appropriate data types
        df['UserId'] = df['UserId'].astype(int)
        df['ContinentId'] = df['ContinentId'].astype(int)
        df['RegionId'] = df['RegionId'].astype(int)
        df['CountryId'] = df['CountryId'].astype(int)
        df['CityId'] = df['CityId'].astype(int)
        
        logger.info(f"Cleaned user data: {df.shape}")
        return df
    
    def _clean_city_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean city data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['CityName'] = df['CityName'].fillna('Unknown')
        df['CountryId'] = df['CountryId'].fillna(0)
        
        # Convert data types
        df['CityId'] = df['CityId'].astype(int)
        df['CountryId'] = df['CountryId'].astype(int)
        
        logger.info(f"Cleaned city data: {df.shape}")
        return df
    
    def _clean_country_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean country data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['Country'] = df['Country'].fillna('Unknown')
        df['RegionId'] = df['RegionId'].fillna(0)
        
        # Convert data types
        df['CountryId'] = df['CountryId'].astype(int)
        df['RegionId'] = df['RegionId'].astype(int)
        
        logger.info(f"Cleaned country data: {df.shape}")
        return df
    
    def _clean_region_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean region data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['Region'] = df['Region'].fillna('Unknown')
        df['ContinentId'] = df['ContinentId'].fillna(0)
        
        # Convert data types
        df['RegionId'] = df['RegionId'].astype(int)
        df['ContinentId'] = df['ContinentId'].astype(int)
        
        logger.info(f"Cleaned region data: {df.shape}")
        return df
    
    def _clean_continent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean continent data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['Continent'] = df['Continent'].fillna('Unknown')
        
        # Convert data types
        df['ContinentId'] = df['ContinentId'].astype(int)
        
        logger.info(f"Cleaned continent data: {df.shape}")
        return df
    
    def _clean_item_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean item (attraction) data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['Attraction'] = df['Attraction'].fillna('Unknown Attraction')
        df['AttractionAddress'] = df['AttractionAddress'].fillna('Unknown Address')
        df['AttractionCityId'] = df['AttractionCityId'].fillna(0)
        df['AttractionTypeId'] = df['AttractionTypeId'].fillna(0)
        
        # Convert data types
        df['AttractionId'] = df['AttractionId'].astype(int)
        df['AttractionCityId'] = df['AttractionCityId'].astype(int)
        df['AttractionTypeId'] = df['AttractionTypeId'].astype(int)
        
        logger.info(f"Cleaned item data: {df.shape}")
        return df
    
    def _clean_type_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean attraction type data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['AttractionType'] = df['AttractionType'].fillna('Unknown Type')
        
        # Convert data types
        df['AttractionTypeId'] = df['AttractionTypeId'].astype(int)
        
        logger.info(f"Cleaned type data: {df.shape}")
        return df
    
    def _clean_mode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean visit mode data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['VisitMode'] = df['VisitMode'].fillna('Unknown')
        
        # Convert data types
        df['VisitModeId'] = df['VisitModeId'].astype(int)
        
        logger.info(f"Cleaned mode data: {df.shape}")
        return df
    
    def save_cleaned_data(self, output_dir: str = "data/processed") -> None:
        """
        Save cleaned dataframes to CSV files.
        
        Args:
            output_dir (str): Directory to save processed files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for name, df in self.dataframes.items():
            filename = f"{name}_cleaned.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {filename}")
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of all dataframes.
        
        Returns:
            Dict: Summary information
        """
        summary = {}
        for name, df in self.dataframes.items():
            summary[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.to_dict()
            }
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize and run data loading
    loader = DataLoader()
    dataframes = loader.load_all_data()
    cleaned_data = loader.clean_data()
    loader.save_cleaned_data()
    
    # Print summary
    summary = loader.get_data_summary()
    for name, info in summary.items():
        print(f"\n{name.upper()}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Columns: {info['columns']}")