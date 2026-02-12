import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMerger:
    """Merge and engineer features from multiple tourism datasets."""
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize DataMerger.
        
        Args:
            data_dir (str): Directory containing processed CSV files
        """
        self.data_dir = data_dir
        self.dataframes = {}
        self.merged_data = None
        self.label_encoders = {}
        self.scalers = {}
        
    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all processed CSV files.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of loaded dataframes
        """
        files = [
            'transaction_cleaned.csv',
            'user_cleaned.csv', 
            'city_cleaned.csv',
            'country_cleaned.csv',
            'region_cleaned.csv',
            'continent_cleaned.csv',
            'item_cleaned.csv',
            'type_cleaned.csv',
            'mode_cleaned.csv'
        ]
        
        for filename in files:
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                df_name = filename.replace('_cleaned.csv', '')
                self.dataframes[df_name] = pd.read_csv(file_path)
                logger.info(f"Loaded {filename}: {self.dataframes[df_name].shape}")
            else:
                logger.warning(f"File not found: {file_path}")
                
        return self.dataframes
    
    def merge_all_data(self) -> pd.DataFrame:
        """
        Merge all dataframes into a single master dataset.
        
        Returns:
            pd.DataFrame: Merged master dataset
        """
        if not self.dataframes:
            self.load_processed_data()
            
        # Start with transaction data as the base
        master_df = self.dataframes['transaction'].copy()
        
        # Merge with user data
        master_df = master_df.merge(
            self.dataframes['user'][['UserId', 'ContinentId', 'RegionId', 'CountryId', 'CityId']],
            on='UserId',
            how='left'
        )
        
        # Merge with item (attraction) data
        # Note: Item table has AttractionCityId which should match CityId
        master_df = master_df.merge(
            self.dataframes['item'][['AttractionId', 'AttractionCityId', 'AttractionTypeId', 'Attraction']],
            on='AttractionId',
            how='left'
        )
        
        # Merge with city data (using AttractionCityId from item)
        city_df = self.dataframes['city'][['CityId', 'CityName', 'CountryId']].copy()
        city_df = city_df.rename(columns={'CityId': 'AttractionCityId', 'CityName': 'AttractionCityName', 'CountryId': 'AttractionCountryId'})
        master_df = master_df.merge(city_df, on='AttractionCityId', how='left')
        
        # Merge with country data (user's country)
        country_df = self.dataframes['country'][['CountryId', 'Country']].copy()
        country_df = country_df.rename(columns={'Country': 'UserCountry'})
        master_df = master_df.merge(country_df, on='CountryId', how='left')
        
        # Merge with region data (user's region)
        region_df = self.dataframes['region'][['RegionId', 'Region']].copy()
        region_df = region_df.rename(columns={'Region': 'UserRegion'})
        master_df = master_df.merge(region_df, on='RegionId', how='left')
        
        # Merge with continent data
        master_df = master_df.merge(
            self.dataframes['continent'][['ContinentId', 'Continent']],
            on='ContinentId',
            how='left'
        )
        
        # Merge with type data
        master_df = master_df.merge(
            self.dataframes['type'][['AttractionTypeId', 'AttractionType']],
            on='AttractionTypeId',
            how='left'
        )
        
        # Merge with mode data (map VisitMode to VisitModeId)
        mode_mapping = dict(zip(self.dataframes['mode']['VisitModeId'], 
                               self.dataframes['mode']['VisitMode']))
        master_df['VisitModeName'] = master_df['VisitMode'].map(mode_mapping)
        
        self.merged_data = master_df
        logger.info(f"Merged data shape: {master_df.shape}")
        return master_df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer additional features for machine learning.
        
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        if self.merged_data is None:
            self.merge_all_data()
            
        df = self.merged_data.copy()
        
        # Time-based features
        df['Season'] = df['VisitMonth'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # User demographics features
        df['UserCountryCount'] = df.groupby('UserId')['CountryId'].transform('nunique')
        df['UserAttractionCount'] = df.groupby('UserId')['AttractionId'].transform('count')
        df['UserAvgRating'] = df.groupby('UserId')['Rating'].transform('mean')
        
        # Attraction popularity features
        df['AttractionVisitCount'] = df.groupby('AttractionId')['TransactionId'].transform('count')
        df['AttractionAvgRating'] = df.groupby('AttractionId')['Rating'].transform('mean')
        df['AttractionRatingStd'] = df.groupby('AttractionId')['Rating'].transform('std').fillna(0)
        
        # Visit mode features
        df['ModeAttractionCount'] = df.groupby('VisitMode')['AttractionId'].transform('nunique')
        df['ModeAvgRating'] = df.groupby('VisitMode')['Rating'].transform('mean')
        
        # Encode categorical variables
        categorical_columns = [
            'Continent', 'Region', 'Country', 'AttractionCityName', 
            'AttractionType', 'VisitModeName', 'Season'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                # Handle missing values
                df[col] = df[col].fillna('Unknown')
                
                # Label encoding
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                
        # Scale numerical features
        numerical_columns = [
            'VisitYear', 'VisitMonth', 'UserCountryCount', 'UserAttractionCount',
            'UserAvgRating', 'AttractionVisitCount', 'AttractionAvgRating', 
            'AttractionRatingStd', 'ModeAttractionCount', 'ModeAvgRating'
        ]
        
        for col in numerical_columns:
            if col in df.columns:
                # Handle missing values
                df[col] = df[col].fillna(df[col].median())
                
                # Standard scaling
                scaler = StandardScaler()
                df[f'{col}_scaled'] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        self.merged_data = df
        return df
    
    def prepare_ml_datasets(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare datasets for machine learning models.
        
        Returns:
            Tuple: (X_regression, y_regression, X_classification, y_classification)
        """
        if self.merged_data is None:
            self.engineer_features()
            
        df = self.merged_data.copy()
        
        # Features for both models
        feature_columns = [
            'VisitYear_scaled', 'VisitMonth_scaled', 'UserCountryCount_scaled',
            'UserAttractionCount_scaled', 'UserAvgRating_scaled', 
            'AttractionVisitCount_scaled', 'AttractionAvgRating_scaled',
            'AttractionRatingStd_scaled', 'ModeAttractionCount_scaled', 
            'ModeAvgRating_scaled', 'Continent_encoded', 'Region_encoded',
            'Country_encoded', 'AttractionCityName_encoded', 'AttractionType_encoded',
            'Season_encoded'
        ]
        
        # Remove any columns that don't exist
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Target variables
        y_regression = df['Rating']  # For rating prediction
        y_classification = df['VisitMode']  # For visit mode prediction
        
        # Feature matrix
        X = df[feature_columns]
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        logger.info(f"ML datasets prepared:")
        logger.info(f"  Features shape: {X.shape}")
        logger.info(f"  Regression target shape: {y_regression.shape}")
        logger.info(f"  Classification target shape: {y_classification.shape}")
        
        return X, y_regression, X, y_classification
    
    def save_merged_data(self, filename: str = "merged_tourism_data.csv") -> None:
        """
        Save merged data to CSV file.
        
        Args:
            filename (str): Output filename
        """
        if self.merged_data is not None:
            output_path = os.path.join(self.data_dir, filename)
            self.merged_data.to_csv(output_path, index=False)
            logger.info(f"Merged data saved to {output_path}")
    
    def get_data_info(self) -> Dict:
        """
        Get information about the merged dataset.
        
        Returns:
            Dict: Dataset information
        """
        if self.merged_data is None:
            return {}
            
        info = {
            'shape': self.merged_data.shape,
            'columns': list(self.merged_data.columns),
            'numeric_columns': self.merged_data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.merged_data.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': self.merged_data.isnull().sum().to_dict(),
            'rating_stats': self.merged_data['Rating'].describe().to_dict() if 'Rating' in self.merged_data.columns else {},
            'visit_mode_distribution': self.merged_data['VisitMode'].value_counts().to_dict() if 'VisitMode' in self.merged_data.columns else {}
        }
        return info

# Example usage
if __name__ == "__main__":
    # Initialize and run data merging
    merger = DataMerger()
    dataframes = merger.load_processed_data()
    merged_data = merger.merge_all_data()
    engineered_data = merger.engineer_features()
    merger.save_merged_data()
    
    # Prepare ML datasets
    X_reg, y_reg, X_clf, y_clf = merger.prepare_ml_datasets()
    
    # Print summary
    info = merger.get_data_info()
    print(f"\nMerged Data Summary:")
    print(f"Shape: {info['shape']}")
    print(f"Numeric columns: {len(info['numeric_columns'])}")
    print(f"Categorical columns: {len(info['categorical_columns'])}")
    print(f"Rating stats: {info['rating_stats']}")
    print(f"Visit mode distribution: {info['visit_mode_distribution']}")