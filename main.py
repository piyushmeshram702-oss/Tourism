import pandas as pd
import os
from src.data_loader import DataLoader
from src.data_merger import DataMerger

def create_merged_dataset():
    """Create the merged dataset if it doesn't exist."""
    
    # Check if merged data already exists
    if os.path.exists('data/processed/merged_tourism_data.csv'):
        print("Merged data already exists")
        return pd.read_csv('data/processed/merged_tourism_data.csv')
    
    print("Creating merged dataset...")
    
    # Load and clean data
    loader = DataLoader()
    dataframes = loader.load_all_data()
    cleaned_data = loader.clean_data()
    loader.save_cleaned_data()
    
    # Merge data
    merger = DataMerger()
    dataframes = merger.load_processed_data()
    merged_data = merger.merge_all_data()
    engineered_data = merger.engineer_features()
    merger.save_merged_data()
    
    print(f"Merged dataset created with shape: {merged_data.shape}")
    return merged_data

if __name__ == "__main__":
    # Create merged dataset
    df = create_merged_dataset()
    print("Dataset ready for analysis!")
    print(f"Columns: {list(df.columns)}")