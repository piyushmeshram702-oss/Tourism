import pandas as pd
import traceback

def inspect_file(filename):
    print(f"\n=== {filename} ===")
    try:
        df = pd.read_excel(filename)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"First 2 rows:\n{df.head(2)}")
        print(f"Missing values:\n{df.isnull().sum()}")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        traceback.print_exc()
        return None

# Files to inspect
files = ['Transaction.xlsx', 'User.xlsx', 'City.xlsx', 'Country.xlsx', 'Region.xlsx', 'Continent.xlsx', 'Item.xlsx', 'Type.xlsx', 'Mode.xlsx']

dataframes = {}
for file in files:
    dataframes[file.replace('.xlsx', '')] = inspect_file(file)

# Show relationship analysis if all files loaded successfully
if all(df is not None for df in dataframes.values()):
    print("\n" + "="*50)
    print("RELATIONSHIP ANALYSIS")
    print("="*50)
    
    if 'Transaction' in dataframes and 'Transaction' in dataframes:
        trans_df = dataframes['Transaction']
        print(f"Transaction - UserId unique: {trans_df['UserId'].nunique() if 'UserId' in trans_df.columns else 'N/A'}")
        print(f"Transaction - AttractionId unique: {trans_df['AttractionId'].nunique() if 'AttractionId' in trans_df.columns else 'N/A'}")
        print(f"Transaction - VisitModeId unique: {trans_df['VisitModeId'].nunique() if 'VisitModeId' in trans_df.columns else 'N/A'}")
        
    if 'User' in dataframes:
        user_df = dataframes['User']
        print(f"User - UserId unique: {user_df['UserId'].nunique() if 'UserId' in user_df.columns else 'N/A'}")
        
    if 'Item' in dataframes:
        item_df = dataframes['Item']
        print(f"Item - AttractionId unique: {item_df['AttractionId'].nunique() if 'AttractionId' in item_df.columns else 'N/A'}")
        print(f"Item - CityId unique: {item_df['CityId'].nunique() if 'CityId' in item_df.columns else 'N/A'}")