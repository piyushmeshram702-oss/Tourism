import pandas as pd

# Just check the basic info for each file
files = ['Transaction.xlsx', 'User.xlsx', 'City.xlsx', 'Country.xlsx', 'Region.xlsx', 'Continent.xlsx', 'Item.xlsx', 'Type.xlsx', 'Mode.xlsx']

for file in files:
    print(f"\n=== {file} ===")
    try:
        df = pd.read_excel(file)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data (first 2 rows):\n{df.head(2)}")
    except Exception as e:
        print(f"Error: {e}")