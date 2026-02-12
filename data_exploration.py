import pandas as pd

# Load all datasets
files = {
    'Transaction': 'Transaction.xlsx',
    'User': 'User.xlsx',
    'City': 'City.xlsx',
    'Country': 'Country.xlsx',
    'Region': 'Region.xlsx',
    'Continent': 'Continent.xlsx',
    'Item': 'Item.xlsx',
    'Type': 'Type.xlsx',
    'Mode': 'Mode.xlsx'
}

dataframes = {}
for name, file in files.items():
    try:
        df = pd.read_excel(file)
        dataframes[name] = df
        print(f"=== {name} ===")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(f"Data types:\n{df.dtypes}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"First 3 rows:\n{df.head(3)}")
        print("-" * 50)
        print()
    except Exception as e:
        print(f"Error loading {name}: {e}")

# Check relationships
print("=== RELATIONSHIP ANALYSIS ===")
print("Transaction keys:")
print(f"UserId unique values: {dataframes['Transaction']['UserId'].nunique()}")
print(f"AttractionId unique values: {dataframes['Transaction']['AttractionId'].nunique()}")
print(f"VisitModeId unique values: {dataframes['Transaction']['VisitModeId'].nunique()}")

print("\nUser keys:")
print(f"UserId unique values: {dataframes['User']['UserId'].nunique()}")

print("\nItem keys:")
print(f"AttractionId unique values: {dataframes['Item']['AttractionId'].nunique()}")
print(f"CityId unique values: {dataframes['Item']['CityId'].nunique()}")
print(f"AttractionTypeId unique values: {dataframes['Item']['AttractionTypeId'].nunique()}")

print("\nCity keys:")
print(f"CityId unique values: {dataframes['City']['CityId'].nunique()}")
print(f"CountryId unique values: {dataframes['City']['CountryId'].nunique()}")

print("\nCountry keys:")
print(f"CountryId unique values: {dataframes['Country']['CountryId'].nunique()}")
print(f"RegionId unique values: {dataframes['Country']['RegionId'].nunique()}")

print("\nRegion keys:")
print(f"RegionId unique values: {dataframes['Region']['RegionId'].nunique()}")
print(f"ContinentId unique values: {dataframes['Region']['ContinentId'].nunique()}")

print("\nContinent keys:")
print(f"ContinentId unique values: {dataframes['Continent']['ContinentId'].nunique()}")

print("\nType keys:")
print(f"AttractionTypeId unique values: {dataframes['Type']['AttractionTypeId'].nunique()}")

print("\nMode keys:")
print(f"VisitModeId unique values: {dataframes['Mode']['VisitModeId'].nunique()}")