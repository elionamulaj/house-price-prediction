from pathlib import Path
from sklearn.datasets import fetch_openml

# Define the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the dataset directory
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Download the Ames Housing dataset
print("Downloading Ames Housing dataset...")

housing = fetch_openml(
    data_id=42165,
    as_frame=True,
    target_column="SalePrice"
)

# Convert the dataset to a DataFrame
df = housing.frame

# Save the dataset as a CSV file
file_path = DATA_DIR / "house_prices.csv"
df.to_csv(file_path, index=False)

# Display dataset information
print("Dataset downloaded successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Saved to: {file_path}")

print("\nFirst five rows:")
print(df.head())
