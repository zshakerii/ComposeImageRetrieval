import pandas as pd
import pprint

df = pd.read_parquet(
    "./royokongcirr_val/val-00000-of-00002.parquet"
)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst row:")
pprint.pp(df.iloc[0].to_dict())

print("\nTARGET:")
print(type(df.iloc[0]["target"]))
pprint.pp(df.iloc[0]["target"])

print("\nCANDIDATE:")
print(type(df.iloc[0]["candidate"]))
pprint.pp(df.iloc[0]["candidate"])