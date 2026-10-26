from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "data" / "Cleaned_Durian_Data_ML.xlsx"
    if (BASE_DIR / "data" / "Cleaned_Durian_Data_ML.xlsx").exists()
    else BASE_DIR / "Cleaned_Durian_Data_ML.xlsx"
)

OUTPUT_DIR = BASE_DIR / "results" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Reading: {DATA_PATH}")

xl = pd.ExcelFile(DATA_PATH)
print("\nAvailable sheets:")
print(xl.sheet_names)

df = pd.read_excel(DATA_PATH, sheet_name=0)

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(" ", "", regex=False)
    .str.replace("_", "", regex=False)
    .str.lower()
)

print(f"\nDataset dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

possible_names = {
    "yield": ["yieldha", "yield_ha", "yield"],
    "total_water": ["totalwaterha", "total_water_ha", "totalwater"],
    "productive_trees": [
        "productivetreeha",
        "productive_tree_ha",
        "productivetreesha",
        "productive_trees_ha"
    ],
    "water_intensity_old": [
        "waterintensity",
        "water_intensity"
    ],
    "temperature_growing": [
        "tempgrow",
        "temp_grow",
        "growingseasontemperature"
    ],
    "district": [
        "farmlocation",
        "farm_location",
        "district",
        "location"
    ],
    "tree_age": [
        "treeageyears",
        "tree_age_years",
        "treeage"
    ],
    "fertilizer_total": [
        "totalfertha",
        "total_fert_ha",
        "totalfertilizerha"
    ],
    "fert_intensity_old": [
        "fertintensity",
        "fert_intensity"
    ]
}

def locate_column(candidates, columns):
    for name in candidates:
        if name in columns:
            return name
    return None

resolved = {
    key: locate_column(candidates, df.columns)
    for key, candidates in possible_names.items()
}

print("\nResolved key columns:")
for key, value in resolved.items():
    print(f"{key}: {value}")

essential = [
    "yield",
    "total_water",
    "productive_trees",
    "temperature_growing",
    "district"
]

missing = [name for name in essential if resolved[name] is None]

if missing:
    raise ValueError(
        "\nMissing required variables: "
        + ", ".join(missing)
        + "\nPlease inspect the printed column names and update "
        "the candidate names in possible_names."
    )

yield_col = resolved["yield"]
water_col = resolved["total_water"]
trees_col = resolved["productive_trees"]
temp_col = resolved["temperature_growing"]
district_col = resolved["district"]

numeric_columns = [
    yield_col,
    water_col,
    trees_col,
    temp_col
]

for col in numeric_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["ln_yield_ha"] = np.where(
    df[yield_col] > 0,
    np.log(df[yield_col]),
    np.nan
)

df["water_per_productive_tree"] = np.where(
    df[trees_col] > 0,
    df[water_col] / df[trees_col],
    np.nan
)

if resolved["fertilizer_total"] is not None:
    fert_col = resolved["fertilizer_total"]

    df[fert_col] = (
        df[fert_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    df[fert_col] = pd.to_numeric(df[fert_col], errors="coerce")

    df["fertilizer_per_productive_tree"] = np.where(
        df[trees_col] > 0,
        df[fert_col] / df[trees_col],
        np.nan
    )

print("\nCore-variable descriptive statistics:")
core_cols = [
    yield_col,
    "ln_yield_ha",
    water_col,
    trees_col,
    "water_per_productive_tree",
    temp_col
]
print(df[core_cols].describe().T)

if resolved["water_intensity_old"] is not None:
    old_water_col = resolved["water_intensity_old"]

    df[old_water_col] = pd.to_numeric(
        df[old_water_col],
        errors="coerce"
    )

    df["water_per_yield_check"] = np.where(
        df[yield_col] > 0,
        df[water_col] / df[yield_col],
        np.nan
    )

    water_check = pd.DataFrame({
        "stored_Waterintensity": df[old_water_col],
        "Totalwaterha_divided_by_Yieldha": df["water_per_yield_check"],
        "Water_per_productive_tree": df["water_per_productive_tree"]
    })

    print("\nWater-intensity diagnostic:")
    print(water_check.describe().T)

    print(
        "\nCorrelation between stored Waterintensity "
        "and Totalwaterha/Yieldha:"
    )
    print(
        water_check[
            ["stored_Waterintensity", "Totalwaterha_divided_by_Yieldha"]
        ].corr()
    )

    water_check.to_csv(
        OUTPUT_DIR / "water_intensity_diagnostic.csv",
        index=False
    )

temperature_by_district = (
    df.groupby(district_col, dropna=False)
    .agg(
        number_of_farms=(temp_col, "size"),
        number_of_different_temperature_values=(
            temp_col,
            lambda x: x.nunique(dropna=True)
        ),
        lowest_temperature_c=(temp_col, "min"),
        highest_temperature_c=(temp_col, "max")
    )
    .reset_index()
    .sort_values(district_col)
)

temperature_by_district.to_csv(
    OUTPUT_DIR / "temperature_by_district.csv",
    index=False
)

print("\nTemperature variation by district:")
print(temperature_by_district.to_string(index=False))

temperature_summary = {
    "n_observations": int(df.shape[0]),
    "n_districts": int(df[district_col].nunique(dropna=True)),
    "temperature_variable": temp_col,
    "overall_unique_temperature_values": int(
        df[temp_col].nunique(dropna=True)
    ),
    "overall_min_temperature_c": float(df[temp_col].min()),
    "overall_max_temperature_c": float(df[temp_col].max()),
    "farms_with_valid_yield": int(df[yield_col].notna().sum()),
    "farms_with_valid_water_per_productive_tree": int(
        df["water_per_productive_tree"].notna().sum()
    )
}

with open(
    OUTPUT_DIR / "data_audit_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(temperature_summary, f, indent=4)

df.to_excel(
    OUTPUT_DIR / "durian_analysis_dataset_clean.xlsx",
    index=False
)

print("\nFiles created:")
print(OUTPUT_DIR / "temperature_by_district.csv")
print(OUTPUT_DIR / "water_intensity_diagnostic.csv")
print(OUTPUT_DIR / "data_audit_summary.json")
print(OUTPUT_DIR / "durian_analysis_dataset_clean.xlsx")

print("\nAudit completed successfully.")