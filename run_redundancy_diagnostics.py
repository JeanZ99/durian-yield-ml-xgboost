from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "data" / "Cleaned_Durian_Data_ML.xlsx"
    if (BASE_DIR / "data" / "Cleaned_Durian_Data_ML.xlsx").exists()
    else BASE_DIR / "Cleaned_Durian_Data_ML.xlsx"
)

OUTPUT_DIR = BASE_DIR / "results" / "output_redundancy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA_PATH, sheet_name="Data")
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(" ", "", regex=False)
    .str.replace("_", "", regex=False)
    .str.lower()
)

numeric_raw = [
    "yieldha",
    "totalwaterha",
    "totalfertha",
    "totalorgfertha",
    "productivetreeha",
    "treedensityha",
    "treeageyears",
    "pesticidesha",
    "experienceyears",
    "extensionvisits",
    "age",
    "education",
    "familymember",
    "ph",
    "om",
    "n",
    "p",
    "k",
    "ec",
    "soilfertilityindex",
    "pdt",
    "gapscoreoverall"
]

for col in numeric_raw:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

df["water_per_productive_tree"] = np.where(
    df["productivetreeha"] > 0,
    df["totalwaterha"] / df["productivetreeha"],
    np.nan
)

df["fertilizer_per_productive_tree"] = np.where(
    df["productivetreeha"] > 0,
    df["totalfertha"] / df["productivetreeha"],
    np.nan
)

df["organic_fertilizer_per_productive_tree"] = np.where(
    df["productivetreeha"] > 0,
    df["totalorgfertha"] / df["productivetreeha"],
    np.nan
)

df["productive_tree_share"] = np.where(
    df["treedensityha"] > 0,
    df["productivetreeha"] / df["treedensityha"],
    np.nan
)

features = [
    "totalwaterha",
    "water_per_productive_tree",
    "totalfertha",
    "fertilizer_per_productive_tree",
    "totalorgfertha",
    "organic_fertilizer_per_productive_tree",
    "pesticidesha",
    "treeageyears",
    "treedensityha",
    "productivetreeha",
    "productive_tree_share",
    "gapscoreoverall",
    "ipm",
    "soilimprovement",
    "pdt",
    "plantmonthong",
    "plantkradumthong",
    "plantkanyao",
    "experienceyears",
    "training",
    "extensionvisits",
    "education",
    "age",
    "familymember",
    "groupmember",
    "creditaccess",
    "ph",
    "om",
    "n",
    "p",
    "k",
    "ec",
    "soilfertilityindex"
]

features = [
    col for col in features
    if col in df.columns and df[col].nunique(dropna=True) > 1
]

analysis = df[features].copy()

for col in features:
    analysis[col] = pd.to_numeric(
        analysis[col],
        errors="coerce"
    )

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(
    imputer.fit_transform(analysis),
    columns=features,
    index=analysis.index
)

pearson = X.corr(method="pearson")
spearman = X.corr(method="spearman")

pearson.to_csv(
    OUTPUT_DIR / "correlation_pearson.csv"
)
spearman.to_csv(
    OUTPUT_DIR / "correlation_spearman.csv"
)

def long_pairs(correlation_matrix, method):
    mask = np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)

    pairs = (
        correlation_matrix.where(mask)
        .stack()
        .reset_index()
    )

    pairs.columns = [
        "feature_1",
        "feature_2",
        f"{method}_r"
    ]

    pairs["abs_r"] = pairs[
        f"{method}_r"
    ].abs()

    pairs["redundancy_class"] = pd.cut(
        pairs["abs_r"],
        bins=[-0.01, 0.50, 0.70, 0.85, 1.00],
        labels=[
            "low (<0.50)",
            "moderate (0.50–0.69)",
            "high (0.70–0.84)",
            "near-duplicate (>=0.85)"
        ]
    )

    return pairs.sort_values(
        "abs_r",
        ascending=False
    ).reset_index(drop=True)

pearson_pairs = long_pairs(
    pearson,
    "pearson"
)

spearman_pairs = long_pairs(
    spearman,
    "spearman"
)

pearson_pairs.to_csv(
    OUTPUT_DIR / "correlation_pairs_pearson.csv",
    index=False
)

spearman_pairs.to_csv(
    OUTPUT_DIR / "correlation_pairs_spearman.csv",
    index=False
)

focus_variables = [
    "treeageyears",
    "treedensityha",
    "productivetreeha",
    "productive_tree_share",
    "totalwaterha",
    "water_per_productive_tree",
    "totalfertha",
    "fertilizer_per_productive_tree",
    "totalorgfertha",
    "organic_fertilizer_per_productive_tree",
    "experienceyears"
]

focus_variables = [
    col for col in focus_variables
    if col in features
]

focus_pearson = pearson.loc[
    focus_variables,
    focus_variables
]

focus_spearman = spearman.loc[
    focus_variables,
    focus_variables
]

focus_pearson.to_csv(
    OUTPUT_DIR / "focus_correlation_pearson.csv"
)

focus_spearman.to_csv(
    OUTPUT_DIR / "focus_correlation_spearman.csv"
)

high_pairs = pearson_pairs.loc[
    pearson_pairs["abs_r"] >= 0.70
].copy()

high_pairs.to_csv(
    OUTPUT_DIR / "high_redundancy_pairs.csv",
    index=False
)

plt.figure(figsize=(16, 13))
sns.heatmap(
    pearson,
    cmap="vlag",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"label": "Pearson correlation"}
)
plt.title("Pearson correlations among leakage-free predictors")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "correlation_heatmap_all_predictors.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(
    focus_pearson,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"label": "Pearson correlation"}
)
plt.title("Correlation diagnostics for key predictors")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "correlation_heatmap_focus_predictors.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\n" + "=" * 80)
print("REDUNDANCY DIAGNOSTICS")
print("=" * 80)

print("\nHigh-correlation Pearson pairs (|r| >= 0.70):")
if len(high_pairs) == 0:
    print("No pairs met the |r| >= 0.70 threshold.")
else:
    print(
        high_pairs[
            [
                "feature_1",
                "feature_2",
                "pearson_r",
                "abs_r",
                "redundancy_class"
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

print("\nFocused Pearson correlation matrix:")
print(
    focus_pearson
    .round(4)
    .to_string()
)

print("\nFiles saved:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(path.name)

print("\nCompleted.")