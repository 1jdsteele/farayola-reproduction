import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "compas_5278_raw.csv"
OUTPUT_FILE = ROOT / "data" / "compas_farayola_numeric.csv"

df = pd.read_csv(INPUT_FILE)

# ---------------------------------------------------------
# Reconstructed 11-feature set used for modeling
# ---------------------------------------------------------

columns = [
    "age",
    "age_cat",
    "sex",
    "race",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_charge_degree",
    "score_text",
    "decile_score",
    "two_year_recid",
]

data = df[columns].copy()

# ---------------------------------------------------------
# Encodings specified by Farayola et al.
# ---------------------------------------------------------

data["age_cat"] = data["age_cat"].map({
    "Less than 25": 0,
    "25 - 45": 1,
    "Greater than 45": 2,
})

data["score_text"] = data["score_text"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2,
})

data["sex"] = data["sex"].map({
    "Female": 0,
    "Male": 1,
})

data["race"] = data["race"].map({
    "African-American": 0,
    "Caucasian": 1,
})

data["c_charge_degree"] = data["c_charge_degree"].map({
    "M": 0,
    "F": 1,
})

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert len(data) == 5278

assert data["two_year_recid"].value_counts().sort_index().to_dict() == {
    0: 2795,
    1: 2483
}

assert data["race"].value_counts().sort_index().to_dict() == {
    0: 3175,
    1: 2103
}

assert data["sex"].isna().sum() == 0
assert data["race"].isna().sum() == 0
assert data["age_cat"].isna().sum() == 0
assert data["score_text"].isna().sum() == 0
assert data["c_charge_degree"].isna().sum() == 0

assert data.isna().sum().sum() == 0

non_numeric = data.select_dtypes(exclude="number").columns.tolist()

assert not non_numeric, (
    f"Non-numeric columns remain: {non_numeric}"
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

data.to_csv(OUTPUT_FILE, index=False)

print("Farayola-style COMPAS dataset created.")
print()
print(f"Shape: {data.shape}")
print()
print("Columns:")
for i, col in enumerate(data.columns, start=1):
    print(f"{i:2}. {col}")
print()
print("Dtypes:")
print(data.dtypes)
print()
print("First five rows:")
print(data.head())
print()
print("Target counts:")
print(data["two_year_recid"].value_counts().sort_index())
print()
print("Race counts:")
print(data["race"].value_counts().sort_index())
print()
print(f"Saved: {OUTPUT_FILE}")
