import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "compas-analysis" / "compas-scores-two-years.csv"
OUTPUT_FILE = ROOT / "data" / "compas_5278_raw.csv"

# ---------------------------------------------------------
# Load original ProPublica two-year COMPAS data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Original dataset:")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print()

# ---------------------------------------------------------
# Reproduce ProPublica filtering
#
# These are the standard filters used in the original
# ProPublica COMPAS analysis and AIF360 preprocessing.
# ---------------------------------------------------------

filtered = df[
    (df["days_b_screening_arrest"] <= 30)
    & (df["days_b_screening_arrest"] >= -30)
    & (df["is_recid"] != -1)
    & (df["c_charge_degree"] != "O")
    & (df["score_text"] != "N/A")
].copy()

print("After ProPublica filtering:")
print(f"Rows: {len(filtered)}")

assert len(filtered) == 6172, (
    f"Expected 6172 rows after ProPublica filters, "
    f"but got {len(filtered)}"
)

# ---------------------------------------------------------
# Restrict race to the two groups used by Farayola et al.
# ---------------------------------------------------------

compas = filtered[
    filtered["race"].isin(["African-American", "Caucasian"])
].copy()

print()
print("After restricting race:")
print(f"Rows: {len(compas)}")

# ---------------------------------------------------------
# Verification checks
# ---------------------------------------------------------

assert len(compas) == 5278, (
    f"Expected 5278 rows, but got {len(compas)}"
)

label_counts = compas["two_year_recid"].value_counts().sort_index()

print()
print("two_year_recid counts:")
print(label_counts)

assert label_counts[0] == 2795, (
    f"Expected 2795 non-recidivists, got {label_counts[0]}"
)

assert label_counts[1] == 2483, (
    f"Expected 2483 recidivists, got {label_counts[1]}"
)

race_counts = compas["race"].value_counts()

print()
print("Race counts:")
print(race_counts)

assert race_counts["African-American"] == 3175, (
    "Unexpected African-American count"
)

assert race_counts["Caucasian"] == 2103, (
    "Unexpected Caucasian count"
)

# ---------------------------------------------------------
# Useful cross-tabulation
# ---------------------------------------------------------

print()
print("Race x two_year_recid:")
print(
    pd.crosstab(
        compas["race"],
        compas["two_year_recid"],
        margins=True
    )
)

# ---------------------------------------------------------
# Save exact filtered cohort
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

compas.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Saved: {OUTPUT_FILE}")
print()
print("COMPAS cohort reconstruction PASSED.")
