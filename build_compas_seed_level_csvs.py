from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"

LONG_OUTPUT = ROOT / "compas_seed_level_analysis_long.csv"
WIDE_OUTPUT = ROOT / "compas_seed_level_results_wide.csv"

EXPECTED_SEEDS_PER_RUN = 10


# Run -> model mapping.
# This follows the actual execution order, Runs 009-034.
CONFIGS = [
    (9, "DIR"),
    (10, "Re"),
    (11, "EGR"),
    (12, "EO"),
    (13, "ROC"),
    (14, "Re+EGR"),
    (15, "DIR+EGR"),
    (16, "EGR+EO"),
    (17, "EGR+ROC"),
    (18, "Re+EO"),
    (19, "Re+ROC"),
    (20, "DIR+EO"),
    (21, "DIR+ROC"),
    (22, "Re+EGR+EO"),
    (23, "Re+EGR+ROC"),
    (24, "DIR+EGR+EO"),
    (25, "DIR+EGR+ROC"),
    (26, "AL"),
    (27, "Re+AL"),
    (28, "DIR+AL"),
    (29, "AL+EO"),
    (30, "AL+ROC"),
    (31, "Re+AL+EO"),
    (32, "Re+AL+ROC"),
    (33, "DIR+AL+EO"),
    (34, "DIR+AL+ROC"),
]


# Accepted metric names -> canonical short name
METRIC_ALIASES = {
    "statistical_parity_difference": "SPD",
    "spd": "SPD",
    "disparate_impact": "DI",
    "di": "DI",
    "equal_opportunity_difference": "EOD",
    "eod": "EOD",
    "predictive_equality_difference": "PED",
    "ped": "PED",
    "accuracy": "Acc",
    "acc": "Acc",
    "accuracy_score": "Acc",
}

METRIC_ORDER = ["SPD", "DI", "EOD", "PED", "Acc"]

SEED_COLUMN_CANDIDATES = [
    "seed",
    "random_seed",
    "split_seed",
]


def find_run_directory(run_number: int) -> Path:
    matches = sorted(RUNS_DIR.glob(f"run_{run_number:03d}_*"))

    if len(matches) == 0:
        raise FileNotFoundError(f"Could not find directory for run {run_number:03d}")

    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple directories for run {run_number:03d}: "
            f"{[p.name for p in matches]}"
        )

    return matches[0]


def find_seed_column(df: pd.DataFrame) -> str:
    lower_to_original = {str(column).strip().lower(): column for column in df.columns}

    for candidate in SEED_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    raise ValueError(
        "Could not identify the seed column. " f"Columns found: {list(df.columns)}"
    )


def normalize_metric(metric_name: str) -> str | None:
    normalized = str(metric_name).strip().lower()
    return METRIC_ALIASES.get(normalized)


def read_seed_results(run_number: int, model: str) -> pd.DataFrame:
    run_dir = find_run_directory(run_number)
    csv_path = run_dir / "metadata" / "per_seed_results_A.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing per-seed results file: {csv_path}")

    df = pd.read_csv(csv_path)

    seed_col = find_seed_column(df)

    if df[seed_col].isna().any():
        raise ValueError(f"{csv_path} contains missing seed values.")

    # ---------------------------------------------------------
    # FORMAT 1:
    # Wide input:
    #
    # seed,accuracy,disparate_impact,...
    # ---------------------------------------------------------

    column_metric_map = {}

    for column in df.columns:
        canonical = normalize_metric(column)

        if canonical is not None:
            column_metric_map[column] = canonical

    if set(column_metric_map.values()) == set(METRIC_ORDER):
        records = []

        for _, row in df.iterrows():
            seed = row[seed_col]

            for source_column, metric in column_metric_map.items():
                records.append(
                    {
                        "Model": model,
                        "Run": f"{run_number:03d}",
                        "Seed": seed,
                        "Metric": metric,
                        "Value": float(row[source_column]),
                    }
                )

        result = pd.DataFrame(records)

    # ---------------------------------------------------------
    # FORMAT 2:
    # Long input:
    #
    # seed,metric,value
    # ---------------------------------------------------------

    elif "metric" in [str(c).strip().lower() for c in df.columns]:
        lower_to_original = {
            str(column).strip().lower(): column for column in df.columns
        }

        metric_col = lower_to_original["metric"]

        value_col = None

        for candidate in ["value", "score", "metric_value"]:
            if candidate in lower_to_original:
                value_col = lower_to_original[candidate]
                break

        if value_col is None:
            raise ValueError(
                f"{csv_path} appears to be long format, "
                "but no value column was found. "
                f"Columns: {list(df.columns)}"
            )

        records = []

        for _, row in df.iterrows():
            metric = normalize_metric(row[metric_col])

            # Ignore unrelated metrics if any exist.
            if metric is None:
                continue

            records.append(
                {
                    "Model": model,
                    "Run": f"{run_number:03d}",
                    "Seed": row[seed_col],
                    "Metric": metric,
                    "Value": float(row[value_col]),
                }
            )

        result = pd.DataFrame(records)

    else:
        raise ValueError(
            f"Could not determine the format of {csv_path}.\n"
            f"Columns found: {list(df.columns)}"
        )

    validate_run_seed_results(
        result=result,
        run_number=run_number,
        model=model,
        csv_path=csv_path,
    )

    return result


def validate_run_seed_results(
    result: pd.DataFrame,
    run_number: int,
    model: str,
    csv_path: Path,
) -> None:

    seed_count = result["Seed"].nunique()

    if seed_count != EXPECTED_SEEDS_PER_RUN:
        raise ValueError(
            f"Run {run_number:03d} ({model}) has {seed_count} unique seeds; "
            f"expected {EXPECTED_SEEDS_PER_RUN}.\n"
            f"Source: {csv_path}"
        )

    duplicates = result.duplicated(
        subset=["Seed", "Metric"],
        keep=False,
    )

    if duplicates.any():
        bad_rows = result.loc[
            duplicates,
            ["Seed", "Metric"],
        ]

        raise ValueError(
            f"Run {run_number:03d} ({model}) contains duplicate "
            f"Seed × Metric rows:\n{bad_rows}"
        )

    metrics_per_seed = result.groupby("Seed")["Metric"].apply(set)

    expected_metrics = set(METRIC_ORDER)

    for seed, metrics in metrics_per_seed.items():
        if metrics != expected_metrics:
            raise ValueError(
                f"Run {run_number:03d} ({model}), seed {seed}, "
                f"has metrics {sorted(metrics)}; "
                f"expected {sorted(expected_metrics)}"
            )


# =============================================================
# READ ALL 26 RUNS
# =============================================================

all_frames = []

seed_sets = {}

for run_number, model in CONFIGS:
    print(f"Reading Run {run_number:03d}: {model}")

    frame = read_seed_results(
        run_number=run_number,
        model=model,
    )

    all_frames.append(frame)

    seed_sets[run_number] = set(frame["Seed"].unique())


long_df = pd.concat(
    all_frames,
    ignore_index=True,
)


# =============================================================
# VERIFY ALL CONFIGURATIONS USE THE SAME SEEDS
# =============================================================

reference_run = CONFIGS[0][0]
reference_seeds = seed_sets[reference_run]

for run_number, model in CONFIGS[1:]:
    if seed_sets[run_number] != reference_seeds:
        missing = reference_seeds - seed_sets[run_number]
        extra = seed_sets[run_number] - reference_seeds

        raise ValueError(
            f"Seed mismatch in Run {run_number:03d} ({model}).\n"
            f"Missing relative to Run {reference_run:03d}: "
            f"{sorted(missing)}\n"
            f"Extra relative to Run {reference_run:03d}: "
            f"{sorted(extra)}\n"
            "\nPaired analysis should not proceed until this is understood."
        )


# =============================================================
# SORT LONG DATA
# =============================================================

run_order = {
    f"{run_number:03d}": index for index, (run_number, _) in enumerate(CONFIGS)
}

metric_order = {metric: index for index, metric in enumerate(METRIC_ORDER)}

long_df["_run_order"] = long_df["Run"].map(run_order)
long_df["_metric_order"] = long_df["Metric"].map(metric_order)

long_df = (
    long_df.sort_values(
        by=[
            "_run_order",
            "Seed",
            "_metric_order",
        ]
    )
    .drop(
        columns=[
            "_run_order",
            "_metric_order",
        ]
    )
    .reset_index(drop=True)
)


# =============================================================
# CREATE WIDE VERSION
# =============================================================

wide_df = long_df.pivot(
    index=["Model", "Run", "Seed"],
    columns="Metric",
    values="Value",
).reset_index()

wide_df.columns.name = None

wide_df = wide_df[
    [
        "Model",
        "Run",
        "Seed",
        "SPD",
        "DI",
        "EOD",
        "PED",
        "Acc",
    ]
]

wide_df["_run_order"] = wide_df["Run"].map(run_order)

wide_df = (
    wide_df.sort_values(
        by=[
            "_run_order",
            "Seed",
        ]
    )
    .drop(columns="_run_order")
    .reset_index(drop=True)
)


# =============================================================
# FINAL VALIDATION
# =============================================================

expected_long_rows = len(CONFIGS) * EXPECTED_SEEDS_PER_RUN * len(METRIC_ORDER)

expected_wide_rows = len(CONFIGS) * EXPECTED_SEEDS_PER_RUN

if len(long_df) != expected_long_rows:
    raise ValueError(
        f"Expected {expected_long_rows} long-format rows, "
        f"but produced {len(long_df)}."
    )

if len(wide_df) != expected_wide_rows:
    raise ValueError(
        f"Expected {expected_wide_rows} wide-format rows, "
        f"but produced {len(wide_df)}."
    )


# =============================================================
# WRITE OUTPUT
# =============================================================

long_df.to_csv(
    LONG_OUTPUT,
    index=False,
    float_format="%.15g",
)

wide_df.to_csv(
    WIDE_OUTPUT,
    index=False,
    float_format="%.15g",
)


print()
print("=" * 72)
print("SEED-LEVEL EXPORT COMPLETE")
print("=" * 72)

print()
print(f"Runs processed:       {len(CONFIGS)}")
print(f"Seeds per run:        {EXPECTED_SEEDS_PER_RUN}")
print(f"Metrics per seed:     {len(METRIC_ORDER)}")
print(f"Long-format rows:     {len(long_df)}")
print(f"Wide-format rows:     {len(wide_df)}")

print()
print("Shared seed set:")
print(sorted(reference_seeds))

print()
print("Created:")
print(f"  {LONG_OUTPUT}")
print(f"  {WIDE_OUTPUT}")

print()
print("Long-format preview:")
print(long_df.head(15).to_string(index=False))

print()
print("Wide-format preview:")
print(wide_df.head(10).to_string(index=False))
