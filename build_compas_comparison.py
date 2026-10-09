from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"

WIDE_OUTPUT = ROOT / "compas_paper_vs_reproduction.csv"
LONG_OUTPUT = ROOT / "compas_reproduction_analysis_long.csv"


# Paper row order.
# Each tuple:
# model, run number, paper SPD, DI, EOD, PED, Accuracy
CONFIGS = [
    ("Re", 10, 0.097, 1.243, 0.094, 0.031, 0.66),
    ("DIR", 9, 0.284, 2.092, 0.281, 0.211, 0.67),
    ("EGR", 11, 0.036, 1.093, 0.080, 0.088, 0.67),
    ("AL", 26, 0.053, 1.174, 0.115, 0.085, 0.66),
    ("EO", 12, 0.040, 1.125, 0.014, 0.015, 0.64),
    ("ROC", 13, 0.270, 2.100, 0.287, 0.183, 0.68),
    ("Re+EGR", 14, 0.033, 1.086, 0.026, 0.032, 0.66),
    ("DIR+EGR", 15, 0.055, 1.142, 0.038, 0.004, 0.67),
    ("Re+AL", 27, 0.061, 1.168, 0.052, 0.004, 0.67),
    ("DIR+AL", 28, 0.009, 1.020, 0.028, 0.067, 0.67),
    ("Re+EO", 18, 0.034, 1.095, 0.001, 0.002, 0.64),
    ("Re+ROC", 19, 0.081, 1.230, 0.091, 0.002, 0.67),
    ("DIR+EO", 20, 0.032, 1.097, 0.007, 0.005, 0.64),
    ("DIR+ROC", 21, 0.288, 2.160, 0.300, 0.207, 0.68),
    ("EGR+EO", 16, 0.036, 1.095, 0.003, 0.003, 0.65),
    ("EGR+ROC", 17, 0.023, 1.059, 0.015, 0.042, 0.67),
    ("AL+EO", 29, 0.036, 1.099, 0.014, 0.016, 0.66),
    ("AL+ROC", 30, 0.079, 1.159, 0.037, 0.033, 0.68),
    ("Re+EGR+EO", 22, 0.034, 1.088, 0.001, 0.0002, 0.65),
    ("Re+EGR+ROC", 23, 0.023, 1.059, 0.015, 0.042, 0.67),
    ("Re+AL+ROC", 32, 0.079, 1.159, 0.037, 0.033, 0.68),
    ("Re+AL+EO", 31, 0.036, 1.099, 0.014, 0.016, 0.66),
    ("DIR+EGR+ROC", 25, 0.055, 1.142, 0.038, 0.004, 0.67),
    ("DIR+EGR+EO", 24, 0.037, 1.097, 0.002, 0.003, 0.66),
    ("DIR+AL+ROC", 34, 0.030, 1.056, 0.006, 0.032, 0.68),
    ("DIR+AL+EO", 33, 0.038, 1.089, 0.001, 0.001, 0.66),
]


# Maps aggregate_summary_A metric names to short names.
METRICS = {
    "statistical_parity_difference": "SPD",
    "disparate_impact": "DI",
    "equal_opportunity_difference": "EOD",
    "predictive_equality_difference": "PED",
    "accuracy": "Acc",
}


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


def read_run_summary(run_number: int) -> pd.DataFrame:
    run_dir = find_run_directory(run_number)
    csv_path = run_dir / "metadata" / "aggregate_summary_A.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing aggregate summary: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {
        "metric",
        "n",
        "mean",
        "sample_std",
        "min",
        "max",
        "ci95_lower",
        "ci95_upper",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    return df


wide_rows = []
long_rows = []


for (
    model,
    run_number,
    paper_spd,
    paper_di,
    paper_eod,
    paper_ped,
    paper_acc,
) in CONFIGS:

    summary = read_run_summary(run_number)

    paper_values = {
        "SPD": paper_spd,
        "DI": paper_di,
        "EOD": paper_eod,
        "PED": paper_ped,
        "Acc": paper_acc,
    }

    reproduced = {}

    for source_metric, short_metric in METRICS.items():
        match = summary.loc[summary["metric"] == source_metric]

        if len(match) != 1:
            raise ValueError(
                f"Run {run_number:03d}: expected exactly one row "
                f"for {source_metric}, found {len(match)}"
            )

        row = match.iloc[0]

        my_n = int(row["n"])
        my_mean = float(row["mean"])
        my_std = float(row["sample_std"])
        my_min = float(row["min"])
        my_max = float(row["max"])
        my_ci_lower = float(row["ci95_lower"])
        my_ci_upper = float(row["ci95_upper"])

        paper_value = float(paper_values[short_metric])

        signed_difference = my_mean - paper_value
        absolute_difference = abs(signed_difference)
        paper_within_ci95 = my_ci_lower <= paper_value <= my_ci_upper

        # Percent difference is useful, but can be misleading when
        # the paper value is extremely close to zero.
        if paper_value != 0:
            percent_difference = (absolute_difference / abs(paper_value)) * 100
        else:
            percent_difference = None

        reproduced[short_metric] = my_mean

        long_rows.append(
            {
                "Model": model,
                "Run": f"{run_number:03d}",
                "Metric": short_metric,
                "Paper": paper_value,
                "My N": my_n,
                "My Mean": my_mean,
                "My Sample Std": my_std,
                "My Min": my_min,
                "My Max": my_max,
                "My CI95 Lower": my_ci_lower,
                "My CI95 Upper": my_ci_upper,
                "Signed Difference": signed_difference,
                "Absolute Difference": absolute_difference,
                "Paper Within My CI95": paper_within_ci95,
                "Percent Difference": percent_difference,
            }
        )

    # Wide Google-Sheets-friendly table
    wide_rows.append(
        {
            "Model": model,
            "Paper SPD": paper_spd,
            "My SPD": reproduced["SPD"],
            "Paper DI": paper_di,
            "My DI": reproduced["DI"],
            "Paper EOD": paper_eod,
            "My EOD": reproduced["EOD"],
            "Paper PED": paper_ped,
            "My PED": reproduced["PED"],
            "Paper Acc": paper_acc,
            "My Acc": reproduced["Acc"],
        }
    )


wide_df = pd.DataFrame(wide_rows)
long_df = pd.DataFrame(long_rows)


wide_df.to_csv(
    WIDE_OUTPUT,
    index=False,
    float_format="%.15g",
)

long_df.to_csv(
    LONG_OUTPUT,
    index=False,
    float_format="%.15g",
)


print()
print("=" * 80)
print("WIDE COMPARISON TABLE")
print("=" * 80)
print()
print(wide_df.to_string(index=False))

print()
print("=" * 80)
print("ANALYSIS TABLE")
print("=" * 80)
print()
print(long_df.to_string(index=False))

print()
print("Created:")
print(f"  {WIDE_OUTPUT}")
print(f"  {LONG_OUTPUT}")
print()
