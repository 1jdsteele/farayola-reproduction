from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# Command line
# =====================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tag",
    default="A",
    help="Run 036 result tag, e.g. A",
)

args = parser.parse_args()
TAG = args.tag


# =====================================================================
# Paths
# =====================================================================

SCRIPT_PATH = Path(__file__).resolve()
RUN036_ROOT = SCRIPT_PATH.parents[1]
RUNS_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[3]

METADATA_DIR = RUN036_ROOT / "metadata"
ANALYSIS_DIR = RUN036_ROOT / "analysis"
BOXPLOT_DIR = ANALYSIS_DIR / "boxplots"

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BOXPLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# Metrics
# =====================================================================

METRICS = [
    "accuracy",
    "disparate_impact",
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "predictive_equality_difference",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "disparate_impact": "DI",
    "statistical_parity_difference": "SPD",
    "equal_opportunity_difference": "EOD",
    "predictive_equality_difference": "PED",
}

DIFFERENCE_METRICS = {
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "predictive_equality_difference",
}


# =====================================================================
# Run 036 configurations
# =====================================================================

CONFIGS = {
    "NONE": {
        "fixed_file":
            METADATA_DIR
            / f"none_per_seed_results_{TAG}.csv",

        "old_file":
            RUNS_ROOT
            / "run_007_none_race_stratified_10seed"
            / "metadata"
            / "per_seed_results_B.csv",
    },

    "DIR+EGR+EO": {
        "fixed_file":
            METADATA_DIR
            / f"dir_egr_eo_per_seed_results_{TAG}.csv",

        "old_file":
            RUNS_ROOT
            / "run_024_dir_egr_eo"
            / "metadata"
            / "per_seed_results_A.csv",
    },

    "AL": {
        "fixed_file":
            METADATA_DIR
            / f"al_per_seed_results_{TAG}.csv",

        "old_file":
            RUNS_ROOT
            / "run_026_individual_al"
            / "metadata"
            / "per_seed_results_A.csv",
    },

    "DIR+AL+EO": {
        "fixed_file":
            METADATA_DIR
            / f"dir_al_eo_per_seed_results_{TAG}.csv",

        "old_file":
            RUNS_ROOT
            / "run_033_dir_al_eo"
            / "metadata"
            / "per_seed_results_A.csv",
    },

    "DIR+AL+ROC": {
        "fixed_file":
            METADATA_DIR
            / f"dir_al_roc_per_seed_results_{TAG}.csv",

        "old_file":
            RUNS_ROOT
            / "run_034_dir_al_roc"
            / "metadata"
            / "per_seed_results_A.csv",
    },
}


# =====================================================================
# Paper values
#
# Values already recorded in the project's paper-vs-reproduction table.
# These are used only for comparison; no parameter selection or tuning
# is performed against them.
# =====================================================================

PAPER = {
    "NONE": {
        "accuracy": 0.68,
        "disparate_impact": 2.13,
        "statistical_parity_difference": 0.25,
        "equal_opportunity_difference": 0.025,
        "predictive_equality_difference": 0.18,
    },

    "DIR+EGR+EO": {
        "accuracy": 0.66,
        "disparate_impact": 1.097,
        "statistical_parity_difference": 0.037,
        "equal_opportunity_difference": 0.002,
        "predictive_equality_difference": 0.003,
    },

    "AL": {
        "accuracy": 0.66,
        "disparate_impact": 1.174,
        "statistical_parity_difference": 0.053,
        "equal_opportunity_difference": 0.115,
        "predictive_equality_difference": 0.085,
    },

    "DIR+AL+EO": {
        "accuracy": 0.66,
        "disparate_impact": 1.089,
        "statistical_parity_difference": 0.038,
        "equal_opportunity_difference": 0.001,
        "predictive_equality_difference": 0.001,
    },

    "DIR+AL+ROC": {
        "accuracy": 0.68,
        "disparate_impact": 1.056,
        "statistical_parity_difference": 0.030,
        "equal_opportunity_difference": 0.006,
        "predictive_equality_difference": 0.032,
    },
}


# =====================================================================
# Column normalization
# =====================================================================

ALIASES = {
    "seed": [
        "seed",
        "algorithm_seed",
    ],

    "accuracy": [
        "accuracy",
        "acc",
    ],

    "disparate_impact": [
        "disparate_impact",
        "di",
    ],

    "statistical_parity_difference": [
        "statistical_parity_difference",
        "spd",
    ],

    "equal_opportunity_difference": [
        "equal_opportunity_difference",
        "eod",
    ],

    "predictive_equality_difference": [
        "predictive_equality_difference",
        "ped",
        "false_positive_rate_difference",
    ],
}


def normalized_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def normalize_result_columns(df):
    df = df.copy()

    existing = {
        normalized_name(column): column
        for column in df.columns
    }

    rename = {}

    for canonical, aliases in ALIASES.items():

        for alias in aliases:

            key = normalized_name(alias)

            if key in existing:
                rename[
                    existing[key]
                ] = canonical
                break

    return df.rename(
        columns=rename
    )


# =====================================================================
# Validation helpers
# =====================================================================

def require_metric_columns(df, description):
    missing = [
        metric
        for metric in METRICS
        if metric not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{description} is missing metric columns: "
            f"{missing}"
        )


def summarize(values):
    values = pd.Series(
        values,
        dtype=float,
    ).dropna()

    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "sample_std": float(
            values.std(ddof=1)
        ) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median()),
    }


def safe_ratio(numerator, denominator):
    if np.isclose(
        denominator,
        0.0,
        rtol=0.0,
        atol=1e-15,
    ):
        return np.nan

    return float(
        numerator / denominator
    )


# =====================================================================
# Locate old per-seed result files robustly
# =====================================================================

def find_old_per_seed_csv(run_dir):

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Old run directory not found: {run_dir}"
        )

    candidates = []

    for path in sorted(
        run_dir.rglob("*.csv")
    ):

        lower_name = path.name.lower()

        if any(
            token in lower_name
            for token in [
                "summary",
                "aggregate",
                "manifest",
            ]
        ):
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        df = normalize_result_columns(
            df
        )

        if "seed" not in df.columns:
            continue

        if not all(
            metric in df.columns
            for metric in METRICS
        ):
            continue

        # Prior sensitivity runs should have approximately
        # ten seed-level rows.
        if len(df) < 8:
            continue

        score = 0

        if "per_seed" in lower_name:
            score += 10

        if "results" in lower_name:
            score += 5

        if "10seed" in lower_name:
            score += 2

        score += min(
            len(df),
            20,
        ) / 100.0

        candidates.append(
            (
                score,
                path,
                df,
            )
        )

    if not candidates:
        raise RuntimeError(
            "Could not automatically identify a "
            f"seed-level result CSV in {run_dir}"
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            str(item[1]),
        ),
        reverse=True,
    )

    _, selected_path, selected_df = (
        candidates[0]
    )

    return (
        selected_path,
        selected_df,
    )


# =====================================================================
# Load Run 036 fixed-split results
# =====================================================================

fixed_frames = {}
old_frames = {}
old_paths = {}

for config, info in CONFIGS.items():

    fixed_path = info[
        "fixed_file"
    ]

    if not fixed_path.exists():
        raise FileNotFoundError(
            f"Run 036 file missing for {config}: "
            f"{fixed_path}"
        )

    fixed_df = pd.read_csv(
        fixed_path
    )

    fixed_df = (
        normalize_result_columns(
            fixed_df
        )
    )

    require_metric_columns(
        fixed_df,
        f"Run 036 {config}",
    )

    if "seed" not in fixed_df.columns:
        raise RuntimeError(
            f"Run 036 {config} lacks a seed column."
        )

    if len(fixed_df) != 20:
        raise RuntimeError(
            f"Expected 20 Run 036 rows for {config}; "
            f"found {len(fixed_df)}."
        )

    fixed_df = fixed_df.copy()

    fixed_df[
        "configuration"
    ] = config

    fixed_df[
        "experiment_design"
    ] = "fixed_split"

    fixed_frames[
        config
    ] = fixed_df

    old_path = info["old_file"]

    if not old_path.exists():
        raise FileNotFoundError(
            f"Historical seed-level file missing for {config}: "
            f"{old_path}"
        )

    old_df = pd.read_csv(
        old_path
    )

    old_df = normalize_result_columns(
        old_df
    )

    require_metric_columns(
        old_df,
        f"Old {config}",
    )

    old_df = old_df.copy()

    old_df[
        "configuration"
    ] = config

    old_df[
        "experiment_design"
    ] = "changing_split"

    old_frames[
        config
    ] = old_df

    old_paths[
        config
    ] = old_path


# =====================================================================
# Combined Run 036 seed-level table
# =====================================================================

all_fixed = pd.concat(
    [
        fixed_frames[
            config
        ]
        for config in CONFIGS
    ],
    ignore_index=True,
)

ALL_FIXED_PATH = (
    ANALYSIS_DIR
    / "run_036_all_per_seed.csv"
)

all_fixed.to_csv(
    ALL_FIXED_PATH,
    index=False,
    float_format="%.15g",
)


# =====================================================================
# Run 036 fixed-split summary
# =====================================================================

summary_rows = []

for config in CONFIGS:

    df = fixed_frames[
        config
    ]

    for metric in METRICS:

        stats = summarize(
            df[metric]
        )

        summary_rows.append({
            "configuration":
                config,

            "metric":
                metric,

            **stats,
        })

run036_summary = pd.DataFrame(
    summary_rows
)

SUMMARY_PATH = (
    ANALYSIS_DIR
    / "run_036_summary.csv"
)

run036_summary.to_csv(
    SUMMARY_PATH,
    index=False,
    float_format="%.15g",
)


# =====================================================================
# Old changing-split vs Run 036 fixed-split comparison
# =====================================================================

old_vs_fixed_rows = []

for config in CONFIGS:

    old_df = old_frames[
        config
    ]

    fixed_df = fixed_frames[
        config
    ]

    for metric in METRICS:

        old_stats = summarize(
            old_df[metric]
        )

        fixed_stats = summarize(
            fixed_df[metric]
        )

        ratio = safe_ratio(
            fixed_stats[
                "sample_std"
            ],
            old_stats[
                "sample_std"
            ],
        )

        if np.isnan(ratio):
            reduction = np.nan
        else:
            reduction = float(
                100.0
                * (
                    1.0
                    - ratio
                )
            )

        old_vs_fixed_rows.append({
            "configuration":
                config,

            "metric":
                metric,

            "old_source_file":
                str(
                    old_paths[
                        config
                    ].relative_to(
                        REPO_ROOT
                    )
                ),

            "old_n":
                old_stats["n"],

            "old_mean":
                old_stats["mean"],

            "old_sample_std":
                old_stats[
                    "sample_std"
                ],

            "old_min":
                old_stats["min"],

            "old_max":
                old_stats["max"],

            "fixed_n":
                fixed_stats["n"],

            "fixed_mean":
                fixed_stats["mean"],

            "fixed_sample_std":
                fixed_stats[
                    "sample_std"
                ],

            "fixed_min":
                fixed_stats["min"],

            "fixed_max":
                fixed_stats["max"],

            "mean_shift_fixed_minus_old":
                fixed_stats["mean"]
                - old_stats["mean"],

            "sd_ratio_fixed_over_old":
                ratio,

            "sd_reduction_percent":
                reduction,
        })

old_vs_fixed = pd.DataFrame(
    old_vs_fixed_rows
)

OLD_FIXED_PATH = (
    ANALYSIS_DIR
    / "run_036_old_vs_fixed_split.csv"
)

old_vs_fixed.to_csv(
    OLD_FIXED_PATH,
    index=False,
    float_format="%.15g",
)


# =====================================================================
# Paper comparison
#
# For SPD/EOD/PED we report both:
#
#   1. whether the paper value falls in the raw signed range;
#   2. whether it falls in the range of absolute per-seed disparities.
#
# This is deliberate because exact Table 1 sign/absolute aggregation
# semantics remain underdocumented.
# =====================================================================

paper_rows = []

for config in CONFIGS:

    df = fixed_frames[
        config
    ]

    for metric in METRICS:

        values = pd.Series(
            df[metric],
            dtype=float,
        )

        paper_value = float(
            PAPER[
                config
            ][metric]
        )

        observed_min = float(
            values.min()
        )

        observed_max = float(
            values.max()
        )

        paper_in_raw_range = bool(
            observed_min
            <= paper_value
            <= observed_max
        )

        row = {
            "configuration":
                config,

            "metric":
                metric,

            "paper_value":
                paper_value,

            "fixed_mean":
                float(
                    values.mean()
                ),

            "fixed_sample_std":
                float(
                    values.std(ddof=1)
                ),

            "fixed_min":
                observed_min,

            "fixed_max":
                observed_max,

            "paper_minus_fixed_mean":
                paper_value
                - float(
                    values.mean()
                ),

            "paper_in_raw_observed_range":
                paper_in_raw_range,

            "abs_range_applicable":
                metric
                in DIFFERENCE_METRICS,

            "abs_fixed_min":
                np.nan,

            "abs_fixed_max":
                np.nan,

            "paper_in_abs_observed_range":
                np.nan,
        }

        if metric in DIFFERENCE_METRICS:

            abs_values = (
                values.abs()
            )

            abs_min = float(
                abs_values.min()
            )

            abs_max = float(
                abs_values.max()
            )

            row[
                "abs_fixed_min"
            ] = abs_min

            row[
                "abs_fixed_max"
            ] = abs_max

            row[
                "paper_in_abs_observed_range"
            ] = bool(
                abs_min
                <= paper_value
                <= abs_max
            )

        paper_rows.append(
            row
        )

paper_comparison = pd.DataFrame(
    paper_rows
)

PAPER_PATH = (
    ANALYSIS_DIR
    / "run_036_paper_comparison.csv"
)

paper_comparison.to_csv(
    PAPER_PATH,
    index=False,
    float_format="%.15g",
)


# =====================================================================
# 36-D vs 36-E:
# identical DIR+AL upstream model, EO vs ROC postprocessing
# =====================================================================

d_df = fixed_frames[
    "DIR+AL+EO"
].sort_values(
    "seed"
).reset_index(drop=True)

e_df = fixed_frames[
    "DIR+AL+ROC"
].sort_values(
    "seed"
).reset_index(drop=True)

assert np.array_equal(
    d_df["seed"].to_numpy(),
    e_df["seed"].to_numpy(),
)

# Exact hash validation, if present after normalization/loading.
if (
    "pre_eo_prediction_sha256"
    in d_df.columns
    and
    "pre_roc_prediction_sha256"
    in e_df.columns
):
    assert np.array_equal(
        d_df[
            "pre_eo_prediction_sha256"
        ].astype(str).to_numpy(),
        e_df[
            "pre_roc_prediction_sha256"
        ].astype(str).to_numpy(),
    )

if (
    "pre_eo_score_sha256"
    in d_df.columns
    and
    "pre_roc_score_sha256"
    in e_df.columns
):
    assert np.array_equal(
        d_df[
            "pre_eo_score_sha256"
        ].astype(str).to_numpy(),
        e_df[
            "pre_roc_score_sha256"
        ].astype(str).to_numpy(),
    )

# Verify upstream metric equality.
for metric in METRICS:

    d_column = (
        f"pre_eo_{metric}"
    )

    e_column = (
        f"pre_roc_{metric}"
    )

    if (
        d_column
        not in d_df.columns
        or e_column
        not in e_df.columns
    ):
        raise RuntimeError(
            "Missing pre-postprocessor metric "
            f"for {metric}"
        )

    assert np.allclose(
        d_df[
            d_column
        ].astype(float),
        e_df[
            e_column
        ].astype(float),
        rtol=0.0,
        atol=1e-12,
    )


# =====================================================================
# Before/after postprocessor table
# =====================================================================

before_after_rows = []

for postprocessor, df, pre_prefix in [
    (
        "EO",
        d_df,
        "pre_eo_",
    ),
    (
        "ROC",
        e_df,
        "pre_roc_",
    ),
]:

    for metric in METRICS:

        pre_values = (
            df[
                f"{pre_prefix}{metric}"
            ].astype(float)
        )

        final_values = (
            df[
                metric
            ].astype(float)
        )

        pre_stats = summarize(
            pre_values
        )

        final_stats = summarize(
            final_values
        )

        sd_ratio = safe_ratio(
            final_stats[
                "sample_std"
            ],
            pre_stats[
                "sample_std"
            ],
        )

        if np.isnan(sd_ratio):
            sd_reduction = np.nan
        else:
            sd_reduction = (
                100.0
                * (
                    1.0
                    - sd_ratio
                )
            )

        before_after_rows.append({
            "postprocessor":
                postprocessor,

            "metric":
                metric,

            "pre_mean":
                pre_stats["mean"],

            "pre_sample_std":
                pre_stats[
                    "sample_std"
                ],

            "pre_min":
                pre_stats["min"],

            "pre_max":
                pre_stats["max"],

            "final_mean":
                final_stats["mean"],

            "final_sample_std":
                final_stats[
                    "sample_std"
                ],

            "final_min":
                final_stats["min"],

            "final_max":
                final_stats["max"],

            "mean_change_final_minus_pre":
                final_stats["mean"]
                - pre_stats["mean"],

            "sd_ratio_final_over_pre":
                sd_ratio,

            "sd_reduction_percent":
                sd_reduction,
        })

before_after = pd.DataFrame(
    before_after_rows
)

BEFORE_AFTER_PATH = (
    ANALYSIS_DIR
    / "run_036_postprocessor_before_after.csv"
)

before_after.to_csv(
    BEFORE_AFTER_PATH,
    index=False,
    float_format="%.15g",
)


# =====================================================================
# Direct EO-vs-ROC final comparison
# =====================================================================

eo_roc_rows = []

for metric in METRICS:

    eo_stats = summarize(
        d_df[metric]
    )

    roc_stats = summarize(
        e_df[metric]
    )

    eo_roc_rows.append({
        "metric":
            metric,

        "eo_mean":
            eo_stats["mean"],

        "eo_sample_std":
            eo_stats[
                "sample_std"
            ],

        "eo_min":
            eo_stats["min"],

        "eo_max":
            eo_stats["max"],

        "roc_mean":
            roc_stats["mean"],

        "roc_sample_std":
            roc_stats[
                "sample_std"
            ],

        "roc_min":
            roc_stats["min"],

        "roc_max":
            roc_stats["max"],

        "roc_minus_eo_mean":
            roc_stats["mean"]
            - eo_stats["mean"],

        "roc_over_eo_sd_ratio":
            safe_ratio(
                roc_stats[
                    "sample_std"
                ],
                eo_stats[
                    "sample_std"
                ],
            ),
    })

eo_vs_roc = pd.DataFrame(
    eo_roc_rows
)

EO_ROC_PATH = (
    ANALYSIS_DIR
    / "run_036_eo_vs_roc.csv"
)

eo_vs_roc.to_csv(
    EO_ROC_PATH,
    index=False,
    float_format="%.15g",
)


# =====================================================================
# Boxplots — Run 036 fixed-split configurations
# =====================================================================

config_order = list(
    CONFIGS.keys()
)

for metric in METRICS:

    values = [
        fixed_frames[
            config
        ][metric].astype(float).to_numpy()
        for config in config_order
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.boxplot(
        values,
        tick_labels=config_order,
        showmeans=True,
    )

    ax.set_title(
        "Run 036 fixed-split seed sensitivity — "
        f"{METRIC_LABELS[metric]}"
    )

    ax.set_ylabel(
        METRIC_LABELS[
            metric
        ]
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        BOXPLOT_DIR
        / f"fixed_split_{metric}.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# Boxplots — old changing split vs Run 036 fixed split
# =====================================================================

for metric in METRICS:

    values = []
    labels = []

    for config in config_order:

        values.append(
            old_frames[
                config
            ][metric]
            .astype(float)
            .to_numpy()
        )

        labels.append(
            f"{config}\nold"
        )

        values.append(
            fixed_frames[
                config
            ][metric]
            .astype(float)
            .to_numpy()
        )

        labels.append(
            f"{config}\nfixed"
        )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.boxplot(
        values,
        tick_labels=labels,
        showmeans=True,
    )

    ax.set_title(
        "Changing-split vs fixed-split variability — "
        f"{METRIC_LABELS[metric]}"
    )

    ax.set_ylabel(
        METRIC_LABELS[
            metric
        ]
    )

    ax.tick_params(
        axis="x",
        rotation=40,
    )

    fig.tight_layout()

    fig.savefig(
        BOXPLOT_DIR
        / f"old_vs_fixed_{metric}.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# Findings markdown
# =====================================================================

def get_summary(
    table,
    config,
    metric,
):
    row = table.loc[
        (
            table[
                "configuration"
            ] == config
        )
        &
        (
            table[
                "metric"
            ] == metric
        )
    ]

    assert len(row) == 1

    return row.iloc[0]


def get_before_after(
    postprocessor,
    metric,
):
    row = before_after.loc[
        (
            before_after[
                "postprocessor"
            ] == postprocessor
        )
        &
        (
            before_after[
                "metric"
            ] == metric
        )
    ]

    assert len(row) == 1

    return row.iloc[0]


def fmt(value, digits=4):
    if pd.isna(value):
        return "NA"

    return f"{float(value):.{digits}f}"


lines = []

lines.append(
    "# Run 036 — Fixed-Split Random-Seed Sensitivity"
)

lines.append("")

lines.append(
    "## Experimental question"
)

lines.append("")

lines.append(
    "Run 036 holds the train/test split fixed at split seed 101 "
    "and varies algorithmic seeds 101–120. The purpose is to "
    "separate algorithmic randomness from variability caused by "
    "changing which observations enter the train and test sets."
)

lines.append("")

lines.append(
    "The five configurations are NONE, DIR+EGR+EO, AL, "
    "DIR+AL+EO, and DIR+AL+ROC."
)

lines.append("")


# NONE ---------------------------------------------------------------

none_acc = get_summary(
    run036_summary,
    "NONE",
    "accuracy",
)

none_di = get_summary(
    run036_summary,
    "NONE",
    "disparate_impact",
)

lines.append(
    "## 1. NONE — negative control"
)

lines.append("")

lines.append(
    "NONE is invariant across all 20 nominal algorithm-seed "
    "records. Accuracy SD is "
    f"{fmt(none_acc['sample_std'], 12)} and DI SD is "
    f"{fmt(none_di['sample_std'], 12)}. This validates the "
    "fixed-split experiment: assigning different seed labels does "
    "not itself create artificial variability."
)

lines.append("")


# DIR+EGR+EO ---------------------------------------------------------

egr_di = get_summary(
    run036_summary,
    "DIR+EGR+EO",
    "disparate_impact",
)

egr_eod = get_summary(
    run036_summary,
    "DIR+EGR+EO",
    "equal_opportunity_difference",
)

lines.append(
    "## 2. DIR+EGR+EO — stochastic but stable"
)

lines.append("")

lines.append(
    "DIR+EGR+EO produces seed-dependent hard predictions, but its "
    "aggregate fairness metrics remain tightly distributed. "
    f"DI SD is {fmt(egr_di['sample_std'])}; "
    f"EOD SD is {fmt(egr_eod['sample_std'])}. "
    "This demonstrates that stochastic predictions do not "
    "necessarily imply unstable group-level fairness conclusions."
)

lines.append("")


# AL -----------------------------------------------------------------

al_di = get_summary(
    run036_summary,
    "AL",
    "disparate_impact",
)

al_eod = get_summary(
    run036_summary,
    "AL",
    "equal_opportunity_difference",
)

al_ped = get_summary(
    run036_summary,
    "AL",
    "predictive_equality_difference",
)

lines.append(
    "## 3. AL — intrinsic algorithmic seed sensitivity"
)

lines.append("")

lines.append(
    "AL remains highly variable even though every seed uses the "
    "same train/test observations. "
    f"DI ranges from {fmt(al_di['min'])} to {fmt(al_di['max'])} "
    f"(SD {fmt(al_di['sample_std'])}); "
    f"EOD ranges from {fmt(al_eod['min'])} to {fmt(al_eod['max'])} "
    f"(SD {fmt(al_eod['sample_std'])}); and "
    f"PED ranges from {fmt(al_ped['min'])} to {fmt(al_ped['max'])}. "
    "Therefore the previously observed AL variability cannot be "
    "attributed primarily to changing data splits."
)

lines.append("")


# DIR+AL+EO ----------------------------------------------------------

eo_di = get_before_after(
    "EO",
    "disparate_impact",
)

eo_spd = get_before_after(
    "EO",
    "statistical_parity_difference",
)

eo_eod = get_before_after(
    "EO",
    "equal_opportunity_difference",
)

eo_ped = get_before_after(
    "EO",
    "predictive_equality_difference",
)

eo_acc = get_before_after(
    "EO",
    "accuracy",
)

lines.append(
    "## 4. DIR+AL+EO — EO stabilizes fairness"
)

lines.append("")

lines.append(
    "DIR+AL remains seed-sensitive before postprocessing, but EO "
    "substantially compresses the fairness distributions. "
    "The reduction factors in fairness SD are approximately "
    f"{fmt(eo_di['pre_sample_std'] / eo_di['final_sample_std'], 1)}x "
    "for DI, "
    f"{fmt(eo_spd['pre_sample_std'] / eo_spd['final_sample_std'], 1)}x "
    "for SPD, "
    f"{fmt(eo_eod['pre_sample_std'] / eo_eod['final_sample_std'], 1)}x "
    "for EOD, and "
    f"{fmt(eo_ped['pre_sample_std'] / eo_ped['final_sample_std'], 1)}x "
    "for PED."
)

lines.append("")

lines.append(
    "This does not mean EO makes the underlying models identical. "
    "Rather, different stochastic DIR+AL realizations are mapped "
    "to much more similar group-level fairness outcomes. "
    "The trade-off is predictive performance: mean accuracy moves "
    f"from {fmt(eo_acc['pre_mean'])} before EO to "
    f"{fmt(eo_acc['final_mean'])} after EO, while accuracy SD "
    f"moves from {fmt(eo_acc['pre_sample_std'])} to "
    f"{fmt(eo_acc['final_sample_std'])}."
)

lines.append("")


# DIR+AL+ROC ---------------------------------------------------------

roc_di = get_before_after(
    "ROC",
    "disparate_impact",
)

roc_eod = get_before_after(
    "ROC",
    "equal_opportunity_difference",
)

roc_ped = get_before_after(
    "ROC",
    "predictive_equality_difference",
)

roc_acc = get_before_after(
    "ROC",
    "accuracy",
)

lines.append(
    "## 5. DIR+AL+ROC — ROC does not provide the same stabilization"
)

lines.append("")

lines.append(
    "36-D and 36-E use exactly the same upstream DIR+AL "
    "realizations for every seed. Only the postprocessor differs. "
    "ROC lowers some average disparity measures, but it does not "
    "stabilize the pipeline in the way EO does."
)

lines.append("")

lines.append(
    f"DI SD changes from {fmt(roc_di['pre_sample_std'])} before ROC "
    f"to {fmt(roc_di['final_sample_std'])} after ROC; "
    f"PED SD changes from {fmt(roc_ped['pre_sample_std'])} to "
    f"{fmt(roc_ped['final_sample_std'])}; and accuracy SD changes "
    f"from {fmt(roc_acc['pre_sample_std'])} to "
    f"{fmt(roc_acc['final_sample_std'])}."
)

lines.append("")


# Direct EO / ROC ----------------------------------------------------

lines.append(
    "## 6. Direct EO-versus-ROC comparison"
)

lines.append("")

for metric in METRICS:

    row = eo_vs_roc.loc[
        eo_vs_roc[
            "metric"
        ] == metric
    ].iloc[0]

    lines.append(
        f"- {METRIC_LABELS[metric]}: "
        f"EO SD={fmt(row['eo_sample_std'])}; "
        f"ROC SD={fmt(row['roc_sample_std'])}; "
        f"ROC/EO SD ratio="
        f"{fmt(row['roc_over_eo_sd_ratio'], 2)}."
    )

lines.append("")

lines.append(
    "Because the upstream DIR+AL predictions and scores are "
    "identical between 36-D and 36-E, this comparison isolates the "
    "different response of EO and ROC to the same stochastic model "
    "outputs."
)

lines.append("")


# ROC search diagnostics --------------------------------------------

if all(
    column in e_df.columns
    for column in [
        "roc_threshold_at_low_boundary",
        "roc_threshold_at_high_boundary",
        "roc_margin_is_zero",
        "roc_changed_predictions",
    ]
):

    low_count = int(
        e_df[
            "roc_threshold_at_low_boundary"
        ].astype(bool).sum()
    )

    high_count = int(
        e_df[
            "roc_threshold_at_high_boundary"
        ].astype(bool).sum()
    )

    zero_margin = int(
        e_df[
            "roc_margin_is_zero"
        ].astype(bool).sum()
    )

    min_changed = int(
        e_df[
            "roc_changed_predictions"
        ].min()
    )

    max_changed = int(
        e_df[
            "roc_changed_predictions"
        ].max()
    )

    lines.append(
        "## 7. ROC search behavior"
    )

    lines.append("")

    lines.append(
        f"ROC selects the exact lower threshold boundary in "
        f"{low_count}/20 seeds and the exact upper boundary in "
        f"{high_count}/20 seeds. It selects a zero reject-option "
        f"margin in {zero_margin}/20 seeds. ROC changes between "
        f"{min_changed} and {max_changed} test predictions "
        "depending on seed."
    )

    lines.append("")

    lines.append(
        "These results reinforce the earlier observation that the "
        "DIR+AL+ROC pathway frequently produces aggressive or "
        "boundary-like threshold solutions."
    )

    lines.append("")


# Paper comparison ---------------------------------------------------

lines.append(
    "## 8. Compatibility with published values"
)

lines.append("")

for config in config_order:

    subset = (
        paper_comparison.loc[
            paper_comparison[
                "configuration"
            ] == config
        ]
    )

    raw_inside = subset.loc[
        subset[
            "paper_in_raw_observed_range"
        ].astype(bool),
        "metric",
    ].tolist()

    raw_outside = subset.loc[
        ~subset[
            "paper_in_raw_observed_range"
        ].astype(bool),
        "metric",
    ].tolist()

    inside_labels = [
        METRIC_LABELS[x]
        for x in raw_inside
    ]

    outside_labels = [
        METRIC_LABELS[x]
        for x in raw_outside
    ]

    lines.append(
        f"- **{config}:** paper value falls inside the raw "
        f"20-seed observed range for "
        f"{', '.join(inside_labels) if inside_labels else 'none'}; "
        f"outside for "
        f"{', '.join(outside_labels) if outside_labels else 'none'}."
    )

lines.append("")

lines.append(
    "Range overlap means only that seed randomness is compatible "
    "with a published value; it does not prove that seed choice "
    "caused a discrepancy. Conversely, a paper value outside the "
    "entire fixed-split seed range is evidence against seed "
    "variation alone being a sufficient explanation."
)

lines.append("")

lines.append(
    "For SPD, EOD, and PED, the paper comparison CSV additionally "
    "records whether the paper value falls within the range of "
    "absolute per-seed disparities. This is retained because the "
    "paper/repository treatment of signed versus absolute fairness "
    "differences remains underdocumented."
)

lines.append("")


# Old vs fixed -------------------------------------------------------

lines.append(
    "## 9. Changing-split versus fixed-split variability"
)

lines.append("")

for config in config_order:

    di_row = old_vs_fixed.loc[
        (
            old_vs_fixed[
                "configuration"
            ] == config
        )
        &
        (
            old_vs_fixed[
                "metric"
            ] == "disparate_impact"
        )
    ].iloc[0]

    lines.append(
        f"- **{config} DI:** old SD="
        f"{fmt(di_row['old_sample_std'])}; "
        f"fixed-split SD="
        f"{fmt(di_row['fixed_sample_std'])}; "
        f"fixed/old ratio="
        f"{fmt(di_row['sd_ratio_fixed_over_old'], 2)}."
    )

lines.append("")

lines.append(
    "The old sensitivity runs changed train/test composition and "
    "algorithm seed together. Run 036 fixes the split, so residual "
    "variation is attributable to algorithmic randomness under the "
    "reconstructed implementation."
)

lines.append("")


# Overall ------------------------------------------------------------

lines.append(
    "## Overall interpretation"
)

lines.append("")

lines.append(
    "Run 036 demonstrates that seed sensitivity is method-dependent. "
    "NONE is deterministic; DIR+EGR+EO is stochastic at the "
    "prediction level but stable in aggregate metrics; AL exhibits "
    "large intrinsic model-level seed sensitivity; EO strongly "
    "suppresses the fairness variability of DIR+AL; and ROC does "
    "not provide comparable stabilization."
)

lines.append("")

lines.append(
    "The experiment therefore supports random seed as a plausible "
    "contributor to some AL-related discrepancies, but not as a "
    "universal explanation. In particular, the DIR+AL+ROC result "
    "should remain an unresolved reproduction discrepancy when the "
    "published result lies outside the observed fixed-split seed "
    "distribution."
)

lines.append("")

lines.append(
    "No parameter tuning was performed in Run 036."
)

FINDINGS_PATH = (
    ANALYSIS_DIR
    / "run_036_findings.md"
)

FINDINGS_PATH.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)


# =====================================================================
# Human-readable console summary
# =====================================================================

print("=" * 100)
print("RUN 036 — CONSOLIDATED ANALYSIS")
print("=" * 100)

print()
print("OLD RESULT FILES SELECTED")
print("-------------------------")

for config in config_order:
    print(
        f"{config}: "
        f"{old_paths[config].relative_to(REPO_ROOT)}"
    )

print()
print("FIXED-SPLIT SUMMARY")
print("-------------------")

for config in config_order:

    print()
    print(config)

    subset = (
        run036_summary.loc[
            run036_summary[
                "configuration"
            ] == config
        ]
    )

    for _, row in subset.iterrows():
        print(
            f"  {METRIC_LABELS[row['metric']]}: "
            f"mean={row['mean']:.6f}, "
            f"SD={row['sample_std']:.6f}, "
            f"range=[{row['min']:.6f}, "
            f"{row['max']:.6f}]"
        )

print()
print("EO VS ROC — SAME UPSTREAM DIR+AL")
print("--------------------------------")

for _, row in eo_vs_roc.iterrows():
    print(
        f"{METRIC_LABELS[row['metric']]}: "
        f"EO SD={row['eo_sample_std']:.6f}, "
        f"ROC SD={row['roc_sample_std']:.6f}, "
        f"ROC/EO="
        f"{row['roc_over_eo_sd_ratio']:.2f}x"
    )

print()
print("OUTPUTS")
print("-------")

for path in [
    ALL_FIXED_PATH,
    SUMMARY_PATH,
    PAPER_PATH,
    OLD_FIXED_PATH,
    BEFORE_AFTER_PATH,
    EO_ROC_PATH,
    FINDINGS_PATH,
]:
    print(path)

print(BOXPLOT_DIR)

print()
print("Run 036 consolidated analysis: PASS")
