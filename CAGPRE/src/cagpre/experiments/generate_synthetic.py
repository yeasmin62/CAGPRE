# generate_cagpre_synthetic.py

import os
import random
import numpy as np
import pandas as pd

# Change this import to the filename of your current CAGPRE pipeline
from ..engines.shared import load_ground_truth


# ============================================================
# Configuration
# ============================================================

VARIABLES = ["SBP", "DBP", "PR", "CH", "GLUCOSE"]

N_RECORDS = 20_000
ERROR_RATE = 0.05

# Additional NON-error cases:
# clinically abnormal but still physiologically plausible
ABNORMAL_PLAUSIBLE_RATE = 0.05

# A context-dependent error must be plausible under several OTHER expert
# contexts, not merely under a single donor context. This is independent of
# Global IQR and all other evaluated baselines.
MIN_CONTEXT_DONOR_SUPPORT = 3

# Keep contextual errors close to the target-context plausible boundary.
# This makes them subtle context errors rather than gross magnitude errors.
CONTEXT_MAX_BOUNDARY_FRACTION = 0.10

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

SEED = 42

GT_DIR = os.getenv("CAGPRE_GT_DIR", str(__import__("pathlib").Path(__file__).resolve().parents[3] / "data" / "ground_truth"))
OUTPUT_DIR = os.getenv("CAGPRE_SYNTHETIC_DIR", "synthetic_data")


# Unit transformations.
#
# The generated clean_value is in the expected unit.
# Applying one of these factors simulates storing the value
# numerically in another unit without converting it back.
#
# SBP/DBP: mmHg -> kPa
# CH: mmol/L -> mg/dL
# GLUCOSE: mmol/L -> mg/dL
#
# PR is deliberately omitted because the previous factor 1.0
# did not actually create a unit error.
UNIT_FACTORS = {
    "SBP": [0.133322, 7.50062],
    "DBP": [0.133322, 7.50062],
    "CH": [38.67, 1.0 / 38.67],
    "GLUCOSE": [18.0, 1.0 / 18.0],
}


AGE_BINS = [0, 19, 29, 39, 49, 59, 69, 79, 89, 200]
AGE_LABELS = [
    "0-19", "20-29", "30-39", "40-49",
    "50-59", "60-69", "70-79", "80-89", "90+"
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Basic sampling functions
# ============================================================

def sample_plausible_value(gt_row, rng):
    """
    Generate a physiologically plausible measurement for the
    selected patient context.

    Unlike the previous generator, this does NOT derive sigma
    from Q1/Q3. This avoids constructing the normal synthetic
    distribution directly from the same statistics used by IQR.

    The distribution is centred on the expert median and
    truncated to the expert plausible interval.
    """

    low = float(gt_row.plausible_low)
    high = float(gt_row.plausible_high)
    median = float(gt_row.median)

    width = high - low

    if width <= 0:
        return median

    # Spread based on plausible interval, rather than IQR.
    sigma = max(width / 8.0, 1e-6)

    # Rejection sampling
    for _ in range(100):
        value = rng.normal(median, sigma)

        if low <= value <= high:
            return float(value)

    # Extremely unlikely fallback
    return float(np.clip(median, low, high))


def sample_abnormal_but_plausible(gt_row, rng):
    """
    Generate a measurement that is outside the clinically
    normal range but remains inside the expert plausible range.

    IMPORTANT:
        This is NOT an error.
        is_error must remain 0.
    """

    p_low = float(gt_row.plausible_low)
    p_high = float(gt_row.plausible_high)

    n_low = float(gt_row.normal_low)
    n_high = float(gt_row.normal_high)

    eps = max((p_high - p_low) * 1e-4, 1e-6)

    candidate_intervals = []

    # Below normal, but still plausible
    if n_low > p_low + eps:
        candidate_intervals.append(
            (p_low + eps, min(n_low - eps, p_high - eps))
        )

    # Above normal, but still plausible
    if n_high < p_high - eps:
        candidate_intervals.append(
            (max(n_high + eps, p_low + eps), p_high - eps)
        )

    candidate_intervals = [
        (lo, hi)
        for lo, hi in candidate_intervals
        if hi > lo
    ]

    if not candidate_intervals:
        return None

    lo, hi = candidate_intervals[
        rng.integers(0, len(candidate_intervals))
    ]

    return float(rng.uniform(lo, hi))


# ============================================================
# Context-dependent error
# ============================================================

def sample_context_dependent_error(target_row, all_rows, rng):
    """
    Generate a genuine context-dependent plausibility error.

    The sampled value must satisfy all of the following:
      1. It is outside the expert-defined plausible range of the target context.
      2. It is inside the expert-defined plausible ranges of at least
         MIN_CONTEXT_DONOR_SUPPORT other legitimate contexts.
      3. It lies close to the target plausible-range boundary, within
         CONTEXT_MAX_BOUNDARY_FRACTION of the target interval width.

    This definition is baseline-independent: it does not use Global IQR,
    CAGPRE predictions, or any other evaluated method.

    Requiring support from multiple other expert contexts makes the value
    genuinely context-dependent rather than simply an extreme value that is
    plausible in one exceptional donor context.
    """

    target_low = float(target_row.plausible_low)
    target_high = float(target_row.plausible_high)
    target_width = target_high - target_low

    if not np.isfinite(target_width) or target_width <= 0:
        return None, None

    max_distance = max(
        CONTEXT_MAX_BOUNDARY_FRACTION * target_width,
        1e-6,
    )
    eps = max(0.001 * target_width, 1e-6)

    # Collect OTHER valid expert contexts.
    donors = []

    for donor in all_rows:
        same_context = (
            str(donor.condition).lower() == str(target_row.condition).lower()
            and str(donor.gender).upper() == str(target_row.gender).upper()
            and donor.age_min == target_row.age_min
            and donor.age_max == target_row.age_max
        )

        if same_context:
            continue

        donor_low = float(donor.plausible_low)
        donor_high = float(donor.plausible_high)

        if (
            not np.isfinite(donor_low)
            or not np.isfinite(donor_high)
            or donor_high <= donor_low
        ):
            continue

        donors.append((donor, donor_low, donor_high))

    if not donors:
        return None, None

    # Candidate bands are immediately outside the target expert range.
    target_bands = [
        (
            target_low - max_distance,
            target_low - eps,
        ),
        (
            target_high + eps,
            target_high + max_distance,
        ),
    ]

    eligible_segments = []

    for band_low, band_high in target_bands:
        if band_high <= band_low:
            continue

        # Intersect every donor plausible interval with this target band.
        intersections = []

        for donor, donor_low, donor_high in donors:
            lo = max(band_low, donor_low)
            hi = min(band_high, donor_high)

            if hi > lo:
                intersections.append(
                    (donor, lo, hi)
                )

        if not intersections:
            continue

        # Split the band at every interval boundary. Within each resulting
        # segment, donor-support count is constant.
        breakpoints = {band_low, band_high}

        for _, lo, hi in intersections:
            breakpoints.add(lo)
            breakpoints.add(hi)

        breakpoints = sorted(breakpoints)

        for j in range(len(breakpoints) - 1):
            seg_low = breakpoints[j]
            seg_high = breakpoints[j + 1]

            if seg_high <= seg_low:
                continue

            midpoint = 0.5 * (seg_low + seg_high)

            supporting = [
                donor
                for donor, donor_low, donor_high in donors
                if donor_low <= midpoint <= donor_high
            ]

            if len(supporting) >= MIN_CONTEXT_DONOR_SUPPORT:
                eligible_segments.append(
                    (
                        seg_low,
                        seg_high,
                        supporting,
                    )
                )

    if not eligible_segments:
        return None, None

    # Prefer larger valid segments without using any competing model.
    lengths = np.array(
        [
            seg_high - seg_low
            for seg_low, seg_high, _ in eligible_segments
        ],
        dtype=float,
    )
    probabilities = lengths / lengths.sum()

    chosen_idx = int(
        rng.choice(
            len(eligible_segments),
            p=probabilities,
        )
    )

    lo, hi, supporting = eligible_segments[chosen_idx]
    value = float(rng.uniform(lo, hi))

    # Recalculate support at the exact sampled value for auditability.
    exact_support = [
        donor
        for donor, donor_low, donor_high in donors
        if donor_low <= value <= donor_high
    ]

    if len(exact_support) < MIN_CONTEXT_DONOR_SUPPORT:
        raise RuntimeError(
            "Internal error: sampled contextual value has insufficient "
            "donor-context support."
        )

    # Record one representative donor plus the total number of supporting
    # contexts. The value itself is plausible under all exact_support contexts.
    donor = exact_support[
        rng.integers(0, len(exact_support))
    ]

    donor_description = (
        f"{donor.condition}|"
        f"{donor.gender}|"
        f"{donor.age_min}-{donor.age_max}"
        f"|support={len(exact_support)}"
    )

    return value, donor_description


# ============================================================
# Unit-confusion error
# ============================================================

def make_unit_error(clean_value, gt_row, variable, rng):
    """
    Apply an actual unit-conversion factor.

    We retain only transformations that result in a value
    outside the CURRENT expert plausible interval.
    """

    factors = UNIT_FACTORS.get(variable, [])

    if not factors:
        return None, None

    p_low = float(gt_row.plausible_low)
    p_high = float(gt_row.plausible_high)

    candidates = []

    for factor in factors:
        corrupted = clean_value * factor

        if corrupted < p_low or corrupted > p_high:
            candidates.append(
                (float(corrupted), float(factor))
            )

    if not candidates:
        return None, None

    return candidates[
        rng.integers(0, len(candidates))
    ]


# ============================================================
# Sporadic entry error
# ============================================================

def make_sporadic_error(clean_value, gt_row, rng):
    """
    Simulate obvious transcription / decimal-position errors.
    """

    p_low = float(gt_row.plausible_low)
    p_high = float(gt_row.plausible_high)

    candidates = [
        clean_value * 10.0,
        clean_value * 100.0,
        clean_value / 10.0,
        clean_value / 100.0,
    ]

    candidates = [
        float(v)
        for v in candidates
        if v < p_low or v > p_high
    ]

    if not candidates:
        width = max(p_high - p_low, 1.0)

        candidates = [
            float(p_high + 5.0 * width),
            float(p_low - 5.0 * width),
        ]

    return candidates[
        rng.integers(0, len(candidates))
    ]


# ============================================================
# Create base data
# ============================================================

def generate_base_records(
    gt_index,
    variable,
    n_records,
    rng
):
    """
    Generate the initial CLEAN dataset.

    Every record is plausible for its own patient context.
    """

    rows = []

    gt_rows = gt_index.rows

    for i in range(n_records):

        gt_row = gt_rows[
            rng.integers(0, len(gt_rows))
        ]

        age_min = int(gt_row.age_min)
        age_max = int(gt_row.age_max)

        if age_max <= age_min:
            age_max = age_min + 1

        age = int(
            rng.integers(age_min, age_max)
        )

        clean_value = sample_plausible_value(
            gt_row,
            rng
        )

        rows.append({
            "record_id": i,
            "variable": variable,
            "age": age,
            "gender": gt_row.gender.upper(),
            "condition": gt_row.condition,

            "value": clean_value,
            "clean_value": clean_value,

            "gt_low": float(gt_row.plausible_low),
            "gt_high": float(gt_row.plausible_high),

            "normal_low": float(gt_row.normal_low),
            "normal_high": float(gt_row.normal_high),

            "gt_median": float(gt_row.median),
            "gt_q1": float(gt_row.q1),
            "gt_q3": float(gt_row.q3),

            "is_error": 0,
            "error_type": "none",
            "requested_error_type": "none",
            "injected": 0,

            "donor_context": "",
            "corruption_factor": np.nan,
            "distance_from_gt_boundary": 0.0,

            # Save the original GT row identifier
            "gt_age_min": age_min,
            "gt_age_max": age_max,
        })

    df = pd.DataFrame(rows)

    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS
    ).astype(str)

    return df


# ============================================================
# Fixed train / validation / test split
# ============================================================

def assign_fixed_splits(df, seed):
    """
    Fixed 70/15/15 split saved permanently with the dataset.
    """

    rng = np.random.default_rng(seed)

    indices = np.arange(len(df))
    rng.shuffle(indices)

    n = len(df)

    n_train = int(round(n * TRAIN_FRACTION))
    n_val = int(round(n * VAL_FRACTION))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    df = df.copy()
    df["split"] = ""

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    return df


# ============================================================
# Add abnormal-but-plausible NEGATIVE cases
# ============================================================

def inject_abnormal_but_plausible_cases(
    df,
    gt_index,
    split_name,
    rate,
    rng
):
    """
    Add clinically abnormal but expert-defined plausible observations
    to one fixed split.

    These are hard NEGATIVE cases, not errors. They remain is_error = 0.
    The same rate is used independently in train, validation, and test.
    """

    split_indices = df.index[
        (df["split"] == split_name)
        &
        (df["error_type"] == "none")
    ].to_numpy()

    n_cases = int(
        round(
            (df["split"] == split_name).sum()
            * rate
        )
    )

    rng.shuffle(split_indices)

    changed = 0

    for idx in split_indices:

        if changed >= n_cases:
            break

        row = df.loc[idx]

        gt_row = gt_index.lookup(
            int(row["age"]),
            row["gender"],
            row["condition"]
        )

        value = sample_abnormal_but_plausible(
            gt_row,
            rng
        )

        if value is None:
            continue

        df.at[idx, "value"] = value
        df.at[idx, "clean_value"] = value

        df.at[idx, "is_error"] = 0
        df.at[idx, "error_type"] = "abnormal_but_plausible"
        df.at[idx, "requested_error_type"] = "abnormal_but_plausible"
        df.at[idx, "distance_from_gt_boundary"] = 0.0

        changed += 1

    print(
        f"  {split_name:5s} abnormal-but-plausible negatives: "
        f"{changed:,} / {n_cases:,} requested"
    )

    if changed != n_cases:
        raise RuntimeError(
            f"Could only construct {changed:,} of {n_cases:,} requested "
            f"abnormal-but-plausible cases for split '{split_name}'."
        )

    return df


# ============================================================
# Inject actual errors independently into each split
# ============================================================
# ============================================================
# Inject actual errors ONLY into test set
# ============================================================

def inject_split_errors(
    df,
    gt_index,
    variable,
    split_name,
    error_rate,
    rng
):
    """
    Inject genuine plausibility errors into one fixed split.

    The same controlled error construction is applied independently to
    train, validation, and test. This matches the real-data protocol in
    which implausible observations may occur in every split while labels
    are used only for validation/model selection and final evaluation.

    Error composition is controlled explicitly.

    For SBP, DBP, CH, and GLUCOSE:
        - 1/3 context-dependent plausibility errors
        - 1/3 unit-confusion errors
        - 1/3 sporadic-entry errors

    For PR:
        - 1/2 context-dependent plausibility errors
        - 1/2 sporadic-entry errors

    Context-dependent errors are NEVER replaced by sporadic errors.
    If a requested context-dependent case cannot be constructed for a
    particular record, another eligible record from the same split is tried.
    If the requested count cannot be reached, generation fails rather than
    silently changing the requested error type.

    The same principle is used for unit-confusion errors: failed unit
    transformations are skipped rather than silently relabeled.
    """

    variable = variable.upper()

    candidate_indices = df.index[
        (df["split"] == split_name)
        &
        (df["error_type"] == "none")
    ].to_numpy()

    n_split = int(
        (df["split"] == split_name).sum()
    )

    n_errors = int(
        round(n_split * error_rate)
    )

    if variable in UNIT_FACTORS:
        n_context = n_errors // 3
        n_unit = n_errors // 3
        n_sporadic = (
            n_errors - n_context - n_unit
        )
        requested_counts = {
            "context_dependent": n_context,
            "unit_confusion": n_unit,
            "sporadic_entry": n_sporadic,
        }
    else:
        # PR has no defensible unit-confusion transformation.
        n_context = n_errors // 2
        n_sporadic = n_errors - n_context
        requested_counts = {
            "context_dependent": n_context,
            "sporadic_entry": n_sporadic,
        }

    created_counts = {
        error_type: 0
        for error_type in requested_counts
    }

    all_gt_rows = gt_index.rows
    used_indices = set()

    def available_indices():
        remaining = np.array(
            [
                idx
                for idx in candidate_indices
                if idx not in used_indices
            ],
            dtype=int
        )
        rng.shuffle(remaining)
        return remaining

    def boundary_distance(value, low, high):
        value = float(value)
        low = float(low)
        high = float(high)

        if value < low:
            return float(low - value)
        if value > high:
            return float(value - high)
        return 0.0

    def store_error(
        idx,
        corrupted_value,
        error_type,
        donor_context="",
        corruption_factor=np.nan,
    ):
        row = df.loc[idx]

        is_error = int(
            corrupted_value < float(row["gt_low"])
            or
            corrupted_value > float(row["gt_high"])
        )

        if is_error != 1:
            raise RuntimeError(
                f"Injected error remained plausible: "
                f"variable={variable}, "
                f"type={error_type}, "
                f"value={corrupted_value}, "
                f"range=[{row['gt_low']}, {row['gt_high']}]"
            )

        df.at[idx, "value"] = float(corrupted_value)
        df.at[idx, "is_error"] = 1
        df.at[idx, "injected"] = 1
        df.at[idx, "error_type"] = error_type
        df.at[idx, "requested_error_type"] = error_type
        df.at[idx, "donor_context"] = donor_context
        df.at[idx, "corruption_factor"] = corruption_factor
        df.at[idx, "distance_from_gt_boundary"] = (
            boundary_distance(
                corrupted_value,
                row["gt_low"],
                row["gt_high"],
            )
        )

        used_indices.add(idx)
        created_counts[error_type] += 1

    # ========================================================
    # 1. Context-dependent plausibility errors
    # ========================================================

    requested_context = requested_counts.get(
        "context_dependent",
        0
    )

    for idx in available_indices():

        if (
            created_counts.get("context_dependent", 0)
            >= requested_context
        ):
            break

        row = df.loc[idx]

        target_gt = gt_index.lookup(
            int(row["age"]),
            row["gender"],
            row["condition"],
        )

        corrupted_value, donor_context = (
            sample_context_dependent_error(
                target_gt,
                all_gt_rows,
                rng,
            )
        )

        if corrupted_value is None:
            # Try another target record. Do not substitute
            # another error type.
            continue

        store_error(
            idx=idx,
            corrupted_value=corrupted_value,
            error_type="context_dependent",
            donor_context=donor_context,
        )

    # ========================================================
    # 2. Unit-confusion errors
    # ========================================================

    requested_unit = requested_counts.get(
        "unit_confusion",
        0
    )

    if requested_unit > 0:

        for idx in available_indices():

            if (
                created_counts.get("unit_confusion", 0)
                >= requested_unit
            ):
                break

            row = df.loc[idx]

            target_gt = gt_index.lookup(
                int(row["age"]),
                row["gender"],
                row["condition"],
            )

            clean_value = float(
                row["clean_value"]
            )

            corrupted_value, corruption_factor = (
                make_unit_error(
                    clean_value,
                    target_gt,
                    variable,
                    rng,
                )
            )

            if corrupted_value is None:
                # Try another record. Do not turn it into
                # a sporadic error.
                continue

            store_error(
                idx=idx,
                corrupted_value=corrupted_value,
                error_type="unit_confusion",
                corruption_factor=corruption_factor,
            )

    # ========================================================
    # 3. Sporadic-entry errors
    # ========================================================

    requested_sporadic = requested_counts.get(
        "sporadic_entry",
        0
    )

    for idx in available_indices():

        if (
            created_counts.get("sporadic_entry", 0)
            >= requested_sporadic
        ):
            break

        row = df.loc[idx]

        target_gt = gt_index.lookup(
            int(row["age"]),
            row["gender"],
            row["condition"],
        )

        clean_value = float(
            row["clean_value"]
        )

        corrupted_value = make_sporadic_error(
            clean_value,
            target_gt,
            rng,
        )

        store_error(
            idx=idx,
            corrupted_value=corrupted_value,
            error_type="sporadic_entry",
        )

    # ========================================================
    # Report requested vs achieved composition
    # ========================================================

    print(f"  {split_name:5s} injected errors:")

    for error_type, requested in requested_counts.items():
        created = created_counts.get(
            error_type,
            0
        )

        print(
            f"    {error_type:25s}: "
            f"{created:,} / {requested:,} requested"
        )

        if created < requested:
            print(
                f"    [WARNING] Could only construct "
                f"{created:,} of {requested:,} requested "
                f"{error_type} errors for {variable}."
            )

    total_created = sum(
        created_counts.values()
    )

    achieved_rate = (
        total_created / n_split
        if n_split > 0
        else 0.0
    )

    print(
        f"  {split_name:5s} total injected errors: "
        f"{total_created:,} / {n_errors:,} requested "
        f"({100 * achieved_rate:.2f}% of {split_name} split)"
    )

    if total_created != n_errors:
        raise RuntimeError(
            f"Split '{split_name}' contains {total_created:,} injected "
            f"errors, but {n_errors:,} were requested."
        )

    return df

# ============================================================
# Validation
# ============================================================

def validate_dataset(df):
    """
    Sanity checks before saving the synthetic benchmark.
    """

    # --------------------------------------------------------
    # Label consistency
    # --------------------------------------------------------
    expected = (
        (df["value"] < df["gt_low"])
        |
        (df["value"] > df["gt_high"])
    ).astype(int)

    mismatch = (
        expected != df["is_error"]
    ).sum()

    if mismatch:
        raise RuntimeError(
            f"{mismatch} records have inconsistent "
            f"is_error labels."
        )

    # --------------------------------------------------------
    # Each split must contain the requested error rate.
    # --------------------------------------------------------
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        expected_errors = int(
            round(len(split_df) * ERROR_RATE)
        )
        actual_errors = int(
            split_df["is_error"].sum()
        )

        if actual_errors != expected_errors:
            raise RuntimeError(
                f"Split '{split_name}' contains {actual_errors:,} errors; "
                f"expected {expected_errors:,} ({100 * ERROR_RATE:.2f}%)."
            )

    # --------------------------------------------------------
    # Abnormal-but-plausible cases must not be errors
    # --------------------------------------------------------
    abp = df[
        df["error_type"]
        == "abnormal_but_plausible"
    ]

    if len(abp) > 0:
        assert (abp["is_error"] == 0).all()

    # --------------------------------------------------------
    # Context-dependent errors must retain their identity and
    # must have a donor context and positive boundary distance.
    # --------------------------------------------------------
    context_errors = df[
        df["error_type"] == "context_dependent"
    ]

    if len(context_errors) > 0:
        if not (context_errors["is_error"] == 1).all():
            raise RuntimeError(
                "A context-dependent case is not labeled as an error."
            )

        if (
            context_errors["donor_context"]
            .fillna("")
            .astype(str)
            .str.len()
            .eq(0)
            .any()
        ):
            raise RuntimeError(
                "A context-dependent error is missing donor_context."
            )

        if not (
            context_errors["distance_from_gt_boundary"] > 0
        ).all():
            raise RuntimeError(
                "A context-dependent error has non-positive "
                "distance_from_gt_boundary."
            )

        # Every contextual error records one donor plus support=N.
        support = (
            context_errors["donor_context"]
            .astype(str)
            .str.extract(r"support=(\d+)")[0]
        )

        if support.isna().any():
            raise RuntimeError(
                "A context-dependent error is missing donor-support metadata."
            )

        support = support.astype(int)

        if (support < MIN_CONTEXT_DONOR_SUPPORT).any():
            raise RuntimeError(
                "A context-dependent error has insufficient support from "
                "other expert contexts."
            )

    # --------------------------------------------------------
    # No injected error may silently change type.
    # --------------------------------------------------------
    injected = df[df["injected"] == 1]

    if len(injected) > 0:
        mismatch_type = (
            injected["requested_error_type"]
            != injected["error_type"]
        ).sum()

        if mismatch_type:
            raise RuntimeError(
                f"{mismatch_type} injected records changed "
                f"from their requested error type."
            )

    print("\nDataset validation passed.")


# ============================================================
# Summary
# ============================================================

def print_summary(df, variable):

    print("\n" + "=" * 70)
    print(variable)
    print("=" * 70)

    print("\nSplit sizes:")
    print(
        df["split"]
        .value_counts()
        .sort_index()
    )

    print("\nError rate by split:")
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        print(
            f"  {split_name:5s}: "
            f"{int(split_df['is_error'].sum()):,} / "
            f"{len(split_df):,} = "
            f"{100 * split_df['is_error'].mean():.2f}%"
        )

    print("\nCases by split and type:")
    print(
        pd.crosstab(
            [df["split"], df["error_type"]],
            df["is_error"],
            margins=True
        )
    )


# ============================================================
# Full generation
# ============================================================
# ============================================================
# Full generation
# ============================================================

def generate_dataset_for_variable(
    variable,
    gt_dir,
    n_records,
    seed
):

    rng = np.random.default_rng(seed)

    gt_index = load_ground_truth(
        variable,
        gt_dir
    )

    # 1. Generate clean plausible observations
    df = generate_base_records(
        gt_index=gt_index,
        variable=variable,
        n_records=n_records,
        rng=rng
    )

    # 2. Permanently assign train / val / test
    df = assign_fixed_splits(
        df,
        seed
    )

    # 3. Add hard NEGATIVE cases independently to every split:
    # clinically abnormal but still expert-defined plausible.
    for split_name in ["train", "val", "test"]:
        df = inject_abnormal_but_plausible_cases(
            df=df,
            gt_index=gt_index,
            split_name=split_name,
            rate=ABNORMAL_PLAUSIBLE_RATE,
            rng=rng
        )

    # 4. Inject genuine errors independently into every split.
    # Labels are not passed to the model as training targets; they are
    # retained for validation/model selection and held-out test evaluation.
    for split_name in ["train", "val", "test"]:
        df = inject_split_errors(
            df=df,
            gt_index=gt_index,
            variable=variable,
            split_name=split_name,
            error_rate=ERROR_RATE,
            rng=rng
        )

    # 5. Recompute final label from GT for EVERY row
    df["is_error"] = (
        (df["value"] < df["gt_low"])
        |
        (df["value"] > df["gt_high"])
    ).astype(int)

    # 6. Validate
    validate_dataset(df)

    return df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    set_seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for i, variable in enumerate(VARIABLES):

        print(
            f"\nGenerating synthetic data for "
            f"{variable} ..."
        )

        df = generate_dataset_for_variable(
            variable=variable,
            gt_dir=GT_DIR,
            n_records=N_RECORDS,
            seed=SEED + i
        )

        print_summary(
            df,
            variable
        )

        csv_path = os.path.join(
            OUTPUT_DIR,
            f"synthetic_{variable}.csv"
        )

        df.to_csv(
            csv_path,
            index=False
        )

        print(
            f"\nSaved: {csv_path}"
        )