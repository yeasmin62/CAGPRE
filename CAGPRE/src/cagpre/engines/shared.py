# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import gaussian_kde
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Gender concept IDs
# ---------------------------------------------------------------------------
GENDER_CID = {"MALE": 8507, "FEMALE": 8532}
GENDER_CID_REVERSE = {8507: "MALE", 8532: "FEMALE"}

# ---------------------------------------------------------------------------
# OMOP Concept IDs
# ---------------------------------------------------------------------------
MEASUREMENT_CONCEPT_IDS = {
    "SBP":     [4248525, 4232915, 4292062, 4152194],
    "DBP":     [4236281, 4248524, 4268883, 4154790],
    "PR":      [4301868],
    "CH":      [3023602],
    "GLUCOSE": [3004501, 3037110, 37398242, 46236950, 4218282, 4156660, 3040151, 3020491, 37399654, 4202143, 3013826, 3044242, 3018251, 3038515, 3020399],
    "NT-PROBNP": [3029187, 3031569, 42870364, 42529224, 46236287, 4307029, 3029435, 3052295],
    "BILIRUBIN_TOTAL": [37394117, 3028833, 3024128, 4195338, 3006140],
    "C-REACTIVE_PROTEIN": [4208414, 3020460, 42870365, 46234770],
    "CHOLESTEROL_HDL": [3007070, 3023602, 4042059],
    "CHOLESTEROL_LDL": [3028437, 37393245, 3001308],
    "FIBRINOGEN": [1616312, 37399293, 3016407, 4094436, 3052964],
    "URIC_ACID": [3037556, 37394445, 3026493, 4076924],
    "ALANINE_AMINOTRANSFERASE": [4146380, 3006923],
    "ALBUMIN": [3024561, 4097664, 40757478, 4017497],
    "ASPARTATE_AMINOTRANSFERASE": [3010587, 37394375, 3013721, 4263457],
    "ESR": [3015183, 4212065, 3013688],
    "CREATININE": [37394438, 3051825, 3016723, 3020564, 4013964],
    "EGFR": [46236952, 40764999, 42869913],
    "HAEMOGLOBIN": [40480067, 3000963, 40762351, 4153000, 37208644],
    "POTASSIUM": [3005456, 3023103, 4207483, 46235078, 37394212],
    "RED_BLOOD_CELL_COUNT": [37393849, 4030871],
    "SODIUM": [3000285, 3019550, 4021154, 37393718]
}

# Condition concept_ids relevant per variable, with priority order
CONDITION_CONCEPT_IDS = {
    "SBP": {319826: "Secondary hypertension", 320128: "Essential hypertension"},
    "DBP": {319826: "Secondary hypertension", 320128: "Essential hypertension"},
    "PR":  {437663: "Fever", 4164645: "Fever with chills"},
    "CH":  {4142479: "Hyperthyroidism", 140673: "Hypothyroidism"},
    "GLUCOSE": {201254: "Type 1 diabetes mellitus", 40484648: "Type 1 diabetes mellitus",
                201826: "Type 2 diabetes mellitus", 201820: "Diabetes mellitus"},
    "NT-PROBNP": {},
    "BILIRUBIN_TOTAL": {},
    "C-REACTIVE_PROTEIN": {},
    "CHOLESTEROL_HDL": {},
    "CHOLESTEROL_LDL": {},
    "FIBRINOGEN": {},
    "URIC_ACID": {},
    "ALANINE_AMINOTRANSFERASE": {},
    "ALBUMIN": {},
    "ASPARTATE_AMINOTRANSFERASE": {},
    "ESR": {},
    "CREATININE": {},
    "EGFR": {},
    "HAEMOGLOBIN": {},
    "POTASSIUM": {},
    "RED_BLOOD_CELL_COUNT": {},
    "SODIUM": {}
}

DEFAULT_CONDITION_LABEL = {
    "SBP": "No Hypertension", "DBP": "No Hypertension",
    "PR": "None", "CH": "None", "GLUCOSE": "None",
    "NT-PROBNP": "None",
    "BILIRUBIN_TOTAL": "None",
    "C-REACTIVE_PROTEIN": "None",
    "CHOLESTEROL_HDL": "None",
    "CHOLESTEROL_LDL": "None",
    "FIBRINOGEN": "None",
    "URIC_ACID": "None",
    "ALANINE_AMINOTRANSFERASE": "None",
    "ALBUMIN": "None",
    "ASPARTATE_AMINOTRANSFERASE": "None",
    "ESR": "None",
    "CREATININE": "None",
    "EGFR": "None",
    "HAEMOGLOBIN": "None",
    "POTASSIUM": "None",
    "RED_BLOOD_CELL_COUNT": "None",
    "SODIUM": "None"
}

CONDITION_LABEL_TO_CID = {
    "Essential hypertension": 320128,
    "Secondary hypertension": 319826,
    "No Hypertension": 0,
    "Fever": 437663,
    "Fever with chills": 4164645,
    "Hyperthyroidism": 4142479,
    "Hypothyroidism": 4061268,
    "Type 1 diabetes mellitus": 201254,
    "Type 2 diabetes mellitus": 201826,
    "Diabetes mellitus": 201820,
    "None": 0,
}

GT_FILE_MAP = {
    "SBP": "SBPGT.csv", "DBP": "DBPGT.csv", "PR": "PRGT.csv",
    "CH": "CHGT.csv", "GLUCOSE": "GlucoseGT.csv",
}

# ---------------------------------------------------------------------------
# Database configuration (edit for your server)
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "your_database",
    "user": "your_username",
    "password": "your_password",
    "schema": "public",
}


# ===========================================================================
# Ground Truth
# ===========================================================================
@dataclass
class GroundTruthRow:
    condition: str
    gender: str
    age_min: int
    age_max: int
    plausible_low: float
    plausible_high: float
    median: float
    q1: float
    q3: float
    normal_low: float = 0.0
    normal_high: float = 0.0


class GroundTruthIndex:
    def __init__(self, rows: List[GroundTruthRow]):
        self.rows = rows

    def lookup(self, age: int, gender: str, condition: str) -> GroundTruthRow:
        candidates = []
        for r in self.rows:
            if r.gender.upper() == gender.upper() and r.condition.lower() == condition.lower():
                if r.age_min <= age < r.age_max:
                    candidates.append(r)
        if not candidates:
            for r in self.rows:
                if r.condition.lower() == condition.lower():
                    if r.age_min <= age < r.age_max:
                        candidates.append(r)
        if not candidates:
            candidates = self.rows
        return candidates[-1]

    def global_plausible_range(self) -> Tuple[float, float]:
        lows = [r.plausible_low for r in self.rows]
        highs = [r.plausible_high for r in self.rows]
        return float(np.nanmin(lows)), float(np.nanmax(highs))


def _normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    norm_map = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _to_float(val) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return float("nan")
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        # Try extracting first number
        import re
        m = re.search(r"-?\d+\.?\d*", s)
        return float(m.group()) if m else float("nan")


def _parse_age_range(s: str) -> Tuple[int, int]:
    s = str(s).strip()
    import re
    m = re.match(r"(\d+)\s*[-\u2013]\s*(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2)) + 1
    try:
        v = int(float(s))
        return v, v + 10
    except ValueError:
        return 0, 200


def load_ground_truth(variable: str, gt_dir: str = ".") -> GroundTruthIndex:
    var = variable.upper()
    filename = GT_FILE_MAP.get(var)
    if not filename:
        print(f"  [INFO] No GT file found for variable '{var}'. Using dummy Ground Truth.")
        dummy_row = GroundTruthRow(condition="None", gender="MALE", age_min=0, age_max=200, plausible_low=-np.inf, plausible_high=np.inf, median=0, q1=0, q3=0)
        return GroundTruthIndex([dummy_row])
    path = os.path.join(gt_dir, filename)
    df = pd.read_csv(path)

    cond_col = _find_col(df, "Condition", "condition")
    gender_col = _find_col(df, "Gender", "gender")
    age_col = _find_col(df, "Age range", "Age_range", "age_range")
    median_col = _find_col(df, "Median SBP", "Median DBP", "Median Pulse", "Median HDL",
                           "Median Glucose", "median")
    q1_col = _find_col(df, "SBP Q1", "DBP Q1", "Pulse Q1", "HDL Q1", "Glucose Q1", "Q1")
    q3_col = _find_col(df, "SBP Q3", "DBP Q3", "Pulse Q3", "HDL Q3", "Glucose Q3", "Q3")
    low_col = _find_col(df, "Plasubible_lower_range", "Plausible_lower_range",
                        "plausible_lower_range")
    high_col = _find_col(df, "Plausible_higher_range", "plausible_higher_range")
    norm_low_col = _find_col(df, "Normal_lower_range", "normal_lower_range")
    norm_high_col = _find_col(df, "Normal_higher_Range", "Normal_higher_range",
                              "normal_higher_range")

    if not all([cond_col, gender_col, age_col, median_col, q1_col, q3_col, low_col, high_col]):
        raise ValueError(f"Cannot find required columns in {path}. Columns: {list(df.columns)}")

    rows: List[GroundTruthRow] = []
    for _, row in df.iterrows():
        age_min, age_max = _parse_age_range(row.get(age_col, "0-200"))
        plausible_low = _to_float(row.get(low_col))
        plausible_high = _to_float(row.get(high_col))
        median = _to_float(row.get(median_col))
        q1 = _to_float(row.get(q1_col))
        q3 = _to_float(row.get(q3_col))
        normal_low = _to_float(row.get(norm_low_col)) if norm_low_col else plausible_low
        normal_high = _to_float(row.get(norm_high_col)) if norm_high_col else plausible_high
        if pd.isna(normal_low):
            normal_low = plausible_low
        if pd.isna(normal_high):
            normal_high = plausible_high
        if pd.isna(q1) and not pd.isna(normal_low):
            q1 = normal_low
        if pd.isna(q3) and not pd.isna(normal_high):
            q3 = normal_high
        if pd.isna(median) and not (pd.isna(q1) or pd.isna(q3)):
            median = (q1 + q3) / 2.0
        if any(pd.isna(v) for v in [plausible_low, plausible_high, median, q1, q3]):
            continue
        raw_cond = str(row.get(cond_col, "")).strip()
        cond_str = "None" if raw_cond in ("", "nan", "NaN", "None") else raw_cond
        rows.append(
            GroundTruthRow(
                condition=cond_str,
                gender=str(row.get(gender_col, "")).strip().upper(),
                age_min=age_min,
                age_max=age_max,
                plausible_low=plausible_low,
                plausible_high=plausible_high,
                median=median,
                q1=q1,
                q3=q3,
                normal_low=normal_low,
                normal_high=normal_high,
            )
        )

    if not rows:
        raise ValueError(f"No valid GT rows loaded from {path}")
    print(f"  Loaded {len(rows)} GT rows from {path}")
    return GroundTruthIndex(rows)


# ---------------------------------------------------------------------------
# Healthy baselines (for patients with no matching GT condition)
# ---------------------------------------------------------------------------
HEALTHY_BASELINES = {
    "SBP":     {"median": 120, "q1": 110, "q3": 130, "plausible_low": 60, "plausible_high": 250,
                "normal_low": 90, "normal_high": 129, "condition": "No Hypertension"},
    "DBP":     {"median": 75, "q1": 65, "q3": 80, "plausible_low": 40, "plausible_high": 150,
                "normal_low": 60, "normal_high": 80, "condition": "No Hypertension"},
    "PR":      {"median": 72, "q1": 64, "q3": 80, "plausible_low": 30, "plausible_high": 200,
                "normal_low": 40, "normal_high": 100, "condition": "None"},
    "CH":      {"median": 1.4, "q1": 1.1, "q3": 1.7, "plausible_low": 0.5, "plausible_high": 3.0,
                "normal_low": 1.0, "normal_high": 1.9, "condition": "None"},
    "GLUCOSE": {"median": 5.5, "q1": 5.0, "q3": 6.5, "plausible_low": 0, "plausible_high": 35,
                "normal_low": 3.9, "normal_high": 8.0, "condition": "None"},
}


# ===========================================================================
# Synthetic Data Generation
# ===========================================================================
def generate_synthetic_data(
    gt_index: GroundTruthIndex,
    n_records: int,
    error_rate: float,
    variable: str,
    seed: int,
) -> pd.DataFrame:
    """
    Generate semi-synthetic data with injected errors.

    Normal values:
      - Sampled from a Gaussian centred on the GT/baseline median with
        IQR-derived sigma.
      - Soft-clipped: if a normal sample falls outside the plausible range,
        it is resampled (so normal records stay within plausible bounds).

    Error types (injected into error_rate fraction of records):
      1. Within-range contextual errors (~15%): value is within plausible range
         but outside the NORMAL range.
      2. Borderline errors (~15%): just outside the plausible range (0.01-2 units).
      3. Context-dependent plausibility errors (~20%): from wrong context.
      4. Unit confusion errors (~20%): value * unit-conversion factor.
      5. Sporadic entry errors (~30%): extreme outliers / typos.
    """
    rng = np.random.default_rng(seed)
    var = variable.upper()
    all_gt_rows = gt_index.rows
    healthy = HEALTHY_BASELINES.get(var)

    unit_factors = {"SBP": 0.1333, "DBP": 0.1333, "PR": 1.0, "CH": 38.67, "GLUCOSE": 18.0}
    unit_factor = unit_factors.get(var, 10.0)

    records = []
    n_errors = int(n_records * error_rate)
    error_indices = set(rng.choice(n_records, n_errors, replace=False))

    for i in range(n_records):
        is_error = i in error_indices

        # Pick a GT row or use healthy baseline (50/50 for non-error, biased for error)
        use_healthy = rng.random() < 0.5 and healthy is not None
        if use_healthy:
            gt_row = all_gt_rows[0]
            age = int(rng.integers(18, 100))
            gender = rng.choice(["MALE", "FEMALE"])
            condition = healthy["condition"]
            median = healthy["median"]
            q1 = healthy["q1"]
            q3 = healthy["q3"]
            p_low = healthy["plausible_low"]
            p_high = healthy["plausible_high"]
            # Age effect
            if var in ("SBP", "DBP"):
                median += (age - 50) * 0.3
            elif var == "GLUCOSE":
                median += (age - 50) * 0.02
        else:
            gt_row = random.choice(all_gt_rows)
            age_min = gt_row.age_min
            age_max = gt_row.age_max
            if age_max <= age_min:
                age_max = age_min + 10
            age = int(rng.integers(age_min, age_max))
            gender = gt_row.gender
            condition = gt_row.condition
            median = gt_row.median
            q1 = gt_row.q1
            q3 = gt_row.q3
            p_low = gt_row.plausible_low
            p_high = gt_row.plausible_high

        plausible_range = p_high - p_low
        sigma = max((q3 - q1) / 1.35, 1e-3)

        # --- generate plausible value (soft-clipped to plausible bounds) ---
        for _ in range(10):
            value = float(rng.normal(median, sigma))
            if p_low <= value <= p_high:
                break

        if is_error:
            r = rng.random()
            error_type_label = ""
            if r < 0.15:
                error_type_label = "Type 1: Within-range"
                # --- Type 1: Within-range contextual error ---
                n_low = gt_row.normal_low if not use_healthy else healthy.get("normal_low", q1)
                n_high = gt_row.normal_high if not use_healthy else healthy.get("normal_high", q3)
                if n_high < p_high - 1:
                    gap_above = p_high - n_high
                    value = n_high + rng.uniform(1, gap_above)
                elif n_low > p_low + 1:
                    gap_below = n_low - p_low
                    value = p_low + rng.uniform(0, gap_below - 1)
                else:
                    nudge = rng.uniform(0.01, 2.0)
                    value = p_high + nudge if rng.random() < 0.5 else p_low - nudge

            elif r < 0.30:
                error_type_label = "Type 2: Borderline"
                # --- Type 2: Borderline error ---
                nudge = rng.uniform(0.01, 2.0)
                if rng.random() < 0.5:
                    value = p_high + nudge
                else:
                    value = p_low - nudge

            elif r < 0.50:
                error_type_label = "Type 3: Context-dependent"
                # --- Type 3: Context-dependent plausibility error ---
                if rng.random() < 0.5 and len(all_gt_rows) > 1:
                    other = random.choice(all_gt_rows)
                    other_sigma = max((other.q3 - other.q1) / 1.35, 1e-3)
                    value = float(rng.normal(other.median, other_sigma))
                    if p_low <= value <= p_high:
                        nudge = rng.uniform(0.01, 0.10) * plausible_range
                        value = p_high + nudge if rng.random() < 0.5 else p_low - nudge
                else:
                    nudge = rng.uniform(0.01, 0.15) * plausible_range
                    value = p_high + nudge if rng.random() < 0.5 else p_low - nudge

            elif r < 0.70:
                error_type_label = "Type 4: Unit confusion"
                # --- Type 4: Unit confusion error ---
                value = median * unit_factor
                if p_low <= value <= p_high:
                    value = p_high + rng.uniform(1, 10)

            else:
                error_type_label = "Type 5: Sporadic"
                # --- Type 5: Sporadic entry error ---
                kind = rng.integers(0, 3)
                if kind == 0:
                    value = float(rng.integers(1, 10)) * 1000
                elif kind == 1:
                    value = median * rng.choice([10, 100, 0.1, 0.01])
                else:
                    direction = rng.choice([-1, 1])
                    value = value + direction * rng.uniform(1.5, 4.0) * plausible_range

            # --- VERIFY: for error types 2-5, ensure value is outside plausible ---
            if r >= 0.15 and p_low <= value <= p_high:
                nudge = rng.uniform(0.01, 2.0)
                value = p_high + nudge if rng.random() < 0.5 else p_low - nudge
                is_error_flag = 1
            else:
                is_error_flag = 1
        else:
            is_error_flag = 0
            error_type_label = "Normal"

        records.append({
            "value": round(value, 4),
            "age": age,
            "gender": gender,
            "condition": condition,
            "gender_concept_id": GENDER_CID.get(gender, 0),
            "condition_concept_id": CONDITION_LABEL_TO_CID.get(condition, 0),
            "gt_low": p_low,
            "gt_high": p_high,
            "is_error": is_error_flag,
            "error_type": error_type_label,
        })

    df = pd.DataFrame(records)

    # Bucket age
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 19, 29, 39, 49, 59, 69, 79, 89, 200], 
        labels=["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    ).astype(str)

    print(f"  Generated {len(df):,} records | {df['is_error'].sum():,} errors ({100*df['is_error'].mean():.1f}%)")
    return df


# ===========================================================================
# OMOP Data Loading
# ===========================================================================
def get_data_from_omop(
    variable: str,
    db_params: Dict,
    schema: str,
    gt_index: GroundTruthIndex,
    row_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load real-world data from an OMOP CDM database.
    Uses condition_concept_id as a raw numeric feature.
    """
    # from sqlalchemy import create_engine
    # from sqlalchemy.engine import URL

    var = variable.upper()
    
    # Check cache first
    limit_str = f"_{row_limit}" if row_limit else "_all"
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    schema_name = schema if schema else "public"
    cache_file = os.path.join(cache_dir, f"{var}_omop_{schema_name}{limit_str}.parquet")

    if os.path.exists(cache_file):
        print(f"  [CACHE] Loading data from local cache: {cache_file}")
        df = pd.read_parquet(cache_file)
        n_errors = df["is_error"].sum()
        print(f"  Loaded {len(df):,} records | {n_errors:,} implausible ({100*n_errors/len(df):.2f}%)")
        return df

    # Smart Subsampling Fallback
    if row_limit is not None:
        all_cache_file = os.path.join(cache_dir, f"{var}_omop_{schema_name}_all.parquet")
        if os.path.exists(all_cache_file):
            print(f"  [CACHE-FALLBACK] Full dataset found at {all_cache_file}. Subsampling {row_limit} rows from it instead of querying OMOP...")
            df_all = pd.read_parquet(all_cache_file)
            if len(df_all) > row_limit:
                df = df_all.sample(n=row_limit, random_state=42).reset_index(drop=True)
            else:
                df = df_all
            
            # Save the new limited cache file for future runs
            df.to_parquet(cache_file)
            n_errors = df["is_error"].sum()
            print(f"  Saved subsampled cache: {cache_file}")
            print(f"  Loaded {len(df):,} records | {n_errors:,} implausible ({100*n_errors/max(1, len(df)):.2f}%)")
            return df

    meas_ids = ", ".join(str(i) for i in MEASUREMENT_CONCEPT_IDS[var])
    cond_map = CONDITION_CONCEPT_IDS.get(var, {})
    default_cond_label = DEFAULT_CONDITION_LABEL.get(var, "None")
    schema_prefix = f"{schema}." if schema else ""
    limit_clause = f"\nLIMIT {int(row_limit)}" if row_limit else ""

    # Build CASE WHEN chain for conditions
    if cond_map:
        case_whens = []
        for cid, label in cond_map.items():
            case_whens.append(
                f"""WHEN EXISTS (
                    SELECT 1 FROM {schema_prefix}condition_occurrence co
                    WHERE co.person_id = b.person_id AND co.condition_concept_id = {cid}
                ) THEN {cid}"""
            )
        condition_case = "CASE\n            " + "\n            ".join(case_whens) + "\n            ELSE 0\n        END"
    else:
        condition_case = "0"

    sql = f"""
        WITH base AS (
            SELECT
                m.person_id,
                m.value_as_number AS measurement_value,
                EXTRACT(YEAR FROM m.measurement_date) - p.year_of_birth AS age,
                p.gender_concept_id
            FROM {schema_prefix}measurement m
            JOIN {schema_prefix}person p ON p.person_id = m.person_id
            WHERE m.value_as_number IS NOT NULL
              AND m.measurement_concept_id IN ({meas_ids})
        )
        SELECT
            b.person_id,
            b.measurement_value,
            b.age,
            b.gender_concept_id,
            {condition_case} AS condition_concept_id
        FROM base b{limit_clause}
    """

    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    host = os.getenv("OMOP_DB_HOST") or db_params.get("host", "localhost")
    port = int(os.getenv("OMOP_DB_PORT") or db_params.get("port", 5432))
    dbname = os.getenv("OMOP_DB_NAME") or db_params.get("dbname", "your_database")
    user = os.getenv("OMOP_DB_USER") or db_params.get("user", "your_username")
    password = os.getenv("OMOP_DB_PASSWORD") or db_params.get("password", "your_password")
    db_url_env = os.getenv("OMOP_DB_URL")

    if db_url_env:
        engine = create_engine(db_url_env, pool_pre_ping=True)
    elif dbname != "your_database":
        url = URL.create(
            drivername="postgresql+psycopg2",
            username=user,
            password=password,
            host=host,
            port=port,
            database=dbname,
        )
        engine = create_engine(url, pool_pre_ping=True)
    else:
        raise RuntimeError(
            f"OMOP Database connection is not configured and cache file for '{var}' was not found in 'data_cache/'.\n"
            f"Please update DB_CONFIG in cagpre.engines.shared or set environment variables:\n"
            f"  OMOP_DB_HOST, OMOP_DB_NAME, OMOP_DB_USER, OMOP_DB_PASSWORD, OMOP_DB_PORT, OMOP_DB_SCHEMA\n"
            f"or set OMOP_DB_URL=postgresql://user:pass@host:port/dbname"
        )

    print(f"  Querying OMOP for {var} ...")
    df = pd.read_sql_query(sql, engine)
    engine.dispose()
    print(f"  Loaded {len(df):,} rows from DB")

    if df.empty:
        raise ValueError(f"No data returned for {var}")

    df = df.dropna(subset=["measurement_value", "age"])
    df["age"] = df["age"].astype(int)
    df["gender_concept_id"] = df["gender_concept_id"].astype(int)
    df["condition_concept_id"] = df["condition_concept_id"].astype(int)

    # Filter adults
    df = df[df["age"] >= 18].copy()

    # Bucket age
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 19, 29, 39, 49, 59, 69, 79, 89, 200], 
        labels=["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    ).astype(str)

    # Map concept IDs to GT-compatible strings
    cid_to_label = {v_cid: v_label for v_cid, v_label in cond_map.items()}
    cid_to_label[0] = default_cond_label
    df["gender"] = df["gender_concept_id"].map(GENDER_CID_REVERSE).fillna("MALE")
    df["condition"] = df["condition_concept_id"].map(cid_to_label).fillna(default_cond_label)

    df = df.rename(columns={"measurement_value": "value"})

    # Vectorized GT lookup for is_error labels
    print(f"  Looking up GT plausible ranges for {len(df):,} records (Vectorized) ...")
    df["gt_low"] = np.nan
    df["gt_high"] = np.nan

    if gt_index.rows:
        # Pass 1: Global fallback (using the last rule as default)
        fallback = gt_index.rows[-1]
        df["gt_low"] = fallback.plausible_low
        df["gt_high"] = fallback.plausible_high

        # Pass 2: Apply Condition + Age matches
        for r in gt_index.rows:
            mask = (df["condition"].str.lower() == r.condition.lower()) & \
                   (df["age"] >= r.age_min) & (df["age"] < r.age_max)
            df.loc[mask, "gt_low"] = r.plausible_low
            df.loc[mask, "gt_high"] = r.plausible_high

        # Pass 3: Apply Condition + Gender + Age matches (most specific)
        for r in gt_index.rows:
            mask = (df["condition"].str.lower() == r.condition.lower()) & \
                   (df["gender"].str.upper() == r.gender.upper()) & \
                   (df["age"] >= r.age_min) & (df["age"] < r.age_max)
            df.loc[mask, "gt_low"] = r.plausible_low
            df.loc[mask, "gt_high"] = r.plausible_high

    df["is_error"] = ((df["value"] < df["gt_low"]) | (df["value"] > df["gt_high"])).astype(int)

    n_errors = df["is_error"].sum()
    print(f"  Final: {len(df):,} records | {n_errors:,} implausible ({100*n_errors/len(df):.2f}%)")
    
    print(f"  [CACHE] Saving data to local cache: {cache_file}")
    df.to_parquet(cache_file, index=False)
    
    return df


# ===========================================================================
# Preprocessing
# ===========================================================================
class TabularPreprocessor:
    """Handles encoding of categorical context + scaling of target and numeric features."""

    def __init__(self, cat_cols: List[str] = None, num_cols: List[str] = None, encoding_type: str = "onehot"):
        self.cat_cols = cat_cols or ["gender", "condition", "age_group"]
        self.num_cols = num_cols if num_cols is not None else []
        self.encoding_type = encoding_type
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        from sklearn.preprocessing import OrdinalEncoder
        self.ord = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.num_scaler = RobustScaler()
        # QuantileTransformer maps ALL values (including extreme outliers) to
        # a well-behaved Gaussian. A value of 160,148 mmHg gets mapped to ~3.5
        # instead of 8,001 with RobustScaler. This is the key to tight ranges.
        # self.target_scaler = QuantileTransformer(
        #     output_distribution="uniform", n_quantiles=1000, random_state=42
        # )
        self.target_scaler = QuantileTransformer(
            output_distribution="uniform",  # The silver bullet: fixes the 13.0 ceiling
            n_quantiles=10000,              # Increased resolution for the extreme tails
            subsample=10_000_000,           # Forces it to look at the millions of rows
            random_state=42
        )
        self._fitted = False
        self.cat_dims = []

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        if self.encoding_type == "embedding":
            self.ord.fit(df[self.cat_cols])
            self.cat_dims = [len(c) for c in self.ord.categories_]
        else:
            self.ohe.fit(df[self.cat_cols])
        
        if self.num_cols:
            self.num_scaler.fit(df[self.num_cols].values)
        self.target_scaler.fit(df[["value"]].values)
        self._fitted = True
        return self
        
    def get_category_sizes(self) -> List[int]:
        return self.cat_dims

    def transform_context(self, df: pd.DataFrame) -> np.ndarray:
        num = np.zeros((len(df), 0), dtype=np.float32)
        if self.num_cols:
            num = self.num_scaler.transform(df[self.num_cols].values).astype(np.float32)
            
        if self.encoding_type == "embedding":
            cat = self.ord.transform(df[self.cat_cols])
            # Clip -1 to 0 for unknown values so embeddings don't crash
            cat = np.clip(cat, 0, None).astype(np.float32)
            return np.hstack([num, cat])
        else:
            cat = self.ohe.transform(df[self.cat_cols]).astype(np.float32)
            return np.hstack([num, cat]) if num.shape[1] > 0 else cat

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(values.reshape(-1, 1)).flatten().astype(np.float32)

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()


def prepare_datasets(
    df: pd.DataFrame, extra_context: List[str], seed: int, encoding_type: str = "onehot", ignore_condition: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TabularPreprocessor]:
    # --- HOTFIX: Ensure age_group exists for cached real datasets ---
    if "age_group" not in df.columns and "age" in df.columns:
        df = df.copy()
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 19, 29, 39, 49, 59, 69, 79, 89, 200], 
            labels=["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
        ).astype(str)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    print(f"  Split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    cat_cols = ["gender", "condition", "age_group"]
    if ignore_condition:
        cat_cols = ["gender", "age_group"]
    num_cols = extra_context
    preprocessor = TabularPreprocessor(cat_cols=cat_cols, num_cols=num_cols, encoding_type=encoding_type)
    preprocessor.fit(train_df)
    return train_df, val_df, test_df, preprocessor


# ===========================================================================
# Model Architectures
# ===========================================================================
class CAGPRECVAE(nn.Module):
    """Conditional VAE with Gaussian NLL loss (learned output variance).

    Key differences from the original MSE-based CVAE:
    - Decoder outputs (recon_mu, recon_logvar) instead of a single value
    - BatchNorm + Dropout at every layer for stability
    - Tanh-bounded output logvar to prevent explosion
    - Gaussian NLL loss naturally handles extreme outliers by learning variance
    """

    def __init__(self, context_dim: int, hidden_dim: int = 256, latent_dim: int = 16,
                 dropout_rate: float = 0.2):
        super().__init__()
        input_dim = 1 + context_dim

        # --- ENCODER ---
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 4, latent_dim)

        # --- DECODER ---
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Two-headed output: mean and log-variance
        self.dec_mu = nn.Linear(hidden_dim, 1)
        self.dec_logvar = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.encoder_net(torch.cat([x, c], dim=1))
        mu = self.mu(h)
        logvar = self.logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder_net(torch.cat([z, c], dim=1))
        recon_mu = self.dec_mu(h)
        # Bound output logvar with tanh to prevent explosion (maps to [-4, 4])
        recon_logvar_raw = self.dec_logvar(h)
        recon_logvar = 4.0 * torch.tanh(recon_logvar_raw / 4.0)
        return recon_mu, recon_logvar

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_mu, recon_logvar = self.decode(z, c)
        return recon_mu, recon_logvar, mu, logvar








# ===========================================================================
# Model save / load
# ===========================================================================
def save_model(model: nn.Module, path: str, meta: Dict = None) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)
    print(f"  Model saved to {path}")


def load_model(model: nn.Module, path: str, device: str) -> Dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return checkpoint.get("meta", {})


# ===========================================================================
# Metrics
# ===========================================================================
from sklearn.metrics import confusion_matrix

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Calculate Accuracy and Specificity for reporting on highly imbalanced data
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "accuracy": float(accuracy), "specificity": float(specificity),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }


def compute_iou(pred_low: np.ndarray, pred_high: np.ndarray,
                gt_low: np.ndarray, gt_high: np.ndarray) -> float:
    intersection = np.maximum(0.0, np.minimum(pred_high, gt_high) - np.maximum(pred_low, gt_low))
    union = np.maximum(pred_high, gt_high) - np.minimum(pred_low, gt_low)
    iou = np.where(union > 0, intersection / union, 0.0)
    return float(np.nanmean(iou))


def compute_picp(pred_low: np.ndarray, pred_high: np.ndarray,
                 values: np.ndarray) -> float:
    """Prediction Interval Coverage Probability.
    What fraction of actual values fall within the predicted range."""
    inside = (values >= pred_low) & (values <= pred_high)
    return float(np.mean(inside))


def compute_mpiw(pred_low: np.ndarray, pred_high: np.ndarray) -> float:
    """Mean Prediction Interval Width. Lower = tighter ranges = better."""
    widths = pred_high - pred_low
    return float(np.mean(widths))


def compute_dice(pred_low: np.ndarray, pred_high: np.ndarray,
                 gt_low: np.ndarray, gt_high: np.ndarray) -> float:
    """Dice coefficient for range overlap. Less harsh than IoU on width mismatches."""
    intersection = np.maximum(0.0, np.minimum(pred_high, gt_high) - np.maximum(pred_low, gt_low))
    pred_width = pred_high - pred_low
    gt_width = gt_high - gt_low
    denom = pred_width + gt_width
    dice = np.where(denom > 0, 2.0 * intersection / denom, 0.0)
    return float(np.nanmean(dice))


def compute_containment(pred_low: np.ndarray, pred_high: np.ndarray,
                        gt_low: np.ndarray, gt_high: np.ndarray) -> float:
    """What fraction of the predicted range falls within the GT range.
    1.0 = predicted range is a perfect subset of GT (valid but possibly tighter)."""
    intersection = np.maximum(0.0, np.minimum(pred_high, gt_high) - np.maximum(pred_low, gt_low))
    pred_width = pred_high - pred_low
    containment = np.where(pred_width > 0, intersection / pred_width, 1.0)
    return float(np.nanmean(containment))


def compute_cwc(pred_low: np.ndarray, pred_high: np.ndarray,
                values: np.ndarray, target_coverage: float = 0.95,
                eta: float = 50.0) -> float:
    """Coverage Width-based Criterion. Lower = better.
    Balances range width with a penalty if coverage drops below target."""
    mpiw = compute_mpiw(pred_low, pred_high)
    picp = compute_picp(pred_low, pred_high, values)
    if picp >= target_coverage:
        penalty = 0.0
    else:
        penalty = np.exp(-eta * (picp - target_coverage))
    return float(mpiw * (1.0 + penalty))


def compute_all_range_metrics(
    pred_low: np.ndarray, pred_high: np.ndarray,
    gt_low: np.ndarray, gt_high: np.ndarray,
    values: np.ndarray,
) -> Dict[str, float]:
    """Compute all range evaluation metrics at once."""
    return {
        "iou": compute_iou(pred_low, pred_high, gt_low, gt_high),
        "dice": compute_dice(pred_low, pred_high, gt_low, gt_high),
        "picp": compute_picp(pred_low, pred_high, values),
        "mpiw": compute_mpiw(pred_low, pred_high),
        "containment": compute_containment(pred_low, pred_high, gt_low, gt_high),
        "cwc": compute_cwc(pred_low, pred_high, values),
    }

# ===========================================================================
# Training
# ===========================================================================
def train_cagpre(
    model: CAGPRECVAE,
    x_train: np.ndarray, c_train: np.ndarray,
    x_val: np.ndarray, c_val: np.ndarray,
    epochs: int, lr: float, device: str,
    batch_size: int = 1024,
    patience: int = 15,
    beta_final: float = 0.5,
) -> None:
    """Train CAGPRE CVAE with Gaussian NLL loss.

    Args:
        beta_final: Maximum KL weight after annealing. Lower = tighter ranges.
                    The optuna file uses 0.5. Default MSE-based pipeline used 1.0.
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    n_train = x_train.shape[0]

    # --- Clip scaled values to prevent NaN (no data removed, just bounded) ---
    # With QuantileTransformer, most values are in [-3.5, 3.5] so CLIP=5 is safe
    CLIP = 5.0
    n_clipped_x = int(np.sum(np.abs(x_train) > CLIP))
    n_clipped_c = int(np.sum(np.abs(c_train) > CLIP))
    x_train_clipped = np.clip(x_train, -CLIP, CLIP)
    c_train_clipped = np.clip(c_train, -CLIP, CLIP)
    x_val_clipped = np.clip(x_val, -CLIP, CLIP)
    c_val_clipped = np.clip(c_val, -CLIP, CLIP)
    if n_clipped_x > 0 or n_clipped_c > 0:
        print(f"    [INFO] Clipped scaled values for training stability: "
              f"target={n_clipped_x}/{len(x_train)} ({100*n_clipped_x/len(x_train):.2f}%), "
              f"context={n_clipped_c}/{c_train.size} ({100*n_clipped_c/c_train.size:.2f}%)")
        print(f"    [INFO] Original range: x=[{x_train.min():.1f}, {x_train.max():.1f}]  "
              f"c=[{c_train.min():.1f}, {c_train.max():.1f}]")

    nan_streak = 0  # Track consecutive all-NaN epochs
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    anneal_epochs = max(1, int(epochs * 0.3))
    for epoch in range(epochs):
        # ---- Training step ----
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        # KL annealing: ramp from 0 to beta_final over first 30% of epochs
        kl_weight = min(beta_final, beta_final * (epoch + 1) / anneal_epochs)

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x_b = torch.tensor(x_train_clipped[idx], dtype=torch.float32, device=device).unsqueeze(1)
            c_b = torch.tensor(c_train_clipped[idx], dtype=torch.float32, device=device)
            recon_mu, recon_logvar, mu, logvar = model(x_b, c_b)
            # Gaussian NLL: naturally handles extreme outliers by learning variance
            recon_logvar_stable = torch.clamp(recon_logvar, min=-10.0, max=10.0)
            recon_var = torch.exp(recon_logvar_stable) + 1e-6
            nll_loss = 0.5 * torch.mean(recon_logvar_stable + (x_b - recon_mu).pow(2) / recon_var)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = nll_loss + kl_weight * kl_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            nan_streak += 1
            if nan_streak >= 3 and epoch < 5:
                # All batches NaN for multiple early epochs -- reinit weights
                print(f"    [WARN] Epoch {epoch+1}: All NaN. Reinitializing model weights ...")
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                optimizer = optim.Adam(model.parameters(), lr=lr * 0.5)
                nan_streak = 0
            continue  # skip validation this epoch
        else:
            nan_streak = 0

        # ---- Validation step ----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for start in range(0, len(x_val_clipped), batch_size):
                end = min(start + batch_size, len(x_val_clipped))
                x_vb = torch.tensor(x_val_clipped[start:end], dtype=torch.float32, device=device).unsqueeze(1)
                c_vb = torch.tensor(c_val_clipped[start:end], dtype=torch.float32, device=device)
                recon_mu, recon_logvar, mu, logvar = model(x_vb, c_vb)
                recon_logvar_stable = torch.clamp(recon_logvar, min=-10.0, max=10.0)
                recon_var = torch.exp(recon_logvar_stable) + 1e-6
                nll_loss = 0.5 * torch.mean(recon_logvar_stable + (x_vb - recon_mu).pow(2) / recon_var)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                v_loss = nll_loss + kl_weight * kl_loss
                if not (torch.isnan(v_loss) or torch.isinf(v_loss)):
                    val_loss += v_loss.item()
                    val_batches += 1

        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
        else:
            avg_val_loss = float("nan")

        # Step the LR scheduler
        if val_batches > 0:
            scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs}  train_loss={epoch_loss/n_batches:.4f}  "
                  f"val_loss={avg_val_loss:.4f}  kl_w={kl_weight:.2f}  lr={current_lr:.1e}  "
                  f"patience={patience - epochs_no_improve}")

        # ---- Early stopping check ----
        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"    Restored best model (val_loss={best_val_loss:.4f})")


def _hdr_interval(samples: np.ndarray, coverage: float) -> Tuple[float, float]:
    """Highest Density Region: find the narrowest interval containing `coverage`
    fraction of the samples. Better than symmetric percentiles for skewed data."""
    sorted_s = np.sort(samples)
    n = len(sorted_s)
    k = max(1, int(np.ceil(coverage * n)))
    # Slide a window of size k and find the shortest one
    widths = sorted_s[k:] - sorted_s[:n - k]
    if len(widths) == 0:
        return float(sorted_s[0]), float(sorted_s[-1])
    best_idx = int(np.argmin(widths))
    return float(sorted_s[best_idx]), float(sorted_s[best_idx + k])


def generate_ranges(
    model: CAGPRECVAE,
    context: np.ndarray,
    preprocessor: TabularPreprocessor,
    n_samples: int,
    alpha: float,
    device: str,
    batch_size: int = 512,
    range_method: str = "hdr",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate plausible ranges using batched decoding.

    Args:
        range_method: 'percentile' for symmetric alpha/2 percentiles,
                      'hdr' for Highest Density Region (narrowest interval).
    """
    model.eval()
    n_rows = context.shape[0]
    low_arr = np.empty(n_rows)
    high_arr = np.empty(n_rows)
    latent_dim = model.mu.out_features
    coverage = 1.0 - alpha
    # For 'gaussian' method: k multiplier from alpha (e.g., alpha=0.05 → k≈1.96)
    from scipy.stats import norm as _norm
    k_sigma = _norm.ppf(1.0 - alpha / 2.0) if alpha > 0 else 4.0

    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            chunk = context[start:end]
            chunk_len = end - start
            c_t = torch.tensor(chunk, dtype=torch.float32, device=device)

            # Sub-batch samples to avoid CUDA OOM with large n_samples
            sample_chunk_size = 2000
            all_decoded = []
            for s_start in range(0, n_samples, sample_chunk_size):
                s_end = min(s_start + sample_chunk_size, n_samples)
                s_count = s_end - s_start
                c_rep = c_t.unsqueeze(1).expand(-1, s_count, -1).reshape(-1, c_t.shape[1])
                z = torch.randn(chunk_len * s_count, latent_dim, device=device)
                recon_mu, recon_logvar = model.decode(z, c_rep)
                std = torch.exp(0.5 * recon_logvar)
                eps = torch.randn_like(std)
                dec = (recon_mu + eps * std).squeeze(1).cpu().numpy()
                dec = preprocessor.inverse_target(dec)
                all_decoded.append(dec.reshape(chunk_len, s_count))
            decoded = np.concatenate(all_decoded, axis=1)  # (chunk_len, n_samples)

            if range_method == "hdr":
                for j in range(chunk_len):
                    lo, hi = _hdr_interval(decoded[j], coverage)
                    low_arr[start + j] = lo
                    high_arr[start + j] = hi
            elif range_method == "gaussian":
                # Use decoder's learned variance directly: μ ± k·σ
                # This gives WIDER ranges than percentiles because σ captures
                # the full plausible spread, not just the sample concentration
                for j in range(chunk_len):
                    row_mean = np.mean(decoded[j])
                    row_std = np.std(decoded[j])
                    low_arr[start + j] = row_mean - k_sigma * row_std
                    high_arr[start + j] = row_mean + k_sigma * row_std
            else:  # percentile
                low_arr[start:end] = np.quantile(decoded, alpha / 2.0, axis=1)
                high_arr[start:end] = np.quantile(decoded, 1.0 - alpha / 2.0, axis=1)

    return low_arr, high_arr


def generate_ranges_and_scores(
    model: CAGPRECVAE,
    context: np.ndarray,
    preprocessor: TabularPreprocessor,
    observed_values: np.ndarray,
    n_samples: int,
    alpha: float,
    device: str,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate plausible ranges AND implausibility scores (Eq. 1 from paper).

    Returns (pred_low, pred_high, implausibility_scores).
    The implausibility score = -log p(x_obs | C) estimated via KDE on generated samples.
    """
    model.eval()
    n_rows = context.shape[0]
    low_arr = np.empty(n_rows)
    high_arr = np.empty(n_rows)
    scores = np.empty(n_rows)
    latent_dim = model.mu.out_features

    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            chunk = context[start:end]
            chunk_len = end - start
            c_t = torch.tensor(chunk, dtype=torch.float32, device=device)

            # Sub-batch samples to avoid CUDA OOM
            sample_chunk_size = 2000
            all_decoded = []
            for s_start in range(0, n_samples, sample_chunk_size):
                s_end = min(s_start + sample_chunk_size, n_samples)
                s_count = s_end - s_start
                c_rep = c_t.unsqueeze(1).expand(-1, s_count, -1).reshape(-1, c_t.shape[1])
                z = torch.randn(chunk_len * s_count, latent_dim, device=device)
                recon_mu, recon_logvar = model.decode(z, c_rep)
                std = torch.exp(0.5 * recon_logvar)
                eps = torch.randn_like(std)
                dec = (recon_mu + eps * std).squeeze(1).cpu().numpy()
                dec = preprocessor.inverse_target(dec)
                all_decoded.append(dec.reshape(chunk_len, s_count))
            decoded = np.concatenate(all_decoded, axis=1)

            low_arr[start:end] = np.quantile(decoded, alpha / 2.0, axis=1)
            high_arr[start:end] = np.quantile(decoded, 1.0 - alpha / 2.0, axis=1)

            # Compute implausibility score per row using KDE
            for j in range(chunk_len):
                row_samples = decoded[j]
                x_obs = observed_values[start + j]
                try:
                    kde = gaussian_kde(row_samples)
                    density = kde(np.array([x_obs]))[0]
                    # Avoid log(0)
                    density = max(density, 1e-300)
                    scores[start + j] = -np.log(density)
                except Exception:
                    scores[start + j] = float("inf")

    return low_arr, high_arr, scores






# ===========================================================================
# Evaluation
# ===========================================================================
def evaluate_cagpre(
    model: CAGPRECVAE,
    preprocessor: TabularPreprocessor,
    df: pd.DataFrame,
    n_samples: int,
    alpha: float,
    device: str,
    return_ranges: bool = False,
    return_scores: bool = False,
    range_method: str = "hdr",
) -> Union[Dict[str, float], Tuple]:
    c = preprocessor.transform_context(df)
    values = df["value"].values

    if return_scores:
        pred_low, pred_high, imp_scores = generate_ranges_and_scores(
            model, c, preprocessor, values, n_samples, alpha, device
        )
    else:
        pred_low, pred_high = generate_ranges(
            model, c, preprocessor, n_samples, alpha, device, range_method=range_method
        )
        imp_scores = None

    # --- Diagnostics ---
    n_nan_low = np.isnan(pred_low).sum()
    n_nan_high = np.isnan(pred_high).sum()
    n_errors_in_eval = int(df["is_error"].sum())
    n_total = len(df)
    print(f"    [DIAG] Eval set: {n_total} rows, {n_errors_in_eval} errors ({100*n_errors_in_eval/max(n_total,1):.2f}%)")
    print(f"    [DIAG] Pred range NaN: low={n_nan_low}, high={n_nan_high}")
    if n_nan_low == 0 and n_nan_high == 0:
        print(f"    [DIAG] Pred ranges: low=[{np.min(pred_low):.2f}, {np.max(pred_low):.2f}]  "
              f"high=[{np.min(pred_high):.2f}, {np.max(pred_high):.2f}]")
        print(f"    [DIAG] GT   ranges: low=[{df['gt_low'].min():.2f}, {df['gt_low'].max():.2f}]  "
              f"high=[{df['gt_high'].min():.2f}, {df['gt_high'].max():.2f}]")
        print(f"    [DIAG] Values:      [{df['value'].min():.2f}, {df['value'].max():.2f}]  "
              f"mean={df['value'].mean():.2f}")

    y_pred = (values < pred_low) | (values > pred_high)
    metrics = compute_metrics(df["is_error"].values, y_pred.astype(int))
    metrics["iou"] = compute_iou(pred_low, pred_high, df["gt_low"].values, df["gt_high"].values)

    n_pred_errors = int(y_pred.sum())
    print(f"    [DIAG] Predicted errors: {n_pred_errors} / {n_total}")

    if return_ranges and return_scores:
        return metrics, pred_low, pred_high, imp_scores
    elif return_ranges:
        return metrics, pred_low, pred_high
    elif return_scores:
        return metrics, imp_scores
    return metrics


def run_calibration_experiment(
    model: CAGPRECVAE,
    preprocessor: TabularPreprocessor,
    df: pd.DataFrame,
    n_samples: int,
    device: str,
    range_method: str = "hdr",
    save_dir: str = None,
    coverage_levels: List[float] = None,
) -> pd.DataFrame:
    """Calibration experiment: check if the model's p% range contains p% of
    non-erroneous values.

    For each coverage level (e.g., 60%, 65%, ..., 95%), generate plausible
    ranges with alpha = 1 - coverage, filter to known plausible values
    (is_error == 0), and compute the empirical fraction that falls within
    the predicted range.

    A well-calibrated model shows empirical coverage ≈ theoretical coverage.

    Returns a DataFrame with columns: theoretical_coverage, empirical_coverage.
    If save_dir is provided, saves a calibration plot and CSV file.
    """
    if coverage_levels is None:
        coverage_levels = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # Filter to non-erroneous (plausible) values only
    plausible_mask = df["is_error"].values == 0
    df_plausible = df[plausible_mask].reset_index(drop=True)
    n_plausible = len(df_plausible)
    print(f"\n  [CALIBRATION] Using {n_plausible} non-erroneous values "
          f"(out of {len(df)} total, {100*n_plausible/len(df):.1f}%)")

    c_plausible = preprocessor.transform_context(df_plausible)
    values_plausible = df_plausible["value"].values

    results = []
    for coverage in coverage_levels:
        alpha = 1.0 - coverage
        pred_low, pred_high = generate_ranges(
            model, c_plausible, preprocessor, n_samples, alpha, device,
            range_method=range_method,
        )
        # Empirical coverage: fraction of plausible values inside predicted range
        inside = (values_plausible >= pred_low) & (values_plausible <= pred_high)
        empirical_cov = float(np.mean(inside))
        results.append({
            "theoretical_coverage": coverage,
            "empirical_coverage": empirical_cov,
            "n_inside": int(inside.sum()),
            "n_total": n_plausible,
        })
        print(f"    Coverage {coverage*100:.0f}%: empirical={empirical_cov*100:.1f}%  "
              f"({int(inside.sum())}/{n_plausible})")

    results_df = pd.DataFrame(results)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "calibration_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"    Saved calibration CSV to {csv_path}")

        # Generate calibration plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            x = results_df["theoretical_coverage"].values * 100
            y = results_df["empirical_coverage"].values * 100

            # Perfect calibration line
            ax.plot([55, 100], [55, 100], "k--", linewidth=1, label="Perfect calibration")

            # Actual calibration
            ax.plot(x, y, "bo-", markersize=8, linewidth=2, label="CAGPRE model")

            ax.set_xlabel("Theoretical Coverage (%)", fontsize=12)
            ax.set_ylabel("Empirical Coverage (%)", fontsize=12)
            ax.set_title("Calibration Curve", fontsize=14)
            ax.legend(fontsize=11)
            ax.set_xlim(55, 100)
            ax.set_ylim(55, 100)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")
            fig.tight_layout()

            plot_path = os.path.join(save_dir, "calibration_plot.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"    Saved calibration plot to {plot_path}")
        except ImportError:
            print("    [WARN] matplotlib not available, skipping plot")

    return results_df


def compute_topk_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    gt_low: np.ndarray,
    gt_high: np.ndarray,
    pred_low: np.ndarray,
    pred_high: np.ndarray,
    k_percentages: List[float] = None,
    save_dir: str = None,
    variable: str = "",
) -> pd.DataFrame:
    """Compute Precision@topK, Recall@topK, F1@topK, IoU@topK.

    Ranks all data points by implausibility score (descending).
    At each K% threshold, the top-K% are labeled as predicted errors.

    Args:
        y_true: Ground truth error labels (0=normal, 1=error).
        scores: Implausibility scores (higher = more suspicious).
        gt_low, gt_high: GT plausible range per row.
        pred_low, pred_high: Model's predicted plausible range per row.
        k_percentages: List of K values as percentages, e.g. [1, 5, 10, 20].
        save_dir: If provided, saves CSV and plot.
        variable: Variable name for titles (e.g. "SBP").

    Returns:
        DataFrame with columns: K%, Precision, Recall, F1, IoU.
    """
    if k_percentages is None:
        k_percentages = [1, 2, 5, 10, 15, 20, 30, 50]

    n = len(y_true)

    # Rank by score (descending — highest score = most suspicious)
    ranked_idx = np.argsort(-scores)

    # Per-row IoU between model ranges and GT ranges
    per_row_iou = np.zeros(n)
    for i in range(n):
        inter = max(0.0, min(pred_high[i], gt_high[i]) - max(pred_low[i], gt_low[i]))
        union = max(pred_high[i], gt_high[i]) - min(pred_low[i], gt_low[i])
        per_row_iou[i] = inter / union if union > 0 else 0.0

    # --- AUC-ROC of implausibility scores (threshold-independent ranking quality) ---
    try:
        from sklearn.metrics import roc_auc_score
        auc_roc = float(roc_auc_score(y_true, scores))
    except Exception:
        auc_roc = float("nan")
    print(f"    AUC-ROC (implausibility ranking): {auc_roc:.4f}")

    results = []
    for k_pct in k_percentages:
        k = max(1, int(np.ceil(k_pct / 100.0 * n)))
        topk_idx = ranked_idx[:k]
        y_pred = np.zeros(n, dtype=int)
        y_pred[topk_idx] = 1

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou_at_k = float(np.mean(per_row_iou[topk_idx]))

        tn = n - tp - fp - fn
        results.append({
            "K_percent": k_pct,
            "K_count": k,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "IoU": round(iou_at_k, 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "AUC_ROC": round(auc_roc, 4),
        })
        print(f"    @top{k_pct:5.1f}% (K={k:6d}): "
              f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  IoU={iou_at_k:.4f}  "
              f"TP={tp}  TN={tn}  FP={fp}  FN={fn}  AUC={auc_roc:.4f}")

    results_df = pd.DataFrame(results)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"topk_metrics{'_' + variable if variable else ''}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"    Saved @topK CSV → {csv_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, precision_recall_curve, auc as sklearn_auc
            from scipy.stats import gaussian_kde as scipy_kde

            var_label = f" ({variable})" if variable else ""
            x = results_df["K_percent"].values

            # ── Figure 1: @topK line plots (P / R / F1 / IoU / AUC panel) ──────────────
            fig1, axes1 = plt.subplots(1, 5, figsize=(28, 5))
            for ax, metric in zip(axes1[:4], ["Precision", "Recall", "F1", "IoU"]):
                ax.plot(x, results_df[metric].values, "bo-", markersize=6, linewidth=2)
                ax.set_xlabel("Top K (%)", fontsize=11)
                ax.set_ylabel(f"{metric}@topK", fontsize=11)
                ax.set_title(f"{metric}@topK{var_label}", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.05)
            axes1[4].axhline(auc_roc, color="#c0392b", linewidth=2.5, linestyle="--")
            axes1[4].set_ylim(-0.05, 1.05)
            axes1[4].set_xlabel("Top K (%)", fontsize=11)
            axes1[4].set_ylabel("AUC-ROC", fontsize=11)
            axes1[4].set_title(f"AUC-ROC{var_label}", fontsize=12)
            axes1[4].text(
                x[len(x) // 2], min(auc_roc + 0.07, 0.97),
                f"AUC = {auc_roc:.3f}", ha="center", fontsize=12,
                color="#c0392b", fontweight="bold"
            )
            axes1[4].grid(True, alpha=0.3)
            fig1.suptitle(f"@topK Metrics{var_label}", fontsize=14, y=1.02)
            fig1.tight_layout()
            plot_path = os.path.join(save_dir, f"topk_plot{'_' + variable if variable else ''}.png")
            fig1.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig1)
            print(f"    Saved @topK line plot → {plot_path}")

            # ── Figure 2: Stacked TP / FP / FN bar chart ─────────────────────────────
            tp_vals = results_df["TP"].values.astype(float)
            fp_vals = results_df["FP"].values.astype(float)
            fn_vals = results_df["FN"].values.astype(float)
            k_labels = [f"{k}%" for k in results_df["K_percent"].values]
            bar_x = np.arange(len(k_labels))
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            b_tp = ax2.bar(bar_x, tp_vals, label="True Positives (TP)",
                           color="#2ecc71", alpha=0.9)
            ax2.bar(bar_x, fp_vals, bottom=tp_vals, label="False Positives (FP)",
                    color="#e74c3c", alpha=0.9)
            ax2.bar(bar_x, fn_vals, bottom=tp_vals + fp_vals, label="False Negatives (FN)",
                    color="#3498db", alpha=0.9)
            for rect, val in zip(b_tp, tp_vals):
                if val > 0:
                    ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 2,
                             f"{int(val)}", ha="center", va="center",
                             fontsize=9, fontweight="bold", color="white")
            ax2.set_xticks(bar_x)
            ax2.set_xticklabels(k_labels, fontsize=11)
            ax2.set_xlabel("Top K%", fontsize=12)
            ax2.set_ylabel("Number of Records", fontsize=12)
            ax2.set_title(f"TP / FP / FN at Each Top-K Threshold{var_label}",
                          fontsize=13, fontweight="bold")
            ax2.legend(fontsize=11, loc="upper left")
            ax2.grid(axis="y", alpha=0.3)
            fig2.tight_layout()
            bar_path = os.path.join(save_dir, f"topk_confusion_bar{'_' + variable if variable else ''}.png")
            fig2.savefig(bar_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"    Saved TP/FP/FN bar chart → {bar_path}")

            # ── Figure 3: Implausibility score distribution (error vs normal) ──────────
            scores_normal = scores[y_true == 0]
            scores_error  = scores[y_true == 1]
            sn = scores_normal[np.isfinite(scores_normal)]
            se = scores_error[np.isfinite(scores_error)]
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            all_finite = np.concatenate([sn, se]) if (len(sn) > 0 and len(se) > 0) \
                         else (sn if len(sn) > 0 else se)
            if len(all_finite) > 1:
                x_lo = np.percentile(all_finite, 1)
                x_hi = np.percentile(all_finite, 99)
                x_grid = np.linspace(x_lo, x_hi, 500)
                if len(sn) > 10:
                    kde_n = scipy_kde(sn, bw_method="scott")
                    ax3.fill_between(x_grid, kde_n(x_grid), alpha=0.35, color="#3498db")
                    ax3.plot(x_grid, kde_n(x_grid), color="#3498db", linewidth=2,
                             label=f"Normal (is_error=0)  n={len(sn):,}")
                if len(se) > 5:
                    kde_e = scipy_kde(se, bw_method="scott")
                    ax3.fill_between(x_grid, kde_e(x_grid), alpha=0.45, color="#e74c3c")
                    ax3.plot(x_grid, kde_e(x_grid), color="#e74c3c", linewidth=2,
                             label=f"Error (is_error=1)  n={len(se):,}")
            ax3.set_xlabel("Implausibility Score  (\u2212log KDE density)", fontsize=12)
            ax3.set_ylabel("Density", fontsize=12)
            ax3.set_title(
                f"Implausibility Score Distribution{var_label}\n"
                f"AUC-ROC = {auc_roc:.3f}  (higher score = more suspicious)",
                fontsize=13, fontweight="bold"
            )
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)
            fig3.tight_layout()
            dist_path = os.path.join(save_dir, f"score_distribution{'_' + variable if variable else ''}.png")
            fig3.savefig(dist_path, dpi=150, bbox_inches="tight")
            plt.close(fig3)
            print(f"    Saved score distribution → {dist_path}")

            # ── Figure 4: Precision-Recall curve with operating points ────────────────
            if len(np.unique(y_true)) == 2:
                prec_c, rec_c, _ = precision_recall_curve(y_true, scores)
                pr_auc = sklearn_auc(rec_c, prec_c)
                fig4, ax4 = plt.subplots(figsize=(8, 7))
                ax4.plot(rec_c, prec_c, color="#8e44ad", linewidth=2.5,
                         label=f"PR Curve  (AUC={pr_auc:.3f})")
                baseline = float(np.mean(y_true))
                ax4.axhline(baseline, color="grey", linewidth=1.5, linestyle="--",
                            label=f"Random baseline = {baseline:.3f}")
                for _, row in results_df.iterrows():
                    ax4.scatter(row["Recall"], row["Precision"], s=70, zorder=5,
                                color="#e67e22", edgecolors="black", linewidths=0.5)
                    ax4.annotate(
                        f"K={row['K_percent']}%",
                        xy=(row["Recall"], row["Precision"]),
                        xytext=(6, 4), textcoords="offset points", fontsize=8
                    )
                ax4.set_xlabel("Recall", fontsize=12)
                ax4.set_ylabel("Precision", fontsize=12)
                ax4.set_title(f"Precision-Recall Curve{var_label}",
                              fontsize=13, fontweight="bold")
                ax4.set_xlim(-0.02, 1.02)
                ax4.set_ylim(-0.02, 1.02)
                ax4.legend(fontsize=11)
                ax4.grid(True, alpha=0.3)
                fig4.tight_layout()
                pr_path = os.path.join(save_dir, f"pr_curve{'_' + variable if variable else ''}.png")
                fig4.savefig(pr_path, dpi=150, bbox_inches="tight")
                plt.close(fig4)
                print(f"    Saved P-R curve → {pr_path}")

            # ── Figure 5: ROC curve ────────────────────────────────────────────────────
            if len(np.unique(y_true)) == 2 and not np.isnan(auc_roc):
                fpr, tpr, _ = roc_curve(y_true, scores)
                fig5, ax5 = plt.subplots(figsize=(8, 7))
                ax5.plot(fpr, tpr, color="#c0392b", linewidth=2.5,
                         label=f"ROC Curve  (AUC={auc_roc:.3f})")
                ax5.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random classifier")
                ax5.set_xlabel("False Positive Rate", fontsize=12)
                ax5.set_ylabel("True Positive Rate (Recall)", fontsize=12)
                ax5.set_title(f"ROC Curve \u2014 Implausibility Score{var_label}",
                              fontsize=13, fontweight="bold")
                ax5.legend(fontsize=11)
                ax5.grid(True, alpha=0.3)
                ax5.set_xlim(-0.02, 1.02)
                ax5.set_ylim(-0.02, 1.02)
                fig5.tight_layout()
                roc_path = os.path.join(save_dir, f"roc_curve{'_' + variable if variable else ''}.png")
                fig5.savefig(roc_path, dpi=150, bbox_inches="tight")
                plt.close(fig5)
                print(f"    Saved ROC curve → {roc_path}")

        except Exception as exc:
            print(f"    [WARN] Plotting error: {exc}")

    return results_df


def evaluate_cagpre_topk(
    model: "CAGPRECVAE",
    preprocessor: TabularPreprocessor,
    df: pd.DataFrame,
    n_samples: int,
    alpha: float,
    device: str,
    k_percentages: List[float] = None,
    save_dir: str = None,
    variable: str = "",
) -> pd.DataFrame:
    """Convenience wrapper: compute implausibility scores + @topK metrics."""
    c = preprocessor.transform_context(df)
    values = df["value"].values

    print(f"\n  [TopK] Computing implausibility scores for {len(df)} rows...")
    pred_low, pred_high, imp_scores = generate_ranges_and_scores(
        model, c, preprocessor, values, n_samples, alpha, device
    )

    print(f"  [TopK] Total errors in data: {int(df['is_error'].sum())} / {len(df)} "
          f"({100 * df['is_error'].mean():.2f}%)")
    print(f"  [TopK] Score range: [{np.min(imp_scores):.2f}, {np.max(imp_scores):.2f}], "
          f"median={np.median(imp_scores):.2f}")

    return compute_topk_metrics(
        y_true=df["is_error"].values,
        scores=imp_scores,
        gt_low=df["gt_low"].values,
        gt_high=df["gt_high"].values,
        pred_low=pred_low,
        pred_high=pred_high,
        k_percentages=k_percentages,
        save_dir=save_dir,
        variable=variable,
    )










# ===========================================================================
# Tuning
# ===========================================================================
def tune_cagpre(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    n_trials: int,
    alpha: float,
    n_samples: int,
    device: str,
    save_dir: str,
) -> Tuple[str, Dict[str, float]]:
    best_score = -1.0
    best_path = ""
    best_metrics = {}

    x_train = preprocessor.transform_target(train_df["value"].values)
    c_train = preprocessor.transform_context(train_df)
    x_val = preprocessor.transform_target(val_df["value"].values)
    c_val = preprocessor.transform_context(val_df)

    for trial in tqdm(range(n_trials), desc="Tuning CAGPRE"):
        hidden_dim = random.choice([128, 256])
        latent_dim = random.choice([8, 16, 32])
        lr = random.choice([1e-3, 5e-4, 2e-4, 1e-4])
        epochs = random.choice([50, 80, 120])
        beta_final = random.choice([0.1, 0.3, 0.5, 0.8, 1.0])
        # Also tune alpha: tighter ranges push some FP, improving recall
        trial_alpha = random.choice([0.001, 0.005, 0.01, 0.02, 0.05, 0.10])
        trial_n_samples = random.choice([5000, 10000, 20000])
        trial_range_method = random.choice(["gaussian", "hdr", "percentile"])

        model = CAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
        train_cagpre(model, x_train, c_train, x_val, c_val, epochs, lr, device,
                     beta_final=beta_final)
        metrics = evaluate_cagpre(model, preprocessor, val_df, trial_n_samples, trial_alpha, device,
                                  range_method=trial_range_method)
        # Combined score: harmonic mean of F1 and IoU
        f1_val = metrics["f1"]
        iou_val = metrics["iou"]
        prec_val = metrics["precision"]
        rec_val = metrics["recall"]
        if f1_val > 0 and iou_val > 0:
            score = 2 * (f1_val * iou_val) / (f1_val + iou_val)
        else:
            score = 0.0
        print(f"  Trial {trial+1}: hidden={hidden_dim} latent={latent_dim} lr={lr} "
              f"epochs={epochs} beta={beta_final} alpha={trial_alpha} "
              f"n_samples={trial_n_samples} method={trial_range_method} => "
              f"P={prec_val:.3f} R={rec_val:.3f} "
              f"F1={f1_val:.3f} IoU={iou_val:.3f} score={score:.3f}")
        if score > best_score:
            best_score = score
            best_metrics = metrics
            save_path = os.path.join(save_dir, "best.pt")
            meta = {
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "lr": lr,
                "epochs": epochs,
                "beta_final": beta_final,
                "alpha": trial_alpha,
                "n_samples": trial_n_samples,
                "range_method": trial_range_method,
            }
            save_model(model, save_path, meta)
            best_path = save_path

    return best_path, best_metrics


# ===========================================================================
# Partial results (resume on crash)
# ===========================================================================
def _load_partial_results(csv_path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        results = {}
        for _, row in df.iterrows():
            model_name = row["Model"]
            # Older comparison CSVs may contain exploratory baselines that
            # are not part of the reported experiments. Exclude them on resume.
            if model_name in {"Context_Group_IQR", "Cond_Autoencoder", "Cond_VAE"}:
                continue
            metrics = {k: v for k, v in row.items() if k != "Model" and pd.notna(v)}
            results[model_name] = metrics
        print(f"  [RESUME] Loaded partial results: {list(results.keys())}")
        return results
    except Exception:
        return {}


def _save_partial_results(results: Dict[str, Dict[str, float]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


# ===========================================================================
# Range Comparison CSV (GT vs ALL models)
# ===========================================================================
def _save_range_comparison(
    test_df: pd.DataFrame,
    cagpre_low: Optional[np.ndarray],
    cagpre_high: Optional[np.ndarray],
    cagpre_scores: Optional[np.ndarray],
    iqr_low: Optional[np.ndarray],
    iqr_high: Optional[np.ndarray],
    csv_path: str,
    model_ranges: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> None:
    """Save per-sample range comparison with IoU for ALL models.

    Args:
        test_df: Test dataframe with 'value', 'is_error', 'gt_low', 'gt_high'.
        cagpre_low/high: CAGPRE generated ranges (kept for backward compat).
        cagpre_scores: Implausibility scores from CAGPRE.
        iqr_low/high: IQR ranges (kept for backward compat).
        csv_path: Output path.
        model_ranges: Dict mapping model_name -> (pred_low, pred_high) arrays.
                      If provided, all models' ranges are saved.
    """
    out = test_df[["age", "gender", "condition", "value", "is_error", "gt_low", "gt_high"]].copy()
    out = out.reset_index(drop=True)

    gt_low_arr = out["gt_low"].values.astype(float)
    gt_high_arr = out["gt_high"].values.astype(float)
    values = out["value"].values

    # --- Helper to add a model's ranges + IoU ---
    def _add_model_range(prefix: str, pred_low: np.ndarray, pred_high: np.ndarray):
        out[f"{prefix}_low"] = np.round(pred_low, 4)
        out[f"{prefix}_high"] = np.round(pred_high, 4)
        inter = np.maximum(0.0, np.minimum(pred_high, gt_high_arr) - np.maximum(pred_low, gt_low_arr))
        union = np.maximum(pred_high, gt_high_arr) - np.minimum(pred_low, gt_low_arr)
        out[f"{prefix}_iou"] = np.round(np.where(union > 0, inter / union, 0.0), 4)

    # --- CAGPRE (backward compat) ---
    if cagpre_low is not None and cagpre_high is not None:
        _add_model_range("cagpre", cagpre_low, cagpre_high)
        out["is_implausible"] = ((values < cagpre_low) | (values > cagpre_high)).astype(int)

    if cagpre_scores is not None:
        out["implausibility_score"] = np.round(cagpre_scores, 4)

    # --- IQR (backward compat) ---
    if iqr_low is not None and iqr_high is not None:
        _add_model_range("iqr", iqr_low, iqr_high)

    # --- All other models from model_ranges dict ---
    if model_ranges:
        for model_name, (m_low, m_high) in model_ranges.items():
            if m_low is not None and m_high is not None:
                # Create a safe column prefix (lowercase, underscores)
                prefix = model_name.lower().replace(" ", "_")
                _add_model_range(prefix, m_low, m_high)

    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    out.to_csv(csv_path, index=False)
    print(f"\nRange comparison saved to {csv_path}  ({len(out)} rows)")

    # --- Print summary for all models that have IoU columns ---
    for col in out.columns:
        if col.endswith("_iou"):
            model_prefix = col.replace("_iou", "")
            print(f"  {model_prefix:20s} IoU: mean={out[col].mean():.4f}  "
                  f"median={out[col].median():.4f}  "
                  f"min={out[col].min():.4f}  max={out[col].max():.4f}")
    if "implausibility_score" in out.columns:
        finite = out["implausibility_score"].replace([np.inf], np.nan).dropna()
        if len(finite) > 0:
            print(f"  Implausibility: mean={finite.mean():.4f}  median={finite.median():.4f}")


# ===========================================================================
# GT-Row-Based Evaluation (for real-world data)
# ===========================================================================
def run_gt_evaluation(
    model: CAGPRECVAE,
    preprocessor: TabularPreprocessor,
    gt_index: GroundTruthIndex,
    real_df: pd.DataFrame,
    n_samples: int,
    alpha: float,
    device: str,
    csv_path: str = "",
) -> Dict[str, float]:
    """
    Evaluate model by iterating over GT rows (like CAG_V4_V2_DBP_optuna.py).
    For each GT row: create synthetic patient, generate range, compute IoU.
    Also computes implausibility scores for actual patient measurements.
    """
    print("\n--- GT-Row-Based Evaluation ---")
    results_rows = []

    for gt_row in gt_index.rows:
        age = (gt_row.age_min + gt_row.age_max) // 2
        gender = gt_row.gender
        condition = gt_row.condition
        condition_cid = CONDITION_LABEL_TO_CID.get(condition, 0)
        gender_cid = GENDER_CID.get(gender, 0)

        # Create synthetic patient context
        patient_df = pd.DataFrame([{
            "value": 0.0,
            "age": age,
            "gender": gender,
            "condition": condition,
            "gender_concept_id": gender_cid,
            "condition_concept_id": condition_cid,
        }])

        c = preprocessor.transform_context(patient_df)

        # Generate samples
        model.eval()
        latent_dim = model.mu.out_features
        with torch.no_grad():
            c_t = torch.tensor(c, dtype=torch.float32, device=device)
            c_rep = c_t.expand(n_samples, -1)
            z = torch.randn(n_samples, latent_dim, device=device)
            recon_mu, recon_logvar = model.decode(z, c_rep)
            std = torch.exp(0.5 * recon_logvar)
            eps = torch.randn_like(std)
            decoded = (recon_mu + eps * std).squeeze(1).cpu().numpy()
            decoded = preprocessor.inverse_target(decoded)

        model_lo = np.percentile(decoded, (alpha / 2.0) * 100)
        model_hi = np.percentile(decoded, (1.0 - alpha / 2.0) * 100)

        gt_lo = gt_row.plausible_low
        gt_hi = gt_row.plausible_high

        # IoU
        inter = max(0.0, min(model_hi, gt_hi) - max(model_lo, gt_lo))
        union = max(model_hi, gt_hi) - min(model_lo, gt_lo)
        iou = inter / union if union > 0 else 0.0

        results_rows.append({
            "Gender": gender,
            "Age_Range": f"{gt_row.age_min}-{gt_row.age_max - 1}",
            "Condition": condition,
            "GT_Low": round(gt_lo, 2),
            "GT_High": round(gt_hi, 2),
            "Model_Low": round(float(model_lo), 2),
            "Model_High": round(float(model_hi), 2),
            "IoU": round(iou, 4),
        })

    results_df = pd.DataFrame(results_rows)
    avg_iou = results_df["IoU"].mean()
    print(f"  Average IoU across {len(results_df)} GT rows: {avg_iou:.4f}")

    # Save
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        # Append AVERAGE row
        summary = pd.concat([
            results_df,
            pd.DataFrame([{
                "Gender": "AVERAGE", "Age_Range": "", "Condition": "",
                "GT_Low": "", "GT_High": "",
                "Model_Low": "", "Model_High": "",
                "IoU": round(avg_iou, 4),
            }])
        ], ignore_index=True)
        summary.to_csv(csv_path, index=False)
        print(f"  GT evaluation saved to: {csv_path}")

    # Now compute implausibility scores for actual real-world patient measurements
    if real_df is not None and len(real_df) > 0 and csv_path:
        print(f"\n  Computing implausibility scores for {len(real_df):,} real measurements ...")
        c_real = preprocessor.transform_context(real_df)
        values_real = real_df["value"].values
        pred_low, pred_high, imp_scores = generate_ranges_and_scores(
            model, c_real, preprocessor, values_real, n_samples, alpha, device
        )

        range_df = real_df[["age", "gender", "condition", "value"]].copy().reset_index(drop=True)
        range_df["gt_low"] = real_df["gt_low"].values
        range_df["gt_high"] = real_df["gt_high"].values
        range_df["cagpre_low"] = np.round(pred_low, 4)
        range_df["cagpre_high"] = np.round(pred_high, 4)
        range_df["is_implausible"] = ((values_real < pred_low) | (values_real > pred_high)).astype(int)
        range_df["implausibility_score"] = np.round(imp_scores, 4)

        # Per-row IoU
        gt_l = range_df["gt_low"].values.astype(float)
        gt_h = range_df["gt_high"].values.astype(float)
        inter = np.maximum(0.0, np.minimum(pred_high, gt_h) - np.maximum(pred_low, gt_l))
        union = np.maximum(pred_high, gt_h) - np.minimum(pred_low, gt_l)
        range_df["cagpre_iou"] = np.round(np.where(union > 0, inter / union, 0.0), 4)

        range_csv = csv_path.replace("gt_eval_", "range_comparison_")
        range_df.to_csv(range_csv, index=False)
        print(f"  Range comparison saved to: {range_csv}  ({len(range_df)} rows)")

    return {"iou": avg_iou}


# ===========================================================================
# Run Compare (simulated data)
# ===========================================================================
def run_compare(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    alpha: float,
    n_samples: int,
    device: str,
    seed: int,
    csv_path: str = "",
    tuned_model_dir: str = "",
) -> Dict[str, Dict[str, float]]:
    results = _load_partial_results(csv_path) if csv_path else {}

    x_train = preprocessor.transform_target(train_df["value"].values)
    c_train = preprocessor.transform_context(train_df)
    x_val = preprocessor.transform_target(val_df["value"].values)
    c_val = preprocessor.transform_context(val_df)

    x_test = preprocessor.transform_target(test_df["value"].values)
    c_test = preprocessor.transform_context(test_df)

    # --- CAGPRE ---
    cagpre_low, cagpre_high, cagpre_scores = None, None, None
    if "CAGPRE" not in results:
        t0 = time.time()
        tuned_path = os.path.join(tuned_model_dir, "best.pt") if tuned_model_dir else ""
        if tuned_path and os.path.exists(tuned_path):
            print(f"Loading tuned CAGPRE from {tuned_path} ...")
            checkpoint = torch.load(tuned_path, map_location=device, weights_only=False)
            meta = checkpoint.get("meta", {})
            h_dim = meta.get("hidden_dim", 128)
            l_dim = meta.get("latent_dim", 16)
            best_alpha = meta.get("alpha", alpha)
            best_n_samples = meta.get("n_samples", n_samples)
            print(f"  Tuned hyperparams: hidden={h_dim} latent={l_dim} alpha={best_alpha} n_samples={best_n_samples}")
            cagpre = CAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim)
            cagpre.load_state_dict(checkpoint["state_dict"])
            cagpre.to(device)
            results["CAGPRE"], cagpre_low, cagpre_high, cagpre_scores = evaluate_cagpre(
                cagpre, preprocessor, test_df, best_n_samples, best_alpha, device,
                return_ranges=True, return_scores=True,
            )
        else:
            print("Training CAGPRE (CVAE) from scratch ...")
            cagpre = CAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=128, latent_dim=16)
            train_cagpre(cagpre, x_train, c_train, x_val, c_val, epochs=50, lr=5e-4, device=device)
            results["CAGPRE"], cagpre_low, cagpre_high, cagpre_scores = evaluate_cagpre(
                cagpre, preprocessor, test_df, n_samples, alpha, device,
                return_ranges=True, return_scores=True,
            )
        results["CAGPRE"]["time_sec"] = round(time.time() - t0, 2)
        print(f"  CAGPRE done ({results['CAGPRE']['time_sec']}s): {results['CAGPRE']}")
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print(f"  [RESUME] Skipping CAGPRE metrics (already done)")
        tuned_path = os.path.join(tuned_model_dir, "best.pt") if tuned_model_dir else ""
        if tuned_path and os.path.exists(tuned_path):
            print("  [RESUME] Re-generating CAGPRE ranges from tuned model ...")
            checkpoint = torch.load(tuned_path, map_location=device, weights_only=False)
            meta = checkpoint.get("meta", {})
            h_dim = meta.get("hidden_dim", 128)
            l_dim = meta.get("latent_dim", 16)
            best_alpha = meta.get("alpha", alpha)
            best_n_samples = meta.get("n_samples", n_samples)
            cagpre = CAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim)
            cagpre.load_state_dict(checkpoint["state_dict"])
            cagpre.to(device)
            c_test_ctx = preprocessor.transform_context(test_df)
            cagpre_low, cagpre_high, cagpre_scores = generate_ranges_and_scores(
                cagpre, c_test_ctx, preprocessor, test_df["value"].values,
                best_n_samples, best_alpha, device
            )

    # --- Global IQR ---
    iqr_low_arr, iqr_high_arr = None, None
    if "Global_IQR" not in results:
        print("Evaluating Global IQR ...")
        t0 = time.time()
        results["Global_IQR"], iqr_low_arr, iqr_high_arr = evaluate_global_iqr(
            train_df, test_df, return_ranges=True
        )
        results["Global_IQR"]["time_sec"] = round(time.time() - t0, 2)
        print(f"  Global IQR done ({results['Global_IQR']['time_sec']}s): {results['Global_IQR']}")
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print(f"  [RESUME] Skipping Global_IQR metrics (already done)")
        q1 = train_df["value"].quantile(0.25)
        q3 = train_df["value"].quantile(0.75)
        iqr = q3 - q1
        iqr_low_arr = np.full(len(test_df), q1 - 1.5 * iqr)
        iqr_high_arr = np.full(len(test_df), q3 + 1.5 * iqr)

    full_train = np.hstack([c_train, x_train.reshape(-1, 1)])
    full_val = np.hstack([c_val, x_val.reshape(-1, 1)])
    full_test = np.hstack([c_test, x_test.reshape(-1, 1)])

    # Subsample for sklearn if very large
    MAX_SKLEARN_ROWS = 500_000
    if full_train.shape[0] > MAX_SKLEARN_ROWS:
        print(f"  [NOTE] Subsampling for sklearn: {full_train.shape[0]:,} -> {MAX_SKLEARN_ROWS:,}")
        rng = np.random.default_rng(seed)
        sk_idx = rng.choice(full_train.shape[0], MAX_SKLEARN_ROWS, replace=False)
        sk_train = full_train[sk_idx]
    else:
        sk_train = full_train

    # --- Isolation Forest ---
    if "Isolation_Forest" not in results:
        print("Training Isolation Forest ...")
        t0 = time.time()
        results["Isolation_Forest"] = evaluate_isolation_forest(sk_train, full_test, test_df["is_error"].values, seed)
        results["Isolation_Forest"]["time_sec"] = round(time.time() - t0, 2)
        print(f"  Isolation Forest done ({results['Isolation_Forest']['time_sec']}s)")
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print(f"  [RESUME] Skipping Isolation_Forest")

    # --- LOF ---
    if "LOF" not in results:
        print("Training LOF ...")
        t0 = time.time()
        results["LOF"] = evaluate_lof(sk_train, full_test, test_df["is_error"].values)
        results["LOF"]["time_sec"] = round(time.time() - t0, 2)
        print(f"  LOF done ({results['LOF']['time_sec']}s)")
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print(f"  [RESUME] Skipping LOF")

    # --- Autoencoder ---
    if "Autoencoder" not in results:
        print("Training Standard Autoencoder ...")
        t0 = time.time()
        ae = Autoencoder(input_dim=full_train.shape[1], hidden_dim=128)
        results["Autoencoder"] = evaluate_autoencoder_model(
            ae, full_train, full_val, full_test, test_df["is_error"].values, 50, 5e-4, device
        )
        results["Autoencoder"]["time_sec"] = round(time.time() - t0, 2)
        print(f"  Autoencoder done ({results['Autoencoder']['time_sec']}s)")
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print(f"  [RESUME] Skipping Autoencoder")

    # --- VAE ---
    if "VAE" not in results:
        print("Training Vanilla VAE ...")
        t0 = time.time()
        vae = VanillaVAE(input_dim=full_train.shape[1], hidden_dim=128, latent_dim=16)
        results["VAE"] = evaluate_autoencoder_model(
            vae, full_train, full_val, full_test, test_df["is_error"].values, 50, 5e-4, device, is_vae=True
        )
        results["VAE"]["time_sec"] = round(time.time() - t0, 2)
        print(f"  VAE done ({results['VAE']['time_sec']}s)")
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print(f"  [RESUME] Skipping VAE")

    # --- Transformer AE ---
    if "Transformer_AE" not in results:
        print("Training Transformer Autoencoder ...")
        t0 = time.time()
        d_model = 32
        n_heads = 4
        tae = TransformerAutoencoder(input_dim=full_train.shape[1], d_model=d_model, n_heads=n_heads, n_layers=2)
        results["Transformer_AE"] = evaluate_autoencoder_model(
            tae, full_train, full_val, full_test, test_df["is_error"].values, 50, 5e-4, device
        )
        results["Transformer_AE"]["time_sec"] = round(time.time() - t0, 2)
        print(f"  Transformer AE done ({results['Transformer_AE']['time_sec']}s)")
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print(f"  [RESUME] Skipping Transformer_AE")

    # --- Save range comparison CSV ---
    if csv_path and (cagpre_low is not None or iqr_low_arr is not None):
        range_csv = csv_path.replace("compare_", "range_comparison_")
        _save_range_comparison(
            test_df, cagpre_low, cagpre_high, cagpre_scores,
            iqr_low_arr, iqr_high_arr, range_csv,
            model_ranges={},
        )

    return results


# ===========================================================================
# Utility
# ===========================================================================
def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results_csv(results: Dict[str, Dict[str, float]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "train", "evaluate", "compare", "preview", "scalability"], required=True)
    parser.add_argument("--model", choices=["cagpre", "ae", "vae", "transformer_ae"], default="cagpre")
    parser.add_argument("--dataset", choices=["simulated", "real"], required=True)
    parser.add_argument("--variable", default=None, help="SBP, DBP, PR, CH, Glucose")
    parser.add_argument("--gt-dir", default=".", help="Directory containing GT CSV files")
    parser.add_argument("--save-dir", default="artifacts")
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--row-limit", type=int, default=None)
    # Database connection
    parser.add_argument("--db-host", default=None)
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default=None)
    parser.add_argument("--db-user", default=None)
    parser.add_argument("--db-password", default=None)
    parser.add_argument("--schema", default="public")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-synthetic", type=int, default=20000)
    parser.add_argument("--error-rate", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.dataset == "real":
        if not args.variable:
            raise ValueError("Provide --variable for real data.")
        variable = args.variable
        extra_context: List[str] = []
        gt_index = load_ground_truth(variable, args.gt_dir)

        db_params = {
            "host": args.db_host or DB_CONFIG["host"],
            "port": args.db_port or DB_CONFIG["port"],
            "dbname": args.db_name or DB_CONFIG["dbname"],
            "user": args.db_user or DB_CONFIG["user"],
            "password": args.db_password or DB_CONFIG["password"],
        }
        schema = args.schema if args.schema != "public" else DB_CONFIG.get("schema", "public")

        df = get_data_from_omop(
            variable=variable, db_params=db_params, schema=schema,
            gt_index=gt_index, row_limit=args.row_limit,
        )
    else:
        if not args.variable:
            raise ValueError("Provide --variable for simulated data.")
        variable = args.variable
        gt_index = load_ground_truth(variable, args.gt_dir)
        df = generate_synthetic_data(gt_index, args.n_synthetic, args.error_rate, variable, args.seed)
        extra_context = []

    print(f"Dataset: {args.dataset} | Variable: {variable} | Records: {len(df)} | Errors: {df['is_error'].sum()}")

    # --- Preview mode ---
    if args.mode == "preview":
        csv_dir = os.path.join(args.save_dir, variable.upper())
        os.makedirs(csv_dir, exist_ok=True)
        preview_path = os.path.join(csv_dir, f"preview_{args.dataset}_{variable.upper()}.csv")
        normal = df[df["is_error"] == 0].head(150)
        errors = df[df["is_error"] == 1].head(50)
        sample = pd.concat([normal, errors]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        sample.to_csv(preview_path, index=False)
        print(f"\nPreview saved to: {preview_path}")
        return

    # --- Scalability mode ---
    if args.mode == "scalability":
        sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        all_rows = []
        for n in sizes:
            print(f"\n{'='*70}\nSCALABILITY: {variable.upper()} | n={n:,}\n{'='*70}")
            set_seed(args.seed)
            if args.dataset == "simulated":
                df_n = generate_synthetic_data(gt_index, n, args.error_rate, variable, args.seed)
            else:
                if n > len(df):
                    print(f"Skipping n={n} as it exceeds dataset size ({len(df)})")
                    continue
                df_n = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
                
            train_n, val_n, test_n, prep_n = prepare_datasets(df_n, extra_context, args.seed)
            results_n = run_compare(train_n, val_n, test_n, prep_n, args.alpha, args.n_samples, device, args.seed)
            for model_name, metrics in results_n.items():
                all_rows.append({
                    "n_records": n, "model": model_name,
                    "time_sec": metrics.get("time_sec", 0.0),
                    "precision": round(metrics.get("precision", 0.0), 4),
                    "recall": round(metrics.get("recall", 0.0), 4),
                    "f1": round(metrics.get("f1", 0.0), 4),
                    "iou": round(metrics.get("iou", 0.0), 4) if metrics.get("iou") else "",
                })
        csv_dir = os.path.join(args.save_dir, variable.upper())
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"scalability_{args.dataset}.csv")
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"\nScalability saved to: {csv_path}")
        return

    train_df, val_df, test_df, preprocessor = prepare_datasets(df, extra_context, args.seed)
    save_root = os.path.join(args.save_dir, variable.upper(), args.model)
    csv_dir = os.path.join(args.save_dir, variable.upper())
    os.makedirs(csv_dir, exist_ok=True)

    if args.mode == "compare":
        csv_path = os.path.join(csv_dir, f"compare_{args.dataset}.csv")
        tuned_dir = os.path.join(args.save_dir, variable.upper(), "cagpre")

        if args.dataset == "real":
            # Real-world: also run GT-row-based evaluation
            gt_csv = os.path.join(csv_dir, f"gt_eval_{args.dataset}.csv")

            # First train/load the CAGPRE model
            tuned_path = os.path.join(tuned_dir, "best.pt") if os.path.exists(os.path.join(tuned_dir, "best.pt")) else ""
            x_train = preprocessor.transform_target(train_df["value"].values)
            c_train = preprocessor.transform_context(train_df)
            x_val = preprocessor.transform_target(val_df["value"].values)
            c_val = preprocessor.transform_context(val_df)

            if tuned_path:
                print(f"Loading tuned CAGPRE from {tuned_path} ...")
                checkpoint = torch.load(tuned_path, map_location=device, weights_only=False)
                meta = checkpoint.get("meta", {})
                h_dim = meta.get("hidden_dim", 128)
                l_dim = meta.get("latent_dim", 16)
                best_alpha = meta.get("alpha", args.alpha)
                best_n_samples = meta.get("n_samples", args.n_samples)
                cagpre_model = CAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim)
                cagpre_model.load_state_dict(checkpoint["state_dict"])
                cagpre_model.to(device)
            else:
                print("Training CAGPRE from scratch for real data ...")
                best_alpha = args.alpha
                best_n_samples = args.n_samples
                cagpre_model = CAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=128, latent_dim=16)
                train_cagpre(cagpre_model, x_train, c_train, x_val, c_val, epochs=50, lr=5e-4, device=device)

            # GT-row evaluation
            gt_results = run_gt_evaluation(
                cagpre_model, preprocessor, gt_index, test_df,
                best_n_samples, best_alpha, device, csv_path=gt_csv
            )
            print(f"\n  GT Evaluation IoU: {gt_results['iou']:.4f}")

        # Standard comparison (works for both simulated and real)
        results = run_compare(
            train_df, val_df, test_df, preprocessor, args.alpha, args.n_samples, device, args.seed,
            csv_path=csv_path, tuned_model_dir=tuned_dir,
        )
        save_results_csv(results, csv_path)
        return

    if args.model != "cagpre":
        raise ValueError("tune/train/evaluate modes are for CAGPRE only.")

    x_train = preprocessor.transform_target(train_df["value"].values)
    c_train = preprocessor.transform_context(train_df)
    x_val = preprocessor.transform_target(val_df["value"].values)
    c_val = preprocessor.transform_context(val_df)

    model = CAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=64, latent_dim=8)

    if args.mode == "tune":
        best_path, metrics = tune_cagpre(
            train_df, val_df, preprocessor, args.n_trials, args.alpha, args.n_samples, device, save_root
        )
        results = {"CAGPRE_best": metrics}
        csv_path = os.path.join(csv_dir, f"tune_{args.dataset}.csv")
        save_results_csv(results, csv_path)
        print(f"Best model saved to: {best_path}")
        return

    if args.mode == "train":
        train_cagpre(model, x_train, c_train, x_val, c_val, epochs=50, lr=5e-4, device=device)
        save_path = os.path.join(save_root, "best.pt")
        save_model(model, save_path, {"alpha": args.alpha, "n_samples": args.n_samples})
        metrics = evaluate_cagpre(model, preprocessor, test_df, args.n_samples, args.alpha, device)
        results = {"CAGPRE": metrics}
        csv_path = os.path.join(csv_dir, f"train_{args.dataset}.csv")
        save_results_csv(results, csv_path)
        return

    if args.mode == "evaluate":
        if not args.load_path:
            raise ValueError("Provide --load-path for evaluate mode.")
        meta = load_model(model, args.load_path, device)
        print(f"Loaded model meta: {meta}")
        metrics = evaluate_cagpre(model, preprocessor, test_df, args.n_samples, args.alpha, device)
        results = {"CAGPRE_loaded": metrics}
        csv_path = os.path.join(csv_dir, f"evaluate_{args.dataset}.csv")
        save_results_csv(results, csv_path)



from ..baselines.neural import (Autoencoder, VanillaVAE, TransformerAutoencoder, train_autoencoder, reconstruction_errors, evaluate_autoencoder_model)
from ..baselines.statistical import (evaluate_global_iqr, evaluate_isolation_forest, evaluate_lof)

if __name__ == "__main__":
    main()
