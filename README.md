# CAGPRE

**Context-Aware Generative Framework for Plausible Range Estimation in Medical Data**

CAGPRE estimates **context-specific plausible ranges** for numerical clinical measurements. It uses a residual conditional variational autoencoder (CVAE) to model a measurement conditional on patient context, then applies **Generative Quantile Estimation (GQE)** to derive plausible intervals. These intervals can help identify measurements that are implausible in the context of an individual patient, rather than relying only on fixed, population-wide thresholds.

This repository provides the CAGPRE implementation, code for synthetic data generation, ground-truth range resources, and scripts for evaluating CAGPRE and comparison methods.

## Clinical variables

The implementation covers five target measurements:

| CLI variable | Measurement |
| --- | --- |
| `SBP` | Systolic blood pressure |
| `DBP` | Diastolic blood pressure |
| `PR` | Pulse rate |
| `CH` | Cholesterol (as labeled in the implementation) |
| `GLUCOSE` | Glucose |

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/yeasmin62/CAGPRE.git
cd CAGPRE
```

> **Repository layout:** The original setup instructions use `cd CAGPRE/CAGPRE`. If your clone contains a second `CAGPRE/` directory holding `requirements.txt` and `scripts/`, enter that inner directory before running the commands below.

### 2. Create a Python environment

```bash
python -m venv venv
```

Activate it on **Linux/macOS**:

```bash
source venv/bin/activate
```

Or in **Windows Git Bash**:

```bash
source venv/Scripts/activate
```

Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

> The exact Python version and dependency versions should be checked against the repository's `requirements.txt` and runtime configuration.

## Configuration

Synthetic experiments and real-world experiments have different data-access requirements.

### Synthetic experiments

The synthetic workflow uses the generator and ground-truth range resources included in this repository. The paths documented in the original setup are:

```ini
CAGPRE_GT_DIR=data/ground_truth
CAGPRE_SYNTHETIC_DIR=synthetic_data
```

### Real-world experiments (restricted data)

Real-world experiments require **separately authorized access** to the clinical data in an OMOP CDM PostgreSQL database. The Danish National Health Registry data used in the study are **not distributed** through this repository because of data-protection and access restrictions.

If `.env.example` is present, copy it to `.env`:

```bash
cp .env.example .env
```

Configure the database and local data paths in `.env`:

```ini
OMOP_DB_HOST=localhost
OMOP_DB_PORT=5432
OMOP_DB_NAME=your_omop_database
OMOP_DB_USER=your_username
OMOP_DB_PASSWORD=your_password

CAGPRE_GT_DIR=data/ground_truth
CAGPRE_SYNTHETIC_DIR=synthetic_data
```

**Security:** Never commit `.env`, passwords, connection strings, or patient-level records to GitHub. Make sure these files and directories are excluded by `.gitignore`.

## Running the code

Run the following commands from the directory containing `scripts/`.

### 1. Generate synthetic datasets

```bash
python scripts/generate_synthetic.py
```

The documented generator produces `synthetic_<VARIABLE>.csv` files under `synthetic_data/` for `SBP`, `DBP`, `PR`, `CH`, and `GLUCOSE`.

### 2. Run synthetic-data experiments

**Train CAGPRE:**

```bash
python scripts/run_synthetic.py --variable SBP --mode train --epochs 80
```

**Tune hyperparameters with Optuna:**

```bash
python scripts/run_synthetic.py --variable SBP --mode tune --n-trials 10
```

**Compare CAGPRE with baseline methods:**

```bash
python scripts/run_synthetic.py --variable SBP --mode compare
```

The original instructions list Global IQR, Isolation Forest, Local Outlier Factor (LOF), Autoencoder, Vanilla VAE, and Transformer Autoencoder among the comparison methods.

To use a different measurement, replace `SBP` with `DBP`, `PR`, `CH`, or `GLUCOSE`.

### 3. Run real-world experiments

These commands require authorized data access and the database configuration described above.

**Tune hyperparameters:**

```bash
python scripts/run_real_world.py --dataset real --variable SBP --mode tune --n-trials 10
```

**Compare CAGPRE and baseline methods:**

```bash
python scripts/run_real_world.py --dataset real --variable SBP --mode compare
```

The original setup also describes database-related command-line flags. Prefer environment-based credentials rather than typing passwords into commands, where shell history may retain them.

## Command-line options

The following options and defaults are taken from the supplied setup notes; consult each script's `--help` output for the definitive list supported by your checked-out version.

| Option | Purpose | Documented default |
| --- | --- | --- |
| `--variable` | Target clinical measurement | `SBP` / may be required |
| `--mode` | `train`, `tune`, `compare` | `compare` |
| `--dataset` | `simulated` or `real` | `simulated` |
| `--epochs` | Training epochs | `80` |
| `--batch-size` | Batch size | `1024` |
| `--n-samples` | Samples used for GQE | `1000` |
| `--alpha` | Quantile tail probability | `0.005` |
| `--n-trials` | Optuna trials | `10` |
| `--save-dir` | Directory for outputs/checkpoints | `artifacts_simulated` / `artifacts_residual` |

## Tests

If the repository includes the `tests/` directory and `pytest` is installed, run:

```bash
python -m pytest tests/
```

## Data availability and reproducibility

- **Clinical data:** The Danish National Health Registry data used in the study cannot be made public due to data-protection and access restrictions. Access requires authorization from the responsible data provider.
- **Synthetic data:** This repository includes code to generate synthetic data for experiments. Ground-truth range resources and the documented synthetic workflow are described above.
- **Source code:** The implementation and experiment entry points are provided to support methodological inspection and reproducibility within the constraints of the available data.

## Citation

If you use CAGPRE in your research, please cite the associated manuscript when its official bibliographic details become available:

**CAGPRE: A Context-Aware Generative Framework for Plausible Range Estimation in Medical Data.** Manuscript prepared for submission to *Artificial Intelligence in Medicine*.

> A finalized BibTeX entry will be added after publication. The journal name above indicates the intended submission venue, **not** an accepted or published article.

---

**Repository:** https://github.com/yeasmin62/CAGPRE
