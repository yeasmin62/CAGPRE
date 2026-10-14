"""Guard the paper comparison against unreported exploratory baselines."""
import ast
from pathlib import Path

EXCLUDED = ("Context_Group_IQR", "Cond_Autoencoder", "Cond_VAE", "ConditionalAutoencoder", "ConditionalVAE")
ROOT = Path(__file__).resolve().parents[1]

def test_experiment_comparisons_exclude_exploratory_methods():
    for name in ("real_world.py", "synthetic.py"):
        path = ROOT / "src" / "cagpre" / "experiments" / name
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        for method in EXCLUDED:
            assert method not in source, f"{method} remains in {name}"

def test_resume_filters_exploratory_methods():
    source = (ROOT / "src" / "cagpre" / "engines" / "shared.py").read_text(encoding="utf-8")
    for method in EXCLUDED[:3]:
        assert f'"{method}"' in source
