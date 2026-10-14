"""CLI entry point for the CAGPRE research implementation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from runpy import run_module
if __name__ == "__main__":
    run_module("cagpre.experiments.generate_synthetic",run_name="__main__")
