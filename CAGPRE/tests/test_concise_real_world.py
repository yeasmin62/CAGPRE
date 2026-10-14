from pathlib import Path
import ast

SOURCE = Path(__file__).resolve().parents[1] / "src/cagpre/experiments/real_world.py"

def test_real_world_tuning_quiet_and_trial_reporting():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    tuning = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "tune_residual")
    segment = ast.get_source_segment(source, tuning)
    assert "redirect_stdout(quiet_output)" in segment
    assert "redirect_stderr(quiet_output)" in segment
    assert "*** NEW BEST MODEL FOUND ***" in segment
    assert "_SuppressStudyCreation" in segment
    assert "study.optimize(objective, n_trials=n_trials)" in segment
    assert "f\"score={best_trial_score:.4f}" in segment
