from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
def test_python_source_parses():
    for file in (ROOT/'src').rglob('*.py'):
        ast.parse(file.read_text(), filename=str(file))
def test_ground_truth_files_present():
    assert len(list((ROOT/'data'/'ground_truth').glob('*GT.csv')))==5
