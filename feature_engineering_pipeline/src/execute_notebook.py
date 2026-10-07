import json
import os
import sys
import io
from contextlib import redirect_stdout

# Include root in module path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.transformers import (
    NumericalImputer, CategoricalImputer, StandardScalerTransformer,
    OneHotEncoderTransformer, TargetEncoderTransformer, Pipeline, ColumnTransformer
)
from src.pipeline_builder import build_preprocessing_pipeline
from src.correlation_analysis import analyze_numerical_correlations
from src.leakage_checks import run_leakage_and_integrity_checks

def execute_notebook_cells():
    nb_path = os.path.join(project_root, 'notebooks', 'feature_engineering_pipeline.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    global_env = {'__name__': '__main__'}
    exec_count = 1

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code_str = "".join(cell['source'])
            print(f"Executing Feature Engineering Code Cell {exec_count}...")
            
            f_out = io.StringIO()
            with redirect_stdout(f_out):
                exec(code_str, global_env)
            output_text = f_out.getvalue()

            cell['execution_count'] = exec_count
            cell['outputs'] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": output_text.splitlines(keepends=True)
                }
            ]
            exec_count += 1

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("Feature Engineering Notebook executed successfully and outputs populated!")

if __name__ == '__main__':
    execute_notebook_cells()
