import json
import os
import sys

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import load_and_audit_data
from src.baseline_model import MajorityBaseline, HeuristicBaseline, NumpyLogisticRegression
from src.evaluation import calculate_metrics, print_metrics_summary

def execute_notebook_code():
    nb_path = os.path.join(project_root, 'notebooks', 'reproducible_baseline.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    global_env = {'__name__': '__main__'}
    exec_count = 1
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code_str = "".join(cell['source'])
            print(f"Executing Code Cell {exec_count}...")
            
            # Capture stdout
            import io
            from contextlib import redirect_stdout
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
            
    # Save back executed notebook
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("Notebook executed successfully and outputs populated!")

if __name__ == '__main__':
    execute_notebook_code()
