import os
import sys
import json
import traceback

NB_PATH_SRC = r'd:\working_projects\SIH\cyberCast\nootebooks\CyberCast_Complete.ipynb'
NB_PATH_DEST1 = r'd:\working_projects\SIH\cyberCast\notebooks\CyberCast_Complete.ipynb'
NB_PATH_DEST2 = r'd:\working_projects\SIH\cyberCast\nootebooks\CyberCast_Complete.ipynb'

print("================================================================================")
print("NOTEBOOK EXECUTION & ERROR AUDIT")
print("================================================================================")

with open(NB_PATH_SRC, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded source notebook: {NB_PATH_SRC} ({len(nb['cells'])} cells)")

# Global environment for notebook execution
exec_env = {
    '__name__': '__main__',
    '__file__': NB_PATH_SRC,
}

cells = nb['cells']
passed_count = 0
error_count = 0

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source_code = "".join(cell['source'])
        if not source_code.strip():
            continue
        print(f"\n--- Executing Code Cell {i:02d} ---")
        preview = source_code.strip().split('\n')[0][:80]
        print(f"  Snippet: {preview}")
        try:
            # Execute code cell in shared exec_env
            exec(source_code, exec_env)
            passed_count += 1
            print(f"  Result: CELL {i:02d} PASSED")
        except Exception as e:
            error_count += 1
            print(f"  Result: CELL {i:02d} ERROR: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)

print("\n================================================================================")
print(f"EXECUTION SUMMARY: Passed={passed_count}, Errors={error_count}")
print("================================================================================")
