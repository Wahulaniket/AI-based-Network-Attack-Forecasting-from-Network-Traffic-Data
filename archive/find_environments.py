import sys
import os
import glob
import subprocess

print("Current Python:", sys.executable)

# Look for python.exe in standard locations
search_paths = [
    r"C:\Users\hp\AppData\Local\Programs\Python\*",
    r"C:\Users\hp\anaconda3\*",
    r"C:\Users\hp\miniconda3\*",
    r"C:\ProgramData\anaconda3\*",
    r"C:\Users\hp\.virtualenvs\*",
    r"d:\working_projects\SIH\cyberCast\*",
    r"C:\Python*",
]

found_pythons = []
for p in search_paths:
    for match in glob.glob(p):
        py_exe = os.path.join(match, "python.exe")
        if os.path.exists(py_exe):
            found_pythons.append(py_exe)
        py_scripts = os.path.join(match, "Scripts", "python.exe")
        if os.path.exists(py_scripts):
            found_pythons.append(py_scripts)

print("Found Pythons:", found_pythons)

for py in found_pythons:
    try:
        res = subprocess.run([py, "-c", "import scapy; import torch; import fastapi; print('FOUND ALL IN:', sys.executable)"], capture_output=True, text=True, timeout=5)
        print(py, "->", res.stdout.strip() or res.stderr.strip())
    except Exception as e:
        print(py, "-> Error:", e)
