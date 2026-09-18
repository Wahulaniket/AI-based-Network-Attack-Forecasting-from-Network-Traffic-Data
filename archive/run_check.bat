@echo off
py d:\working_projects\SIH\cyberCast\scratch\check_env.py
if errorlevel 1 (
    python d:\working_projects\SIH\cyberCast\scratch\check_env.py
)
where python
where py
where uvicorn
