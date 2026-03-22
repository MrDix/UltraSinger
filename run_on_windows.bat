@echo off
:: Activate the virtual environment and open cmd in the src directory
:: Suppress compile-time SyntaxWarnings from third-party packages (e.g. pydub)
set PYTHONWARNINGS=ignore::SyntaxWarning
call .venv\Scripts\activate.bat
cd src
cmd /k
