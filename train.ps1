# Rebuild the complete experiment using the project's isolated environment.
& "$PSScriptRoot\.venv\Scripts\python.exe" -u "$PSScriptRoot\src\train.py" @args
exit $LASTEXITCODE
