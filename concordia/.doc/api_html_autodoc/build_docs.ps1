# PowerShell script to build Sphinx autodoc for Concordia
# Usage: Run from the .docs directory in your venv

# Activate the venv if not already active
$venvPath = "../../.pyscrai/Scripts/Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
}

# Build the HTML docs
sphinx-build -b html . ../_build/html

Write-Host "Sphinx autodoc build complete. Output in _build/html."
