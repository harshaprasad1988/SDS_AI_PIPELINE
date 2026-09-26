#!/bin/bash
set -e

ENV_NAME="mds_checker"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo " AI MDS Checker - Step 1 + Step 2 Installer"
echo " Python 3.11 + Streamlit"
echo "=============================================="

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: Conda was not found."
  echo "Install Miniconda/Anaconda first, then run this script again."
  exit 1
fi

# Make conda available in this non-interactive shell.
eval "$(conda shell.bash hook)"

# Create/reuse a clean Python 3.11 environment.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Using existing conda environment: $ENV_NAME"
else
  echo "Creating conda environment: $ENV_NAME (Python 3.11)..."
  conda create -n "$ENV_NAME" python=3.11 pip -y
fi

conda activate "$ENV_NAME"

# System dependencies required by PDF/image OCR on macOS.
if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
  if ! command -v tesseract >/dev/null 2>&1; then
    echo "Installing Tesseract via Homebrew..."
    brew install tesseract
  fi
  if ! command -v pdftoppm >/dev/null 2>&1; then
    echo "Installing Poppler via Homebrew..."
    brew install poppler
  fi
else
  echo "NOTE: This script assumes Tesseract and Poppler are already installed on non-macOS systems."
fi

echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing Step 1 + Step 2 Python dependencies..."
python -m pip install -r requirements.txt

echo "Installing spaCy English model..."
python -m spacy download en_core_web_sm

echo ""
echo "Installation complete."
echo "Starting Streamlit..."
echo ""
streamlit run streamlit_app.py
