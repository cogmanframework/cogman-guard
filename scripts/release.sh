#!/usr/bin/env bash
set -euo pipefail

# Build and upload to PyPI (requires twine)
python -m build
python -m twine upload dist/*
