#!/usr/bin/env bash
# Creates a virtual environment on first use, installs the pinned packages and runs the pipeline.
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
.venv/bin/python run_pipeline.py "$@"
