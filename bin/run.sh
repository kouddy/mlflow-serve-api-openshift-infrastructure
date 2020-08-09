#!/bin/sh

set -e

export FLASK_APP=/app/mlflow/app.py

python -m flask run --host=0.0.0.0