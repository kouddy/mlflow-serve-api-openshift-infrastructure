#!/bin/sh

set -e

mlflow models serve --no-conda \
    -h 0.0.0.0 \
    -p 5000 \
    -m ${MODEL_ARTIFACT_URI}