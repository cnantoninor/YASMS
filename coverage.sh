#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:src"
pytest --cov=wl-ml-server --cov-report html