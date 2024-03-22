#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:src"
uvicorn app:app --reload > uvicorn.err.log 2> uvicorn.log &
