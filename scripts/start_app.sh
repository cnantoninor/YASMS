#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:src"
uvicorn app:app --host 0.0.0.0 --port 8000 > uvicorn.log 2> uvicorn.err.log &
sleep 2
tac ~/wl-semsearch-poc/uvicorn.log
tac ~/wl-semsearch-poc/app.log
