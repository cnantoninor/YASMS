#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:src"
uvicorn app:app --host 0.0.0.0 --port 8000 > uvicorn.log 2> uvicorn.err.log &
sleep 2
tac ~/projects/wl-semsearch-poc/uvicorn.log | head -n 20
tac ~/projects/wl-semsearch-poc/uvicorn.err.log | head -n 20
tac ~/projects/wl-semsearch-poc/app.log | head -n 20
