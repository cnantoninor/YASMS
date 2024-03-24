#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:src"
pytest --cov=src --cov-report html
/mnt/c/Program\ Files\ \(x86\)/Microsoft/Edge/Application/msedge.exe file:////wsl.localhost\\Ubuntu\\home\\arau6\\projects\\wl-semsearch-poc\\htmlcov\\index.html