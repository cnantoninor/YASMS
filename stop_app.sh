#!/bin/bash

# Find the PID of the uvicorn process
pid=$(ps -ef | grep uvicorn | grep -v grep | awk '{print $2}')

# If a PID was found, kill the process
if [ -n "$pid" ]; then
    kill $pid
    echo "Uvicorn process $pid has been stopped."
else
    echo "No running Uvicorn process was found."
fi