#!/bin/bash

# remove all the uvicorn and app.log logs
rm -f uvicorn.*
rm -f app.log
rm -rf data/spam_classifier/GradientBoostingClassifier/test_nino
