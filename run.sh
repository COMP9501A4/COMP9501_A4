#!/bin/bash

echo "base"
python3 src/run.py
echo "regression"
python3 src/run.py --task regression

echo "RNN"
python3 src/run.py --model RNN

echo "channel"
python3 src/run.py --channel 1
python3 src/run.py --channel 5

echo "max_length"
python3 src/run.py --max_length 30
python3 src/run.py --max_length 70  

echo "dropout"
python3 src/run.py --dropout 0
python3 src/run.py --dropout 0.2

echo "embedding_method"
python3 src/run.py --embedding_method random
