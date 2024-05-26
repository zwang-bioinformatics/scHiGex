#!/bin/bash

model_name="GTransCustomModular"
cuda="0"

# Create a new directory with the current timestamp
mkdir figures
mkdir results
mkdir misc

cd src

echo $model_name > model

# Run the Python script
nohup python3 -u main.py -gpu $cuda > ../logs/training_output.log 2>&1 &
