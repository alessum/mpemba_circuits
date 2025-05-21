#!/usr/bin/env bash
# run_thetas.sh
# Loop over theta_to_run values and trigger the GitHub Actions workflow

# Define array of theta values
thetas=(0.35 0.40 0.45 0.50) #  0.35 0.40 0.45 0.50
circuit_to_runs=(10 13 16 19 22 25 28 31 34 37) #  11 12 13 14 15 16 17 18 19

# Fixed parameters
ref="main"
number_of_circuits=3
time_to_run=10000

for circuit_to_run in "${circuit_to_runs[@]}"; do
    for theta in "${thetas[@]}"; do
    echo "Running workflow for theta_to_run=$theta..."
    gh workflow run run-runner.yml \
        --ref "$ref" \
        -f circuit_to_run="$circuit_to_run" \
        -f number_of_circuits="$number_of_circuits" \
        -f theta_to_run="$theta" \
        -f time_to_run="$time_to_run"
    done
done

