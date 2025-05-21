#!/usr/bin/env bash
# run_thetas.sh
# Loop over theta_to_run values and trigger the GitHub Actions workflow

# Define array of theta values
thetas=(0.30 0.35 0.40 0.45 0.50)
circuit_to_runs=(0 5 10 15 20)

# Fixed parameters
ref="main"
number_of_circuits=5
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