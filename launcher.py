#!/usr/bin/env python3
import subprocess

def run_workflows(
    thetas = [0.30, 0.35, 0.40, 0.45, 0.50],
    circuits = [10, 13, 16, 19, 22, 25, 28, 31, 34, 37],
    ref="main",
    number_of_circuits=3,
    time_to_run=10000,
    workflow_file="run-runner.yml"
):
    for circuit in circuits:
        for theta in thetas:
            print(f"Running workflow for circuit_to_run={circuit}, theta_to_run={theta}…")
            subprocess.run([
                "gh", "workflow", "run", workflow_file,
                "--ref", str(ref),
                "-f", f"circuit_to_run={circuit}",
                "-f", f"number_of_circuits={number_of_circuits}",
                "-f", f"theta_to_run={theta}",
                "-f", f"time_to_run={time_to_run}"
            ], check=True)

if __name__ == "__main__":
    run_workflows()
