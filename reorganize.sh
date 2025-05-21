#!/usr/bin/env bash
set -euo pipefail

dst_root="results"
mkdir -p "$dst_root/U1/T10000"

for art in evo norms_s renyi; do
  find "merged/$art/U1" -type f -name 'data.npz' | while read -r src; do
    # extract theta and circuit index from the path
    # e.g. merged/evo/U1/theta0.30/T10000/circuit_real7/data.npz
    theta=$(echo "$src" | sed -E 's|.*/U1/(theta[0-9]+\.[0-9]+)/.*|\1|')
    circuit=$(echo "$src" | sed -E 's|.*/circuit_real([0-9]+)/.*|\1|')

    dest="$dst_root/U1/T10000/$theta/circuit_real$circuit/$art"
    mkdir -p "$dest"
    cp "$src" "$dest/"
  done
done
