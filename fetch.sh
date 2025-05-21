#!/usr/bin/env bash
set -euo pipefail

# Compute cutoff timestamp: 10 hours ago in UTC (BSD/macOS syntax)
SINCE=$(date -u -v-10H +"%Y-%m-%dT%H:%M:%SZ")

# Fetch up to 100 successful runs since that time, emitting one JSON object per line
mapfile -t runs < <(
  gh run list \
    --limit 25 \
    --status success \
    --json databaseId,workflowName,createdAt \
    --jq ".[] | select(.createdAt >= \"$SINCE\") | {id: .databaseId, name: .workflowName, createdAt: .createdAt}"
)
echo "Found ${#runs[@]} successful runs since $SINCE"

# Iterate over each run (each element of the array is a JSON object string)
for run in "${runs[@]}"; do
  run_id=$(jq -r '.id'        <<<"$run")
  name=$(jq -r '.name'         <<<"$run")
  created=$(jq -r '.createdAt' <<<"$run")

  # Sanitize the workflow name for filesystem use
  safe_name=${name// /_}
  dest_dir="artifacts/${safe_name}_${run_id}"

  echo "↓ Downloading artifacts for run $run_id ($name @ $created) into $dest_dir"
  mkdir -p "$dest_dir"
  gh run download "$run_id" --dir "$dest_dir"
  echo "✔ Done with run $run_id"
done

echo "All artifacts downloaded."


# mkdir -p merged/{evo,norms_s,renyi}

# for sub in evo norms_s renyi; do
#   for dir in results_/*/$sub; do
#     # copy everything from each run’s sub‑folder into merged/$sub
#     cp -r "$dir"/* merged/$sub/
#   done
# done
