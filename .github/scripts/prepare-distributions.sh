#!/usr/bin/env bash
set -euo pipefail

artifact_dir="${1:?artifact directory is required}"
dist_dir="${2:?distribution directory is required}"
expected_distributions="${3:?expected distribution count is required}"

if [[ ! -d "$artifact_dir" ]]; then
  echo "::error::Artifact directory does not exist: $artifact_dir"
  exit 1
fi

rm -rf "$dist_dir"
mkdir -p "$dist_dir"

while IFS= read -r -d '' file; do
  filename="$(basename "$file")"
  destination="$dist_dir/$filename"

  if [[ -e "$destination" ]]; then
    echo "::error::Duplicate distribution filename: $filename"
    echo "::error::Duplicate source: $file"
    exit 1
  fi

  cp "$file" "$destination"
done < <(find "$artifact_dir" -type f \( -name "*.whl" -o -name "*.tar.gz" \) -print0)

actual_distributions="$(find "$dist_dir" -type f \( -name "*.whl" -o -name "*.tar.gz" \) -print | wc -l | tr -d "[:space:]")"
if [[ "$actual_distributions" -ne "$expected_distributions" ]]; then
  echo "::error::Expected $expected_distributions distributions, found $actual_distributions"
  find "$artifact_dir" -type f -print
  exit 1
fi

ls -lh "$dist_dir"
