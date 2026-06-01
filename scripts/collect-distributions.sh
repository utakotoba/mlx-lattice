#!/bin/sh
set -eu

expected_count="${1:?usage: scripts/collect-distributions.sh <expected-count> [source-dir] [output-dir]}"
source_dir="${2:-artifacts}"
out_dir="${3:-dist}"

rm -rf "${out_dir}"
mkdir -p "${out_dir}"

manifest="${out_dir}/.distribution-files"
find "${source_dir}" -type f \( -name '*.whl' -o -name '*.tar.gz' \) -print > "${manifest}"

count=0

while IFS= read -r file; do
  filename="$(basename "${file}")"
  if [ -e "${out_dir}/${filename}" ]; then
    echo "::error::Duplicate distribution filename: ${filename}"
    echo "::error::Second: ${file}"
    exit 1
  fi

  cp "${file}" "${out_dir}/${filename}"
  count=$((count + 1))
done < "${manifest}"

rm -f "${manifest}"

if [ "${count}" -ne "${expected_count}" ]; then
  echo "::error::Expected ${expected_count} distributions, found ${count}"
  find "${source_dir}" -type f -print
  exit 1
fi
