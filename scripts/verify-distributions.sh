#!/bin/sh
set -eu

dist_dir="${1:-dist}"

count=0

for wheel in "${dist_dir}"/*.whl; do
  [ -f "${wheel}" ] || continue
  unzip -t "${wheel}"
  count=$((count + 1))
done

for sdist in "${dist_dir}"/*.tar.gz; do
  [ -f "${sdist}" ] || continue
  tar -tzf "${sdist}" >/dev/null
  count=$((count + 1))
done

if [ "${count}" -eq 0 ]; then
  echo "No distributions found in ${dist_dir}" >&2
  exit 1
fi
