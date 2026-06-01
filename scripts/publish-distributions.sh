#!/bin/sh
set -eu

target="${1:?usage: scripts/publish-distributions.sh <main|cuda13> [dist-dir]}"
dist_dir="${2:-dist}"
check_url="${MLX_LATTICE_PUBLISH_CHECK_URL:-https://pypi.org/simple/}"

set --

case "${target}" in
  main)
    for file in "${dist_dir}"/mlx_lattice-[0-9]*.whl "${dist_dir}"/mlx_lattice-*.tar.gz; do
      [ -f "${file}" ] || continue
      set -- "$@" "${file}"
    done
    ;;
  cuda13)
    for file in "${dist_dir}"/mlx_lattice_cuda13-*.whl; do
      [ -f "${file}" ] || continue
      set -- "$@" "${file}"
    done
    ;;
  *)
    echo "Unknown publish target: ${target}" >&2
    exit 1
    ;;
esac
if [ "$#" -eq 0 ]; then
  echo "No distributions found for publish target: ${target}" >&2
  find "${dist_dir}" -type f -print >&2
  exit 1
fi

uv publish \
  --trusted-publishing always \
  --check-url "${check_url}" \
  "$@"
