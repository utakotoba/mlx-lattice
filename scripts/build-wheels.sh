#!/bin/sh
set -eu

target="${1:?usage: scripts/build-wheels.sh <macos|linux-cpu|linux-cuda13>}"
pythons="${MLX_LATTICE_PYTHONS:-3.12 3.13 3.14}"
out_dir="${MLX_LATTICE_WHEEL_DIR:-dist}"
manylinux_tag="${MLX_LATTICE_MANYLINUX_TAG:-manylinux_2_35_x86_64}"
cuda_architectures="${MLX_LATTICE_CUDA_ARCHITECTURES:-80}"

mkdir -p "${out_dir}"
uv python install ${pythons}

retag_linux_wheel() {
  package="${1}"
  python_tag="${2}"
  python_version="${3}"

  wheel_file="$(
    find "${out_dir}" \
      -type f \
      -name "${package}-*-${python_tag}-${python_tag}-linux_x86_64.whl" \
      -print \
      | sed -n '1p'
  )"
  if [ -z "${wheel_file}" ]; then
    echo "Could not find linux_x86_64 wheel for ${package} ${python_tag} in ${out_dir}" >&2
    find "${out_dir}" -type f -name '*.whl' -print >&2
    exit 1
  fi

  UV_PYTHON="${python_version}" uv run --no-sync wheel tags \
    --platform-tag "${manylinux_tag}" \
    --remove \
    "${wheel_file}"
}

for python in ${pythons}; do
  echo "::group::Build ${target} wheel for Python ${python}"
  python_tag="cp$(printf '%s' "${python}" | tr -d '.')"

  case "${target}" in
    macos)
      uv build \
        --wheel \
        --python "${python}" \
        --out-dir "${out_dir}"
      ;;
    linux-cpu)
      rm -rf .venv
      UV_PYTHON="${python}" uv sync \
        --group dev \
        --no-install-project
      UV_PYTHON="${python}" uv build \
        --wheel \
        --python "${python}" \
        --no-build-isolation \
        --out-dir "${out_dir}" \
        -Ccmake.define.MLX_LATTICE_BUILD_CUDA=OFF \
        -Ccmake.define.MLX_LATTICE_BUILD_METAL=OFF
      retag_linux_wheel "mlx_lattice" "${python_tag}" "${python}"
      ;;
    linux-cuda13)
      rm -rf .venv
      UV_PYTHON="${python}" uv sync \
        --group dev \
        --group cuda-build \
        --no-install-project
      UV_PYTHON="${python}" uv build \
        --package mlx-lattice-cuda13 \
        --wheel \
        --python "${python}" \
        --no-build-isolation \
        --out-dir "${out_dir}" \
        -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
        -Ccmake.define.MLX_LATTICE_REQUIRE_CUDA=ON
      retag_linux_wheel "mlx_lattice_cuda13" "${python_tag}" "${python}"
      ;;
    *)
      echo "Unknown wheel target: ${target}" >&2
      exit 1
      ;;
  esac

  echo "::endgroup::"
done
