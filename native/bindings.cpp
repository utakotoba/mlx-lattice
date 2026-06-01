#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "lattice/runtime.h"
#include "ops/conv3d.h"
#include "ops/coords.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "Native extension for mlx-lattice.";

    m.def("version", &mlx_lattice::version);
    m.def("capabilities", []() {
        auto caps = mlx_lattice::capabilities();
        nb::dict out;
        out["cpu"] = caps.cpu;
        out["metal"] = caps.metal;
        out["cuda"] = caps.cuda;
        out["rocm"] = caps.rocm;
        return out;
    });
    m.def(
        "conv3d_feats",
        &mlx_lattice::conv3d_feats,
        "feats"_a,
        "weight"_a,
        "maps"_a,
        "kernels"_a,
        "out_rows"_a,
        nb::kw_only(),
        "stream"_a = nb::none()
    );
    m.def(
        "conv3d_subm_feats",
        &mlx_lattice::conv3d_subm_feats,
        "feats"_a,
        "weight"_a,
        "maps"_a,
        "kernels"_a,
        "center_kernel"_a,
        nb::kw_only(),
        "stream"_a = nb::none()
    );
    m.def(
        "conv3d_residual_feats",
        &mlx_lattice::conv3d_residual_feats,
        "base"_a,
        "feats"_a,
        "weight"_a,
        "maps"_a,
        "kernels"_a,
        "offsets"_a,
        nb::kw_only(),
        "stream"_a = nb::none()
    );
    m.def(
        "pool3d_feats",
        &mlx_lattice::pool3d_feats,
        "feats"_a,
        "maps"_a,
        "kernels"_a,
        "offsets"_a,
        "out_rows"_a,
        nb::kw_only(),
        "stream"_a = nb::none()
    );
    m.def(
        "downsample_coords",
        [](const mlx_lattice::mx::array& coords, int sx, int sy, int sz) {
            return mlx_lattice::downsample_coords(coords, {sx, sy, sz});
        },
        "coords"_a,
        "sx"_a,
        "sy"_a,
        "sz"_a
    );
    m.def(
        "build_kernel_map",
        [](const mlx_lattice::mx::array& coords,
           int kx,
           int ky,
           int kz,
           int sx,
           int sy,
           int sz,
           int px,
           int py,
           int pz) {
            auto out = mlx_lattice::build_kernel_map(
                coords, {kx, ky, kz}, {sx, sy, sz}, {px, py, pz}
            );
            return nb::make_tuple(
                out.maps,
                out.sizes,
                out.kernels,
                out.residual_maps,
                out.residual_kernels,
                out.residual_offsets,
                out.out_coords,
                out.offsets
            );
        },
        "coords"_a,
        "kx"_a,
        "ky"_a,
        "kz"_a,
        "sx"_a,
        "sy"_a,
        "sz"_a,
        "px"_a,
        "py"_a,
        "pz"_a
    );
    m.def(
        "build_generative_map",
        [](const mlx_lattice::mx::array& coords,
           int kx,
           int ky,
           int kz,
           int sx,
           int sy,
           int sz) {
            auto out = mlx_lattice::build_generative_map(
                coords, {kx, ky, kz}, {sx, sy, sz}
            );
            return nb::make_tuple(
                out.maps,
                out.sizes,
                out.kernels,
                out.residual_maps,
                out.residual_kernels,
                out.residual_offsets,
                out.out_coords,
                out.offsets
            );
        },
        "coords"_a,
        "kx"_a,
        "ky"_a,
        "kz"_a,
        "sx"_a,
        "sy"_a,
        "sz"_a
    );
}
