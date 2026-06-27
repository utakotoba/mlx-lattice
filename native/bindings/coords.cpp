#include "bindings/registrations.h"

#include "bindings/array_arg.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "features/coordinates/api.h"

namespace mlx_lattice::bindings {

using namespace nb::literals;

namespace {

Triple triple_from_values(const std::vector<int>& values, const char* name) {
    if (values.size() != 3) {
        throw std::invalid_argument(
            std::string(name) + " must contain exactly 3 values."
        );
    }
    return {values[0], values[1], values[2]};
}

FloatTriple
float_triple_from_values(const std::vector<float>& values, const char* name) {
    if (values.size() != 3) {
        throw std::invalid_argument(
            std::string(name) + " must contain exactly 3 values."
        );
    }
    return {values[0], values[1], values[2]};
}

VoxelReduceOp voxel_reduce_from_name(const std::string& name) {
    if (name == "sum") {
        return VoxelReduceOp::Sum;
    }
    if (name == "mean") {
        return VoxelReduceOp::Mean;
    }
    throw std::invalid_argument("voxel reduction must be 'sum' or 'mean'.");
}

nb::tuple relation_tuple(const NativeKernelRelation& relation) {
    return nb::make_tuple(
        relation.in_rows,
        relation.out_rows,
        relation.kernel_ids,
        relation.output_row_offsets,
        relation.out_coords,
        relation.counts,
        relation.in_row_offsets,
        relation.in_edge_ids,
        relation.kernel_row_offsets,
        relation.kernel_edge_ids
    );
}

nb::tuple neighbor_tuple(const NativeNeighborRelation& relation) {
    return nb::make_tuple(
        relation.query_rows,
        relation.source_rows,
        relation.neighbor_ids,
        relation.distances,
        relation.row_offsets,
        relation.counts
    );
}

nb::tuple implicit_gemm_view_tuple(const NativeRelationImplicitGemmView& view) {
    return nb::make_tuple(view.out_in_map, view.row_masks);
}

nb::tuple coord_set_tuple(const NativeCoordSet& result) {
    return nb::make_tuple(result.coords, result.count);
}

nb::tuple quantization_tuple(const NativeSparseQuantization& result) {
    return nb::make_tuple(
        result.coords, result.active_rows, result.inverse_rows, result.counts
    );
}

} // namespace

void register_coords(nb::module_& module) {
    module.def(
        "downsample_coords",
        [](nb::handle coords, const std::vector<int>& stride) {
            return coord_set_tuple(downsample_coords(
                array_arg(coords, "coords"),
                triple_from_values(stride, "stride")
            ));
        },
        "coords"_a,
        "stride"_a,
        nb::sig(
            "def downsample_coords(coords: mlx.core.array, "
            "stride: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array]"
        ),
        "Downsample sparse coordinates."
    );
    module.def(
        "union_coords",
        [](nb::handle lhs, nb::handle rhs) {
            return coord_set_tuple(
                union_coords(array_arg(lhs, "lhs"), array_arg(rhs, "rhs"))
            );
        },
        "lhs"_a,
        "rhs"_a,
        nb::sig(
            "def union_coords(lhs: mlx.core.array, rhs: mlx.core.array) -> "
            "tuple[mlx.core.array, mlx.core.array]"
        ),
        "Return the ordered union of two coordinate arrays."
    );
    module.def(
        "intersection_coords",
        [](nb::handle lhs, nb::handle rhs) {
            return coord_set_tuple(intersection_coords(
                array_arg(lhs, "lhs"), array_arg(rhs, "rhs")
            ));
        },
        "lhs"_a,
        "rhs"_a,
        nb::sig(
            "def intersection_coords(lhs: mlx.core.array, "
            "rhs: mlx.core.array) -> tuple[mlx.core.array, mlx.core.array]"
        ),
        "Return the ordered intersection of two coordinate arrays."
    );
    module.def(
        "lookup_coords",
        [](nb::handle coords, nb::handle queries) {
            return lookup_coords(
                array_arg(coords, "coords"), array_arg(queries, "queries")
            );
        },
        "coords"_a,
        "queries"_a,
        nb::sig(
            "def lookup_coords(coords: mlx.core.array, "
            "queries: mlx.core.array) -> mlx.core.array"
        ),
        "Return row indices of queries in coords, or -1."
    );
    module.def(
        "morton_codes",
        [](nb::handle coords) {
            return morton_codes(array_arg(coords, "coords"));
        },
        "coords"_a,
        nb::sig("def morton_codes(coords: mlx.core.array) -> mlx.core.array"),
        "Return Gameleon-compatible 3D Morton codes for sparse coordinates."
    );
    module.def(
        "occupancy_downsample",
        [](nb::handle coords, nb::handle active_rows) {
            const auto& result = occupancy_downsample(
                array_arg(coords, "coords"),
                array_arg(active_rows, "active_rows")
            );
            return nb::make_tuple(
                result.coords, result.active_rows, result.occupancy
            );
        },
        "coords"_a,
        "active_rows"_a,
        nb::sig(
            "def occupancy_downsample(coords: mlx.core.array, active_rows: "
            "mlx.core.array) -> tuple[mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Downsample coordinates into parent occupancy codes."
    );
    module.def(
        "occupancy_expand",
        [](nb::handle coords, nb::handle active_rows, nb::handle occupancy) {
            const auto& result = occupancy_expand(
                array_arg(coords, "coords"),
                array_arg(active_rows, "active_rows"),
                array_arg(occupancy, "occupancy")
            );
            return nb::make_tuple(
                result.coords,
                result.active_rows,
                result.parent_rows,
                result.child_indices
            );
        },
        "coords"_a,
        "active_rows"_a,
        "occupancy"_a,
        nb::sig(
            "def occupancy_expand(coords: mlx.core.array, active_rows: "
            "mlx.core.array, occupancy: mlx.core.array) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Expand coordinates from parent occupancy codes."
    );
    module.def(
        "child_coords_from_indices",
        [](nb::handle parent_coords, nb::handle child_indices) {
            return child_coords_from_indices(
                array_arg(parent_coords, "parent_coords"),
                array_arg(child_indices, "child_indices")
            );
        },
        "parent_coords"_a,
        "child_indices"_a,
        nb::sig(
            "def child_coords_from_indices(parent_coords: mlx.core.array, "
            "child_indices: mlx.core.array) -> mlx.core.array"
        ),
        "Expand parent coordinates by child indices."
    );
    module.def(
        "sparse_quantize",
        [](nb::handle points,
           nb::handle batch_indices,
           nb::handle active_rows,
           const std::vector<float>& voxel_size,
           const std::vector<float>& origin) {
            return quantization_tuple(sparse_quantize(
                array_arg(points, "points"),
                array_arg(batch_indices, "batch_indices"),
                array_arg(active_rows, "active_rows"),
                QuantizationSpec{
                    float_triple_from_values(voxel_size, "voxel_size"),
                    float_triple_from_values(origin, "origin"),
                }
            ));
        },
        "points"_a,
        "batch_indices"_a,
        "active_rows"_a,
        "voxel_size"_a,
        "origin"_a,
        nb::sig(
            "def sparse_quantize(points: mlx.core.array, "
            "batch_indices: mlx.core.array, active_rows: mlx.core.array, "
            "voxel_size: collections.abc.Sequence[float], "
            "origin: collections.abc.Sequence[float]) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Quantize dense points into ordered sparse voxel coordinates."
    );
    module.def(
        "voxelize_features",
        [](nb::handle feats,
           nb::handle inverse_rows,
           nb::handle voxel_counts,
           nb::handle active_rows,
           const std::string& reduction) {
            return voxelize_features(
                array_arg(feats, "feats"),
                array_arg(inverse_rows, "inverse_rows"),
                array_arg(voxel_counts, "voxel_counts"),
                array_arg(active_rows, "active_rows"),
                voxel_reduce_from_name(reduction)
            );
        },
        "feats"_a,
        "inverse_rows"_a,
        "voxel_counts"_a,
        "active_rows"_a,
        "reduction"_a,
        nb::sig(
            "def voxelize_features(feats: mlx.core.array, "
            "inverse_rows: mlx.core.array, voxel_counts: mlx.core.array, "
            "active_rows: mlx.core.array, reduction: str) -> mlx.core.array"
        ),
        "Aggregate point features into sparse voxel rows."
    );
    module.def(
        "build_kernel_relation",
        [](nb::handle coords,
           nb::handle active_rows,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
            return relation_tuple(build_kernel_relation(
                array_arg(coords, "coords"),
                array_arg(active_rows, "active_rows"),
                triple_from_values(kernel_size, "kernel_size"),
                triple_from_values(stride, "stride"),
                triple_from_values(padding, "padding"),
                triple_from_values(dilation, "dilation")
            ));
        },
        "coords"_a,
        "active_rows"_a,
        "kernel_size"_a,
        "stride"_a,
        "padding"_a,
        "dilation"_a,
        nb::sig(
            "def build_kernel_relation(coords: mlx.core.array, "
            "active_rows: mlx.core.array, "
            "kernel_size: collections.abc.Sequence[int], "
            "stride: collections.abc.Sequence[int], "
            "padding: collections.abc.Sequence[int], "
            "dilation: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Build a forward sparse kernel relation."
    );
    module.def(
        "build_generative_relation",
        [](nb::handle coords,
           nb::handle active_rows,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride) {
            return relation_tuple(build_generative_relation(
                array_arg(coords, "coords"),
                array_arg(active_rows, "active_rows"),
                triple_from_values(kernel_size, "kernel_size"),
                triple_from_values(stride, "stride")
            ));
        },
        "coords"_a,
        "active_rows"_a,
        "kernel_size"_a,
        "stride"_a,
        nb::sig(
            "def build_generative_relation(coords: mlx.core.array, "
            "active_rows: mlx.core.array, "
            "kernel_size: collections.abc.Sequence[int], "
            "stride: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Build a generative sparse kernel relation."
    );
    module.def(
        "build_transposed_kernel_relation",
        [](nb::handle coords,
           nb::handle active_rows,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
            return relation_tuple(build_transposed_kernel_relation(
                array_arg(coords, "coords"),
                array_arg(active_rows, "active_rows"),
                triple_from_values(kernel_size, "kernel_size"),
                triple_from_values(stride, "stride"),
                triple_from_values(padding, "padding"),
                triple_from_values(dilation, "dilation")
            ));
        },
        "coords"_a,
        "active_rows"_a,
        "kernel_size"_a,
        "stride"_a,
        "padding"_a,
        "dilation"_a,
        nb::sig(
            "def build_transposed_kernel_relation(coords: mlx.core.array, "
            "active_rows: mlx.core.array, "
            "kernel_size: collections.abc.Sequence[int], "
            "stride: collections.abc.Sequence[int], "
            "padding: collections.abc.Sequence[int], "
            "dilation: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Build a transposed sparse kernel relation."
    );
    module.def(
        "build_target_kernel_relation",
        [](nb::handle coords,
           nb::handle active_rows,
           nb::handle target_coords,
           nb::handle target_active_rows,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
            return relation_tuple(build_target_kernel_relation(
                array_arg(coords, "coords"),
                array_arg(active_rows, "active_rows"),
                array_arg(target_coords, "target_coords"),
                array_arg(target_active_rows, "target_active_rows"),
                triple_from_values(kernel_size, "kernel_size"),
                triple_from_values(stride, "stride"),
                triple_from_values(padding, "padding"),
                triple_from_values(dilation, "dilation")
            ));
        },
        "coords"_a,
        "active_rows"_a,
        "target_coords"_a,
        "target_active_rows"_a,
        "kernel_size"_a,
        "stride"_a,
        "padding"_a,
        "dilation"_a,
        nb::sig(
            "def build_target_kernel_relation(coords: mlx.core.array, "
            "active_rows: mlx.core.array, target_coords: mlx.core.array, "
            "target_active_rows: mlx.core.array, "
            "kernel_size: collections.abc.Sequence[int], "
            "stride: collections.abc.Sequence[int], "
            "padding: collections.abc.Sequence[int], "
            "dilation: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Build a target-coordinate sparse kernel relation."
    );
    module.def(
        "build_relation_implicit_gemm_view",
        [](nb::handle source_coords,
           nb::handle source_active_rows,
           nb::handle output_coords,
           nb::handle output_active_rows,
           nb::handle offsets,
           const std::vector<int>& stride,
           const std::vector<int>& padding) {
            return implicit_gemm_view_tuple(build_relation_implicit_gemm_view(
                array_arg(source_coords, "source_coords"),
                array_arg(source_active_rows, "source_active_rows"),
                array_arg(output_coords, "output_coords"),
                array_arg(output_active_rows, "output_active_rows"),
                array_arg(offsets, "offsets"),
                CoordRelationOp::Forward,
                triple_from_values(stride, "stride"),
                triple_from_values(padding, "padding")
            ));
        },
        "source_coords"_a,
        "source_active_rows"_a,
        "output_coords"_a,
        "output_active_rows"_a,
        "offsets"_a,
        "stride"_a,
        "padding"_a,
        nb::sig(
            "def build_relation_implicit_gemm_view(source_coords: "
            "mlx.core.array, source_active_rows: mlx.core.array, "
            "output_coords: mlx.core.array, output_active_rows: "
            "mlx.core.array, offsets: mlx.core.array, "
            "stride: collections.abc.Sequence[int], "
            "padding: collections.abc.Sequence[int]) -> "
            "tuple[mlx.core.array, mlx.core.array]"
        ),
        "Build a direct implicit-GEMM relation view."
    );
    module.def(
        "build_knn_relation",
        [](nb::handle source_coords,
           nb::handle source_active_rows,
           nb::handle query_coords,
           nb::handle query_active_rows,
           int k) {
            return neighbor_tuple(build_knn_relation(
                array_arg(source_coords, "source_coords"),
                array_arg(source_active_rows, "source_active_rows"),
                array_arg(query_coords, "query_coords"),
                array_arg(query_active_rows, "query_active_rows"),
                k
            ));
        },
        "source_coords"_a,
        "source_active_rows"_a,
        "query_coords"_a,
        "query_active_rows"_a,
        "k"_a,
        nb::sig(
            "def build_knn_relation(source_coords: mlx.core.array, "
            "source_active_rows: mlx.core.array, "
            "query_coords: mlx.core.array, "
            "query_active_rows: mlx.core.array, k: int) -> "
            "tuple[mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array, mlx.core.array, mlx.core.array]"
        ),
        "Build a semantic KNN neighbor relation."
    );
    module.def(
        "build_radius_relation",
        [](nb::handle source_coords,
           nb::handle source_active_rows,
           nb::handle query_coords,
           nb::handle query_active_rows,
           double radius,
           int max_neighbors) {
            return neighbor_tuple(build_radius_relation(
                array_arg(source_coords, "source_coords"),
                array_arg(source_active_rows, "source_active_rows"),
                array_arg(query_coords, "query_coords"),
                array_arg(query_active_rows, "query_active_rows"),
                radius,
                max_neighbors
            ));
        },
        "source_coords"_a,
        "source_active_rows"_a,
        "query_coords"_a,
        "query_active_rows"_a,
        "radius"_a,
        "max_neighbors"_a,
        nb::sig(
            "def build_radius_relation(source_coords: mlx.core.array, "
            "source_active_rows: mlx.core.array, "
            "query_coords: mlx.core.array, "
            "query_active_rows: mlx.core.array, radius: float, "
            "max_neighbors: int) -> tuple[mlx.core.array, mlx.core.array, "
            "mlx.core.array, mlx.core.array, mlx.core.array, "
            "mlx.core.array]"
        ),
        "Build a semantic radius-query neighbor relation."
    );
}

} // namespace mlx_lattice::bindings
