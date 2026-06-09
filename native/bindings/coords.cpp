#include "bindings/registrations.h"

#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "ops/coords.h"

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

nb::tuple relation_tuple(const NativeKernelRelation& relation) {
    return nb::make_tuple(
        relation.in_rows,
        relation.out_rows,
        relation.kernel_ids,
        relation.out_coords,
        relation.counts
    );
}

nb::tuple neighbor_tuple(const NativeNeighborRelation& relation) {
    return nb::make_tuple(
        relation.query_rows,
        relation.source_rows,
        relation.neighbor_ids,
        relation.distances,
        relation.counts
    );
}

nb::tuple coord_set_tuple(const NativeCoordSet& result) {
    return nb::make_tuple(result.coords, result.count);
}

} // namespace

void register_coords(nb::module_& module) {
    module.def(
        "downsample_coords",
        [](const mx::array& coords, const std::vector<int>& stride) {
            return coord_set_tuple(
                downsample_coords(coords, triple_from_values(stride, "stride"))
            );
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
        [](const mx::array& lhs, const mx::array& rhs) {
            return coord_set_tuple(union_coords(lhs, rhs));
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
        [](const mx::array& lhs, const mx::array& rhs) {
            return coord_set_tuple(intersection_coords(lhs, rhs));
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
        &lookup_coords,
        "coords"_a,
        "queries"_a,
        nb::sig(
            "def lookup_coords(coords: mlx.core.array, "
            "queries: mlx.core.array) -> mlx.core.array"
        ),
        "Return row indices of queries in coords, or -1."
    );
    module.def(
        "build_kernel_relation",
        [](const mx::array& coords,
           const mx::array& active_rows,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
            return relation_tuple(build_kernel_relation(
                coords,
                active_rows,
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
            "mlx.core.array, mlx.core.array]"
        ),
        "Build a forward sparse kernel relation."
    );
    module.def(
        "build_generative_relation",
        [](const mx::array& coords,
           const mx::array& active_rows,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride) {
            return relation_tuple(build_generative_relation(
                coords,
                active_rows,
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
            "mlx.core.array, mlx.core.array]"
        ),
        "Build a generative sparse kernel relation."
    );
    module.def(
        "build_transposed_kernel_relation",
        [](const mx::array& coords,
           const mx::array& active_rows,
           const std::vector<int>& kernel_size,
           const std::vector<int>& stride,
           const std::vector<int>& padding,
           const std::vector<int>& dilation) {
            return relation_tuple(build_transposed_kernel_relation(
                coords,
                active_rows,
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
            "mlx.core.array, mlx.core.array]"
        ),
        "Build a transposed sparse kernel relation."
    );
    module.def(
        "build_knn_relation",
        [](const mx::array& source_coords,
           const mx::array& source_active_rows,
           const mx::array& query_coords,
           const mx::array& query_active_rows,
           int k) {
            return neighbor_tuple(build_knn_relation(
                source_coords,
                source_active_rows,
                query_coords,
                query_active_rows,
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
            "mlx.core.array, mlx.core.array]"
        ),
        "Build a semantic KNN neighbor relation."
    );
    module.def(
        "build_radius_relation",
        [](const mx::array& source_coords,
           const mx::array& source_active_rows,
           const mx::array& query_coords,
           const mx::array& query_active_rows,
           double radius,
           int max_neighbors) {
            return neighbor_tuple(build_radius_relation(
                source_coords,
                source_active_rows,
                query_coords,
                query_active_rows,
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
            "mlx.core.array, mlx.core.array, mlx.core.array]"
        ),
        "Build a semantic radius-query neighbor relation."
    );
}

} // namespace mlx_lattice::bindings
