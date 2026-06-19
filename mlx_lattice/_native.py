from __future__ import annotations

from mlx_lattice import _ext as ext
from mlx_lattice.backends import cuda


class NativeOps:
    def downsample_coords(self, coords, stride):
        if cuda.selected():
            return cuda.downsample_coords(coords, stride)
        return ext.downsample_coords(coords, stride)

    def union_coords(self, lhs, rhs):
        if cuda.selected():
            return cuda.union_coords(lhs, rhs)
        return ext.union_coords(lhs, rhs)

    def intersection_coords(self, lhs, rhs):
        if cuda.selected():
            return cuda.intersection_coords(lhs, rhs)
        return ext.intersection_coords(lhs, rhs)

    def lookup_coords(self, coords, queries):
        if cuda.selected():
            return cuda.lookup_coords(coords, queries)
        return ext.lookup_coords(coords, queries)

    def morton_codes(self, coords):
        if cuda.selected():
            return cuda.morton_codes(coords)
        return ext.morton_codes(coords)

    def occupancy_downsample(self, coords, active_rows):
        if cuda.selected():
            return cuda.occupancy_downsample(coords, active_rows)
        return ext.occupancy_downsample(coords, active_rows)

    def occupancy_expand(self, coords, active_rows, occupancy):
        if cuda.selected():
            return cuda.occupancy_expand(coords, active_rows, occupancy)
        return ext.occupancy_expand(coords, active_rows, occupancy)

    def child_coords_from_indices(self, parent_coords, child_indices):
        if cuda.selected():
            return cuda.child_coords_from_indices(
                parent_coords, child_indices
            )
        return ext.child_coords_from_indices(parent_coords, child_indices)

    def sparse_quantize(
        self,
        points,
        batch_indices,
        active_rows,
        voxel_size,
        origin,
    ):
        if cuda.selected():
            return cuda.sparse_quantize(
                points,
                batch_indices,
                active_rows,
                voxel_size,
                origin,
            )
        return ext.sparse_quantize(
            points,
            batch_indices,
            active_rows,
            voxel_size,
            origin,
        )

    def voxelize_features(
        self,
        feats,
        inverse_rows,
        voxel_counts,
        active_rows,
        reduction,
    ):
        if cuda.selected():
            return cuda.voxelize_features(
                feats,
                inverse_rows,
                voxel_counts,
                active_rows,
                reduction,
            )
        return ext.voxelize_features(
            feats,
            inverse_rows,
            voxel_counts,
            active_rows,
            reduction,
        )

    def sparse_conv_features(
        self,
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        in_row_offsets,
        in_edge_ids,
        kernel_row_offsets,
        kernel_edge_ids,
        out_capacity,
        n_kernels,
    ):
        if cuda.selected():
            return cuda.sparse_conv_features(
                feats,
                weights,
                in_rows,
                out_rows,
                kernel_ids,
                counts,
                row_offsets,
                in_row_offsets,
                in_edge_ids,
                kernel_row_offsets,
                kernel_edge_ids,
                out_capacity,
                n_kernels,
            )
        return ext.sparse_conv_features(
            feats,
            weights,
            in_rows,
            out_rows,
            kernel_ids,
            counts,
            row_offsets,
            in_row_offsets,
            in_edge_ids,
            kernel_row_offsets,
            kernel_edge_ids,
            out_capacity,
            n_kernels,
        )

    def sparse_pool_features(
        self,
        feats,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        in_row_offsets,
        in_edge_ids,
        input_exclusive,
        reduce,
        out_capacity,
        n_kernels,
    ):
        if cuda.selected():
            return cuda.sparse_pool_features(
                feats,
                in_rows,
                out_rows,
                kernel_ids,
                row_offsets,
                counts,
                in_row_offsets,
                in_edge_ids,
                input_exclusive,
                reduce,
                out_capacity,
                n_kernels,
            )
        return ext.sparse_pool_features(
            feats,
            in_rows,
            out_rows,
            kernel_ids,
            row_offsets,
            counts,
            input_exclusive,
            reduce,
            out_capacity,
            n_kernels,
        )


native = NativeOps()


def backend_info() -> dict[str, object]:
    native_capabilities = ext.capabilities()
    capabilities = dict(native_capabilities)
    capabilities['cuda'] = cuda.runtime_available()
    return {
        'version': ext.version(),
        'capabilities': capabilities,
        'native_capabilities': native_capabilities,
        'cuda': cuda.info(),
    }
