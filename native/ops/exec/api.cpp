#include "ops/exec.h"

#include "ops/exec/dispatch.h"
#include "ops/exec/validation.h"

namespace mlx_lattice {

mx::array spmm_edges(
    const mx::array& feats,
    const mx::array& weights,
    const mx::array& in_rows,
    const mx::array& out_rows,
    const mx::array& kernel_ids,
    int n_out_rows
) {
    validate_spmm_edges(
        feats, weights, in_rows, out_rows, kernel_ids, n_out_rows
    );
    return dispatch_spmm_edges(
        feats, weights, in_rows, out_rows, kernel_ids, n_out_rows
    );
}

} // namespace mlx_lattice
