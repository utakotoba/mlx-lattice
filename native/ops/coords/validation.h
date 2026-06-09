#pragma once

#include "ops/coords.h"

namespace mlx_lattice {

void validate_coords(const mx::array& coords);
void validate_coord_pair(const mx::array& lhs, const mx::array& rhs);
void validate_points(const mx::array& points);
void validate_active_rows(const mx::array& active_rows);
void validate_batch_indices(const mx::array& batch_indices, int rows);
void validate_inverse_rows(const mx::array& inverse_rows, int rows);
void validate_voxel_counts(const mx::array& voxel_counts, int rows);
void validate_feature_matrix(const mx::array& feats);
void validate_positive(Triple values, const char* name);
void validate_positive(FloatTriple values, const char* name);
void validate_nonnegative(Triple values, const char* name);

} // namespace mlx_lattice
