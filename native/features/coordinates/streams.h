#pragma once

#include "features/coordinates/types.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace mlx_lattice {

mx::Device coord_device();
mx::Stream coord_stream(const mx::Device& device);

} // namespace mlx_lattice
