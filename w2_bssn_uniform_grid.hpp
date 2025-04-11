#pragma once

#include <cstddef>
#include <vector>

#include "grid_types.hpp"
#include "tensor_buffer.hpp"

/// Represents grid of point on a 3D spatial slice of 4D space time in bssn formalism
/// with W^2 conformal decomposition.
///
/// Points are at coordinates (i, j, k), where {i,j,k} is in [0, N_{i,j,k}).
class w2_bssn_uniform_grid {
    std::size_t grid_points_x, grid_points_y, grid_points_z;

    using buffer0 = tensor_buffer<0, 3, real, std::allocator<real>>;
    using buffer1 = tensor_buffer<1, 3, real, std::allocator<real>>;

    buffer0 conformal_coefficient_;
    buffer0 lapse_;

    buffer1 shift_;

  public:
    [[nodiscard]]
    explicit w2_bssn_uniform_grid(const grid_size gs);
};
