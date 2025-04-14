#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "grid_types.hpp"
#include "tensor_buffer.hpp"

struct minkowski_spacetime_tag {};

/// Represents grid of point on a 3D spatial slice of 4D space time in bssn formalism
/// with W^2 conformal decomposition.
///
/// Points are at coordinates (i, j, k), where {i,j,k} is in [0, N_{i,j,k}).
class w2_bssn_uniform_grid {
  public:
    using buffer0 = tensor_buffer<0, 3, real, std::allocator<real>>;
    using buffer1 = tensor_buffer<1, 3, real, std::allocator<real>>;
    using buffer2 = tensor_buffer<2, 3, real, std::allocator<real>>;

  private:
    std::size_t grid_points_x, grid_points_y, grid_points_z;

    buffer0 W_;
    buffer0 lapse_;
    buffer0 extrinsic_curvature_trace_;

    buffer1 shift_;
    buffer1 contravariant_conformal_christoffel_trace_;

    buffer2 covariant_conformal_spatial_metric_;
    buffer2 contravariant_conformal_spatial_metric_;
    buffer2 covariant_conformal_A_;

  public:
    [[nodiscard]]
    explicit w2_bssn_uniform_grid(const grid_size gs, minkowski_spacetime_tag);

    void beve_dump(const std::filesystem::path& dump_dir_name = "./w2_bssn_uniform_grid_dump");
};
