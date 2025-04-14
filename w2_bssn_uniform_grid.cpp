#include "w2_bssn_uniform_grid.hpp"

#include <experimental/mdspan>

#include <cmath>
#include <tuple>
#include <type_traits>

#include "idg/einsum.hpp"
using namespace idg::literals;

[[nodiscard]]
w2_bssn_uniform_grid::w2_bssn_uniform_grid(const grid_size gs, minkowski_spacetime_tag)
    : W_(gs),
      lapse_(gs),
      shift_(gs),
      covariant_conformal_spatial_metric_(gs),
      contravariant_conformal_spatial_metric_(gs),
      extrinsic_curvature_trace_(gs),
      covariant_conformal_A_(gs),
      contravariant_conformal_christoffel_trace_(gs) {
    covariant_conformal_spatial_metric_.for_each_index([&](const auto idx, const auto tidx) {
        covariant_conformal_spatial_metric_[idx][tidx] = static_cast<real>(tidx[0] == tidx[1]);
    });
    contravariant_conformal_spatial_metric_.for_each_index([&](const auto idx, const auto tidx) {
        contravariant_conformal_spatial_metric_[idx][tidx] = static_cast<real>(tidx[0] == tidx[1]);
    });

    lapse_.for_each_index([&](const auto idx) { lapse_[idx][] = 1; });
    shift_.for_each_index([&](const auto idx, const auto tidx) { shift_[idx][tidx] = 0; });

    extrinsic_curvature_trace_.for_each_index(
        [&](const auto idx) { extrinsic_curvature_trace_[idx][] = 0; });
    W_.for_each_index([&](const auto idx) { W_[idx][] = 1; });

    covariant_conformal_A_.for_each_index(
        [&](const auto idx, const auto tidx) { covariant_conformal_A_[idx][tidx] = 0; });

    contravariant_conformal_christoffel_trace_.for_each_index([&](const auto idx, const auto tidx) {
        contravariant_conformal_christoffel_trace_[idx][tidx] = 0;
    });
}

[[nodiscard]]
w2_bssn_uniform_grid::buffer0
determinant3D(const w2_bssn_uniform_grid::buffer2& matrix) {
    auto det = w2_bssn_uniform_grid::buffer0(matrix.size());
    det.for_each_index([&](const auto idx) {
        const auto a   = matrix[idx][0, 0];
        const auto b   = matrix[idx][0, 1];
        const auto c   = matrix[idx][0, 2];
        const auto d   = matrix[idx][1, 0];
        const auto e   = matrix[idx][1, 1];
        const auto f   = matrix[idx][1, 2];
        const auto g   = matrix[idx][2, 0];
        const auto h   = matrix[idx][2, 1];
        const auto i   = matrix[idx][2, 2];
        const auto aei = a * e * i;
        const auto bfg = b * f * g;
        const auto cdh = c * d * h;
        const auto ceg = c * e * g;
        const auto bdi = b * d * i;
        const auto afh = a * f * h;
        det[idx][]     = aei + bfg + cdh - ceg - bdi - afh;
    });

    return det;
}

void
w2_bssn_uniform_grid::beve_dump(const std::filesystem::path& dump_dir_name) {
    const auto dir_path = std::filesystem::weakly_canonical(dump_dir_name);
    std::filesystem::create_directory(dir_path);
    auto file_path = [&](const std::filesystem::path& filename) { return dir_path / filename; };

    {
        auto S = determinant3D(covariant_conformal_spatial_metric_);
        S.for_each_index([&](const auto idx) { S[idx][] -= 1; });
        S.write_as_beve(file_path("algebraic_constraint_S.beve"));
    }

    {
        auto conformal_A_trace = buffer0(covariant_conformal_A_.size());
        conformal_A_trace.for_each_index([&](const auto idx) {
            u8"ij,ij"_einsum(conformal_A_trace[idx],
                             contravariant_conformal_spatial_metric_[idx],
                             covariant_conformal_A_[idx]);
        });
        conformal_A_trace.write_as_beve(file_path("algebraic_constraint_conformal_A.beve"));
    }
}
