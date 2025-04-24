#include "w2_bssn_uniform_grid.hpp"

w2_bssn_uniform_grid
w2_bssn_uniform_grid::euler_step(const time_derivative_type& dfdt, const real dt) const {
    auto f = w2_bssn_uniform_grid{ *this };
    f.W_.for_each_index([&](const auto idx) {
        f.W_[idx][] += dt * dfdt.W[idx][];
        f.lapse_[idx][] += dt * dfdt.lapse[idx][];
        f.K_[idx][] += dt * dfdt.K[idx][];
    });

    f.contraconf_christoffel_trace_.for_each_index([&](const auto idx, const auto tidx) {
        f.contraconf_christoffel_trace_[idx][tidx] +=
            dt * dfdt.contraconf_christoffel_trace[idx][tidx];
    });

    f.coconf_A_.for_each_index([&](const auto idx, const auto tidx) {
        f.coconf_A_[idx][tidx] += dt * dfdt.coconf_A[idx][tidx];
        f.coconf_metric_[idx][tidx] += dt * dfdt.coconf_metric[idx][tidx];
    });

    return f;
}
