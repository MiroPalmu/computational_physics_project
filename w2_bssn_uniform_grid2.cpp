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

namespace finite_difference {

/// Calculates derivative of arbitrary tensor T_{abc...} -> T_{abc...,i}
///
/// Assumes that tensor buffer elements are at seperated by h = 1.
template<std::size_t rank, typename T, typename Allocator>
[[nodiscard]]
auto
periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
    const tensor_buffer<rank, 3uz, T, Allocator>& f) {
    using buff_type = tensor_buffer<rank, 3uz, T, Allocator>;
    auto sum        = buff_type(f.size());

    sum.for_each_index([&](const auto idx, const auto tidx) {
        const auto iuz = idx[0];
        const auto juz = idx[1];
        const auto kuz = idx[2];

        const auto i = static_cast<std::ptrdiff_t>(iuz);
        const auto j = static_cast<std::ptrdiff_t>(juz);
        const auto k = static_cast<std::ptrdiff_t>(kuz);

        const auto im3 = (i - 3) % f.size().Nx;
        const auto im2 = (i - 2) % f.size().Nx;
        const auto im1 = (i - 1) % f.size().Nx;
        const auto ip1 = (i + 1) % f.size().Nx;
        const auto ip2 = (i + 2) % f.size().Nx;
        const auto ip3 = (i + 3) % f.size().Nx;

        const auto jm3 = (j - 3) % f.size().Ny;
        const auto jm2 = (j - 2) % f.size().Ny;
        const auto jm1 = (j - 1) % f.size().Ny;
        const auto jp1 = (j + 1) % f.size().Ny;
        const auto jp2 = (j + 2) % f.size().Ny;
        const auto jp3 = (j + 3) % f.size().Ny;

        const auto km3 = (k - 3) % f.size().Nz;
        const auto km2 = (k - 2) % f.size().Nz;
        const auto km1 = (k - 1) % f.size().Nz;
        const auto kp1 = (k + 1) % f.size().Nz;
        const auto kp2 = (k + 2) % f.size().Nz;
        const auto kp3 = (k + 3) % f.size().Nz;

        static constexpr auto a = T{ 6 };
        static constexpr auto b = T{ 15 };
        static constexpr auto c = T{ 20 };

        // x
        sum[idx][tidx] = f[{ im3, juz, kuz }][tidx] - a * f[{ im2, juz, kuz }][tidx]
                         + b * f[{ im1, juz, kuz }][tidx] - c * f[{ iuz, juz, kuz }][tidx]
                         + b * f[{ ip1, juz, kuz }][tidx] - a * f[{ ip2, juz, kuz }][tidx]
                         + f[{ ip3, juz, kuz }][tidx];

        // y
        sum[idx][tidx] += f[{ iuz, jm3, kuz }][tidx] - a * f[{ iuz, jm2, kuz }][tidx]
                          + b * f[{ iuz, jm1, kuz }][tidx] - c * f[{ iuz, juz, kuz }][tidx]
                          + b * f[{ iuz, jp1, kuz }][tidx] - a * f[{ iuz, jp2, kuz }][tidx]
                          + f[{ iuz, jp3, kuz }][tidx];

        // z
        sum[idx][tidx] += f[{ iuz, juz, km3 }][tidx] - a * f[{ iuz, juz, km2 }][tidx]
                          + b * f[{ iuz, juz, km1 }][tidx] - c * f[{ iuz, juz, kuz }][tidx]
                          + b * f[{ iuz, juz, kp1 }][tidx] - a * f[{ iuz, juz, kp2 }][tidx]
                          + f[{ iuz, juz, kp3 }][tidx];
    });

    return sum;
}
} // namespace finite_difference

void
w2_bssn_uniform_grid::time_derivative_type::kreiss_oliger_6th_order(const w2_bssn_uniform_grid& U) {
    /// Hard coded from NR101.
    static constexpr auto coeff = real{ 0.25 } / real{ 64 };

    {
        const auto lapse_derivative_sum =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U.lapse_);
        const auto W_derivative_sum =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U.W_);
        const auto K_derivative_sum =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U.K_);
        W.for_each_index([&](const auto idx) {
            lapse[idx][] += coeff * lapse_derivative_sum[idx][];
            W[idx][] += coeff * W_derivative_sum[idx][];
            K[idx][] += coeff * K_derivative_sum[idx][];
        });
    }

    {
        const auto contraconf_christoffel_trace_derivative_sum =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U.contraconf_christoffel_trace_);
        contraconf_christoffel_trace.for_each_index([&](const auto idx, const auto tidx) {
            contraconf_christoffel_trace[idx][tidx] +=
                coeff * contraconf_christoffel_trace_derivative_sum[idx][tidx];
        });
    }

    {
        const auto coconf_metric_derivative_sum =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U.coconf_metric_);

        coconf_metric.for_each_index([&](const auto idx, const auto tidx) {
            coconf_metric[idx][tidx] += coeff * coconf_metric_derivative_sum[idx][tidx];
        });
    }
}
