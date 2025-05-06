#include "w2_bssn_uniform_grid.hpp"

#include <sycl/sycl.hpp>

w2_bssn_uniform_grid
w2_bssn_uniform_grid::euler_step(const time_derivative_type& dfdt, const real dt) const {
    auto f               = w2_bssn_uniform_grid{ *this };
    auto* const f_ptr    = &f;
    auto* const dfdt_ptr = &dfdt;

    f.W_.for_each_index([=](const auto idx) {
        f_ptr->W_[idx][] += dt * dfdt_ptr->W[idx][];
        f_ptr->lapse_[idx][] += dt * dfdt_ptr->lapse[idx][];
        f_ptr->K_[idx][] += dt * dfdt_ptr->K[idx][];
    });

    f.contraconf_christoffel_trace_.for_each_index([=](const auto idx, const auto tidx) {
        f_ptr->contraconf_christoffel_trace_[idx][tidx] +=
            dt * dfdt_ptr->contraconf_christoffel_trace[idx][tidx];
    });

    f.coconf_A_.for_each_index([=](const auto idx, const auto tidx) {
        f_ptr->coconf_A_[idx][tidx] += dt * dfdt_ptr->coconf_A[idx][tidx];
        f_ptr->coconf_metric_[idx][tidx] += dt * dfdt_ptr->coconf_metric[idx][tidx];
    });

    tensor_buffer_queue.wait();
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

    auto* const sum_ptr        = &sum;
    auto* const f_ptr          = &f;
    const auto [fNx, fNy, fNz] = f.size();

    sum.for_each_index([=](const auto idx, const auto tidx) {
        const auto iuz = idx[0];
        const auto juz = idx[1];
        const auto kuz = idx[2];

        const auto i = static_cast<std::ptrdiff_t>(iuz);
        const auto j = static_cast<std::ptrdiff_t>(juz);
        const auto k = static_cast<std::ptrdiff_t>(kuz);

        const auto im3 = (i - 3) % fNx;
        const auto im2 = (i - 2) % fNx;
        const auto im1 = (i - 1) % fNx;
        const auto ip1 = (i + 1) % fNx;
        const auto ip2 = (i + 2) % fNx;
        const auto ip3 = (i + 3) % fNx;

        const auto jm3 = (j - 3) % fNy;
        const auto jm2 = (j - 2) % fNy;
        const auto jm1 = (j - 1) % fNy;
        const auto jp1 = (j + 1) % fNy;
        const auto jp2 = (j + 2) % fNy;
        const auto jp3 = (j + 3) % fNy;

        const auto km3 = (k - 3) % fNz;
        const auto km2 = (k - 2) % fNz;
        const auto km1 = (k - 1) % fNz;
        const auto kp1 = (k + 1) % fNz;
        const auto kp2 = (k + 2) % fNz;
        const auto kp3 = (k + 3) % fNz;

        static constexpr auto a = T{ 6 };
        static constexpr auto b = T{ 15 };
        static constexpr auto c = T{ 20 };

        // x
        (*sum_ptr)[idx][tidx] =
            (*f_ptr)[{ im3, juz, kuz }][tidx] - a * (*f_ptr)[{ im2, juz, kuz }][tidx]
            + b * (*f_ptr)[{ im1, juz, kuz }][tidx] - c * (*f_ptr)[{ iuz, juz, kuz }][tidx]
            + b * (*f_ptr)[{ ip1, juz, kuz }][tidx] - a * (*f_ptr)[{ ip2, juz, kuz }][tidx]
            + (*f_ptr)[{ ip3, juz, kuz }][tidx];

        // y
        (*sum_ptr)[idx][tidx] +=
            (*f_ptr)[{ iuz, jm3, kuz }][tidx] - a * (*f_ptr)[{ iuz, jm2, kuz }][tidx]
            + b * (*f_ptr)[{ iuz, jm1, kuz }][tidx] - c * (*f_ptr)[{ iuz, juz, kuz }][tidx]
            + b * (*f_ptr)[{ iuz, jp1, kuz }][tidx] - a * (*f_ptr)[{ iuz, jp2, kuz }][tidx]
            + (*f_ptr)[{ iuz, jp3, kuz }][tidx];

        // z
        (*sum_ptr)[idx][tidx] +=
            (*f_ptr)[{ iuz, juz, km3 }][tidx] - a * (*f_ptr)[{ iuz, juz, km2 }][tidx]
            + b * (*f_ptr)[{ iuz, juz, km1 }][tidx] - c * (*f_ptr)[{ iuz, juz, kuz }][tidx]
            + b * (*f_ptr)[{ iuz, juz, kp1 }][tidx] - a * (*f_ptr)[{ iuz, juz, kp2 }][tidx]
            + (*f_ptr)[{ iuz, juz, kp3 }][tidx];
    });

    tensor_buffer_queue.wait();
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
        auto* const lapse_derivative_sum_ptr = &lapse_derivative_sum;
        auto* const W_derivative_sum_ptr     = &W_derivative_sum;
        auto* const K_derivative_sum_ptr     = &K_derivative_sum;
        W.for_each_index([=, this](const auto idx) {
            lapse[idx][] += coeff * (*lapse_derivative_sum_ptr)[idx][];
            W[idx][] += coeff * (*W_derivative_sum_ptr)[idx][];
            K[idx][] += coeff * (*K_derivative_sum_ptr)[idx][];
        });

        tensor_buffer_queue.wait();
    }

    {
        const auto contraconf_christoffel_trace_derivative_sum =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U.contraconf_christoffel_trace_);

        auto* const contraconf_christoffel_trace_derivative_sum_ptr =
            &contraconf_christoffel_trace_derivative_sum;

        contraconf_christoffel_trace.for_each_index([=, this](const auto idx, const auto tidx) {
            contraconf_christoffel_trace[idx][tidx] +=
                coeff * (*contraconf_christoffel_trace_derivative_sum_ptr)[idx][tidx];
        });

        tensor_buffer_queue.wait();
    }

    {
        const auto coconf_metric_derivative_sum =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U.coconf_metric_);
        auto* const coconf_metric_derivative_sum_ptr = &coconf_metric_derivative_sum;

        coconf_metric.for_each_index([=, this](const auto idx, const auto tidx) {
            coconf_metric[idx][tidx] += coeff * (*coconf_metric_derivative_sum_ptr)[idx][tidx];
        });
    }
    tensor_buffer_queue.wait();
}

void
w2_bssn_uniform_grid::clamp_W(const real W) {
    W_.for_each_index([=, this](const auto idx) { W_[idx][] = sycl::max(W, W_[idx][]); });
    tensor_buffer_queue.wait();
}
