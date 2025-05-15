#include "w2_bssn_uniform_grid.hpp"

#include "ranges.hpp"

#include <execution>
#include <fstream>
#include <numeric>
#include <ranges>
#include <thread>
#include <vector>

#include <sycl/sycl.hpp>

std::shared_ptr<w2_bssn_uniform_grid>
w2_bssn_uniform_grid::euler_step(const std::shared_ptr<time_derivative_type>& dfdt_ptr,
                                 const real dt) const {
    auto f_ptr = allocate_shared_w2(*this);

    f_ptr->W_.for_each_index([dt, SPTR(f_ptr, dfdt_ptr)](const auto idx) {
        f_ptr->W_[idx][] += dt * dfdt_ptr->W[idx][];
        f_ptr->lapse_[idx][] += dt * dfdt_ptr->lapse[idx][];
        f_ptr->K_[idx][] += dt * dfdt_ptr->K[idx][];
    });

    f_ptr->contraconf_christoffel_trace_.for_each_index(
        [dt, SPTR(f_ptr, dfdt_ptr)](const auto idx, const auto tidx) {
            f_ptr->contraconf_christoffel_trace_[idx][tidx] +=
                dt * dfdt_ptr->contraconf_christoffel_trace[idx][tidx];
        });

    f_ptr->coconf_A_.for_each_index([dt, SPTR(f_ptr, dfdt_ptr)](const auto idx, const auto tidx) {
        f_ptr->coconf_A_[idx][tidx] += dt * dfdt_ptr->coconf_A[idx][tidx];
        f_ptr->coconf_metric_[idx][tidx] += dt * dfdt_ptr->coconf_metric[idx][tidx];
    });

    tensor_buffer_queue.wait();
    return f_ptr;
}

namespace finite_difference {

/// Calculates derivative of arbitrary tensor T_{abc...} -> T_{abc...,i}
///
/// Assumes that tensor buffer elements are at seperated by dx = 1 / Nx.
template<std::size_t rank, typename T, typename Allocator>
[[nodiscard]]
auto
periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
    const tensor_buffer<rank, 3uz, T, Allocator>& f) {
    auto sum_ptr = w2_bssn_uniform_grid::allocate_buffer<rank>(f.size());

    auto* const f_ptr = &f;
    // const auto [fNx, fNy, fNz] = f.size();
    const auto [fNx, _, _] = f.size();

    // Assume that x coordinates are 0, 1 / Nx, ..., (Nx - 1) / Nx.
    const auto dx = real{ 1 } / static_cast<real>(fNx);

    sum_ptr->for_each_index([=, SPTR(sum_ptr)](const auto idx, const auto tidx) {
        const auto iuz = idx[0];
        const auto juz = idx[1];
        const auto kuz = idx[2];

        const auto i = static_cast<std::ptrdiff_t>(iuz);
        // const auto j = static_cast<std::ptrdiff_t>(juz);
        // const auto k = static_cast<std::ptrdiff_t>(kuz);

        const auto im3 = (fNx + i - 3) % fNx;
        const auto im2 = (fNx + i - 2) % fNx;
        const auto im1 = (fNx + i - 1) % fNx;
        const auto ip1 = (fNx + i + 1) % fNx;
        const auto ip2 = (fNx + i + 2) % fNx;
        const auto ip3 = (fNx + i + 3) % fNx;

        // const auto jm3 = (fNy + j - 3) % fNy;
        // const auto jm2 = (fNy + j - 2) % fNy;
        // const auto jm1 = (fNy + j - 1) % fNy;
        // const auto jp1 = (fNy + j + 1) % fNy;
        // const auto jp2 = (fNy + j + 2) % fNy;
        // const auto jp3 = (fNy + j + 3) % fNy;

        // const auto km3 = (fNz + k - 3) % fNz;
        // const auto km2 = (fNz + k - 2) % fNz;
        // const auto km1 = (fNz + k - 1) % fNz;
        // const auto kp1 = (fNz + k + 1) % fNz;
        // const auto kp2 = (fNz + k + 2) % fNz;
        // const auto kp3 = (fNz + k + 3) % fNz;

        static constexpr auto a = T{ 6 };
        static constexpr auto b = T{ 15 };
        static constexpr auto c = T{ 20 };

        // x
        (*sum_ptr)[idx][tidx] =
            (*f_ptr)[{ im3, juz, kuz }][tidx] - a * (*f_ptr)[{ im2, juz, kuz }][tidx]
            + b * (*f_ptr)[{ im1, juz, kuz }][tidx] - c * (*f_ptr)[{ iuz, juz, kuz }][tidx]
            + b * (*f_ptr)[{ ip1, juz, kuz }][tidx] - a * (*f_ptr)[{ ip2, juz, kuz }][tidx]
            + (*f_ptr)[{ ip3, juz, kuz }][tidx];

        // Here we only divide by one dx as rest are cancelled by dx^5
        // from Kreiss-Oliger dissipation coefficient multiplying this sum.
        (*sum_ptr)[idx][tidx] /= dx;

        // y
        // (*sum_ptr)[idx][tidx] +=
        //     (*f_ptr)[{ iuz, jm3, kuz }][tidx] - a * (*f_ptr)[{ iuz, jm2, kuz }][tidx]
        //     + b * (*f_ptr)[{ iuz, jm1, kuz }][tidx] - c * (*f_ptr)[{ iuz, juz, kuz }][tidx]
        //     + b * (*f_ptr)[{ iuz, jp1, kuz }][tidx] - a * (*f_ptr)[{ iuz, jp2, kuz }][tidx]
        //     + (*f_ptr)[{ iuz, jp3, kuz }][tidx];

        // z
        // (*sum_ptr)[idx][tidx] +=
        //     (*f_ptr)[{ iuz, juz, km3 }][tidx] - a * (*f_ptr)[{ iuz, juz, km2 }][tidx]
        //     + b * (*f_ptr)[{ iuz, juz, km1 }][tidx] - c * (*f_ptr)[{ iuz, juz, kuz }][tidx]
        //     + b * (*f_ptr)[{ iuz, juz, kp1 }][tidx] - a * (*f_ptr)[{ iuz, juz, kp2 }][tidx]
        //     + (*f_ptr)[{ iuz, juz, kp3 }][tidx];
    });

    // We have to wait, because after return argument f might move before kernel is executed.
    // This is bad architecture, but too late to refactor whole thing.
    tensor_buffer_queue.wait();
    return sum_ptr;
}
} // namespace finite_difference

void
w2_bssn_uniform_grid::time_derivative_type::kreiss_oliger_6th_order(
    const std::shared_ptr<w2_bssn_uniform_grid>& U,
    const real epsilon) {

    const auto coeff = epsilon / real{ 64 };

    {
        const auto lapse_derivative_sum_ptr =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U->lapse_);
        const auto W_derivative_sum_ptr =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U->W_);
        const auto K_derivative_sum_ptr =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U->K_);
        W.for_each_index(
            [SPTR(lapse_derivative_sum_ptr, W_derivative_sum_ptr, K_derivative_sum_ptr),
             this,
             coeff](const auto idx) {
                lapse[idx][] += coeff * (*lapse_derivative_sum_ptr)[idx][];
                W[idx][] += coeff * (*W_derivative_sum_ptr)[idx][];
                K[idx][] += coeff * (*K_derivative_sum_ptr)[idx][];
            });
        tensor_buffer_queue.wait();
    }

    {
        const auto contraconf_christoffel_trace_derivative_sum_ptr =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U->contraconf_christoffel_trace_);

        contraconf_christoffel_trace.for_each_index(
            [SPTR(contraconf_christoffel_trace_derivative_sum_ptr), this, coeff](const auto idx,
                                                                                 const auto tidx) {
                contraconf_christoffel_trace[idx][tidx] +=
                    coeff * (*contraconf_christoffel_trace_derivative_sum_ptr)[idx][tidx];
            });
        tensor_buffer_queue.wait();
    }

    {
        const auto coconf_metric_derivative_sum_ptr =
            finite_difference::periodic_2th_order_central_6th_order_kreiss_oliger_derivative_sum(
                U->coconf_metric_);

        coconf_metric.for_each_index(
            [SPTR(coconf_metric_derivative_sum_ptr), this, coeff](const auto idx, const auto tidx) {
                coconf_metric[idx][tidx] += coeff * (*coconf_metric_derivative_sum_ptr)[idx][tidx];
            });
        tensor_buffer_queue.wait();
    }
}

void
w2_bssn_uniform_grid::clamp_W(const real W) {
    W_.for_each_index([W, this](const auto idx) { W_[idx][] = sycl::max(W, W_[idx][]); });
    tensor_buffer_queue.wait();
}

void
w2_bssn_uniform_grid::append_output(const real time,
                                    const constraints_type& constraints,
                                    const std::filesystem::path& output_dir_path) {
    auto t = std::jthread{ [time, constraints, output_dir_path, W = W_, g = coconf_metric_] {
        // Assumes grid size of N x 1 x 1!
        const auto N = constraints.hamiltonian.size().Nx;

        // Time:

        {
            auto time_file =
                std::fstream(output_dir_path / "time.txt", std::ios::out | std::ios::app);
            time_file << time << std::endl;
        }
        // Hamiltonian sum:

        {
            auto hsum_file = std::fstream(output_dir_path / "hamiltonian_sum.txt",
                                          std::ios::out | std::ios::app);
            auto Hsum      = real{ 0 };
            for (const auto i : rv::iota(0uz, N)) {
                Hsum +=
                    constraints.hamiltonian[i, 0uz, 0uz][] * constraints.hamiltonian[i, 0uz, 0uz][];
            }
            hsum_file << Hsum << std::endl;
        }

        // Momentum sum:

        {
            auto msum_file =
                std::fstream(output_dir_path / "momentum_sum.txt", std::ios::out | std::ios::app);

            auto Msum = real{ 0 };
            for (const auto n : rv::iota(0uz, N)) {
                for (const auto i : rv::iota(0uz, 3uz)) {
                    Msum +=
                        constraints.momentum[n, 0uz, 0uz][i] * constraints.momentum[n, 0uz, 0uz][i];
                }
            }
            msum_file << Msum << std::endl;
        }

        // g_00 grid:

        {
            auto g00_file =
                std::fstream(output_dir_path / "g00.txt", std::ios::out | std::ios::app);

            for (const auto n : rv::iota(0uz, N)) {
                const auto g00 = g[n, 0uz, 0uz][0, 0] * W[n, 0uz, 0uz][] * W[n, 0uz, 0uz][];
                g00_file << g00 << " ";
            }
            g00_file << std::endl;
        }
    } };

    t.detach();
}
