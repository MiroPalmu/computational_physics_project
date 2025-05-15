#include "w2_bssn_uniform_grid.hpp"

#include <experimental/mdspan>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <functional>
#include <numbers>
#include <print>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "idg/einsum.hpp"
#include "mdspan_utils.hpp"

using namespace idg::literals;

namespace finite_difference {

/// Calculates derivative of arbitrary tensor T_{abc...} -> T_{abc...,i}
///
/// Assumes that tensor buffer elements are at seperated by dx = 1 / Nx.
template<std::size_t rank, typename T, typename Allocator>
[[nodiscard]]
auto
periodic_4th_order_central_1st_derivative(const tensor_buffer<rank, 3uz, T, Allocator>& f) {
    auto derivatives_ptr = w2_bssn_uniform_grid::allocate_buffer<rank + 1uz>(f.size());

    auto* const f_ptr = &f;
    // const auto [fNx, fNy, fNz] = f.size();
    const auto [fNx, _, _] = f.size();

    // Assume that x coordinates are 0, 1 / Nx, ..., (Nx - 1) / Nx.
    const auto dx = real{ 1 } / static_cast<real>(fNx);

    f.for_each_index([=, SPTR(derivatives_ptr)](const auto idx, const auto tidx) {
        const auto iuz = idx[0];
        const auto juz = idx[1];
        const auto kuz = idx[2];

        const auto i = static_cast<std::ptrdiff_t>(iuz);
        // const auto j = static_cast<std::ptrdiff_t>(juz);
        // const auto k = static_cast<std::ptrdiff_t>(kuz);

        const auto im2 = (fNx + i - 2) % fNx;
        const auto im1 = (fNx + i - 1) % fNx;
        const auto ip1 = (fNx + i + 1) % fNx;
        const auto ip2 = (fNx + i + 2) % fNx;

        // const auto jm2 = (fNy + j - 2) % fNy;
        // const auto jm1 = (fNy + j - 1) % fNy;
        // const auto jp1 = (fNy + j + 1) % fNy;
        // const auto jp2 = (fNy + j + 2) % fNy;

        // const auto km2 = (fNz + k - 2) % fNz;
        // const auto km1 = (fNz + k - 1) % fNz;
        // const auto kp1 = (fNz + k + 1) % fNz;
        // const auto kp2 = (fNz + k + 2) % fNz;

        using derivative_tidx_type = std::array<std::size_t, rank + 1>;

        auto xtidx = derivative_tidx_type{};
        auto ytidx = derivative_tidx_type{};
        auto ztidx = derivative_tidx_type{};
        for (auto n = 0uz; n < tidx.size(); ++n) {
            xtidx[n] = tidx[n];
            ytidx[n] = tidx[n];
            ztidx[n] = tidx[n];
        }
        xtidx.back() = 0uz;
        ytidx.back() = 1uz;
        ztidx.back() = 2uz;

        static constexpr auto a = T{ 1 } / T{ 12 };
        static constexpr auto b = T{ 2 } / T{ 3 };

        (*derivatives_ptr)[idx][xtidx] =
            a * (*f_ptr)[{ im2, juz, kuz }][tidx] - b * (*f_ptr)[{ im1, juz, kuz }][tidx]
            + b * (*f_ptr)[{ ip1, juz, kuz }][tidx] - a * (*f_ptr)[{ ip2, juz, kuz }][tidx];

        (*derivatives_ptr)[idx][xtidx] /= dx;

        (*derivatives_ptr)[idx][ytidx] = real{ 0 };
        //     a * (*f_ptr)[{ iuz, jm2, kuz }][tidx] - b * (*f_ptr)[{ iuz, jm1, kuz }][tidx]
        //     + b * (*f_ptr)[{ iuz, jp1, kuz }][tidx] - a * (*f_ptr)[{ iuz, jp2, kuz }][tidx];

        (*derivatives_ptr)[idx][ztidx] = real{ 0 };
        //     a * (*f_ptr)[{ iuz, juz, km2 }][tidx] - b * (*f_ptr)[{ iuz, juz, km1 }][tidx]
        //     + b * (*f_ptr)[{ iuz, juz, kp1 }][tidx] - a * (*f_ptr)[{ iuz, juz, kp2 }][tidx];
    });

    // We have to wait, because after return argument f might move before kernel is executed.
    // This is bad architecture, but too late to refactor whole thing.
    tensor_buffer_queue.wait();
    return derivatives_ptr;
}

/// Covariant christoffel symbols, i.e. Christoffel symbols of the 1st kind.
///
/// Also, as a side product returns derivatives of metric g_{ij}, i.e. g_{ij,k}.
template<typename T, typename Allocator>
[[nodiscard]]
std::pair<std::shared_ptr<w2_bssn_uniform_grid::buffer3>,
          std::shared_ptr<w2_bssn_uniform_grid::buffer3>>
co_christoffel_symbols(const tensor_buffer<2, 3, T, Allocator>& co_metric) {
    auto dg_ptr = periodic_4th_order_central_1st_derivative(co_metric);
    auto c_ptr  = w2_bssn_uniform_grid::allocate_buffer<3>(co_metric.size());

    using tidx_type = std::array<std::size_t, 3>;

    c_ptr->for_each_index([SPTR(c_ptr, dg_ptr)](const auto idx, const tidx_type tidx) {
        const auto cab = tidx;
        const auto cba = tidx_type{ cab[0], cab[2], cab[1] };
        const auto abc = tidx_type{ cab[1], cab[2], cab[0] };

        (*c_ptr)[idx][cab] =
            (T{ 1 } / T{ 2 }) * ((*dg_ptr)[idx][cab] + (*dg_ptr)[idx][cba] - (*dg_ptr)[idx][abc]);
    });

    // We have to wait, because after return arguments might move before kernel is executed.
    // This is bad architecture, but too late to refactor whole thing.
    tensor_buffer_queue.wait();
    return { std::move(c_ptr), std::move(dg_ptr) };
}

} // namespace finite_difference

[[nodiscard]]
w2_bssn_uniform_grid::w2_bssn_uniform_grid(const grid_size gs, minkowski_spacetime_tag)
    : grid_size_(gs),
      W_(gs),
      lapse_(gs),
      coconf_metric_(gs),
      K_(gs),
      coconf_A_(gs),
      contraconf_christoffel_trace_(gs) {
    coconf_metric_.for_each_index([this](const auto idx, const auto tidx) {
        coconf_metric_[idx][tidx] = static_cast<real>(tidx[0] == tidx[1]);
    });

    lapse_.for_each_index([this](const auto idx) { lapse_[idx][] = 1; });

    K_.for_each_index([this](const auto idx) { K_[idx][] = 0; });
    W_.for_each_index([this](const auto idx) { W_[idx][] = 1; });

    coconf_A_.for_each_index([this](const auto idx, const auto tidx) { coconf_A_[idx][tidx] = 0; });

    contraconf_christoffel_trace_.for_each_index(
        [this](const auto idx, const auto tidx) { contraconf_christoffel_trace_[idx][tidx] = 0; });

    // We have to wait, because after return this might move before kernel is executed.
    // This is bad architecture, but too late to refactor whole thing.
    tensor_buffer_queue.wait();
}

[[nodiscard]]
std::pair<std::shared_ptr<w2_bssn_uniform_grid::buffer0>,
          std::shared_ptr<w2_bssn_uniform_grid::buffer2>>
det_n_inv3D(const w2_bssn_uniform_grid::buffer2& matrix) {
    auto det_ptr = w2_bssn_uniform_grid::allocate_buffer<0>(matrix.size());
    auto inv_ptr = w2_bssn_uniform_grid::allocate_buffer<2>(matrix.size());

    auto* const matrix_ptr = &matrix;

    det_ptr->for_each_index([=, SPTR(det_ptr, inv_ptr)](const auto idx) {
        const auto a = (*matrix_ptr)[idx][0, 0];
        const auto b = (*matrix_ptr)[idx][0, 1];
        const auto c = (*matrix_ptr)[idx][0, 2];
        const auto d = (*matrix_ptr)[idx][1, 0];
        const auto e = (*matrix_ptr)[idx][1, 1];
        const auto f = (*matrix_ptr)[idx][1, 2];
        const auto g = (*matrix_ptr)[idx][2, 0];
        const auto h = (*matrix_ptr)[idx][2, 1];
        const auto i = (*matrix_ptr)[idx][2, 2];

        const auto A = (e * i - f * h);
        const auto B = -(d * i - f * g);
        const auto C = (d * h - e * g);

        const auto D = -(b * i - c * h);
        const auto E = (a * i - c * g);
        const auto F = -(a * h - b * g);

        const auto G = (b * f - c * e);
        const auto H = -(a * f - c * d);
        const auto I = (a * e - d * d);

        (*det_ptr)[idx][] = a * A + b * B + c * C;

        (*inv_ptr)[idx][0, 0] = A / (*det_ptr)[idx][];
        (*inv_ptr)[idx][1, 0] = B / (*det_ptr)[idx][];
        (*inv_ptr)[idx][2, 0] = C / (*det_ptr)[idx][];
        (*inv_ptr)[idx][0, 1] = D / (*det_ptr)[idx][];
        (*inv_ptr)[idx][1, 1] = E / (*det_ptr)[idx][];
        (*inv_ptr)[idx][2, 1] = F / (*det_ptr)[idx][];
        (*inv_ptr)[idx][0, 2] = G / (*det_ptr)[idx][];
        (*inv_ptr)[idx][1, 2] = H / (*det_ptr)[idx][];
        (*inv_ptr)[idx][2, 2] = I / (*det_ptr)[idx][];
    });

    // We have to wait, because after return arguments might move before kernel is executed.
    // This is bad architecture, but too late to refactor whole thing.
    tensor_buffer_queue.wait();
    return { std::move(det_ptr), std::move(inv_ptr) };
}

w2_bssn_uniform_grid::time_derivative_type::time_derivative_type(const grid_size gs)
    : lapse(gs),
      W(gs),
      coconf_metric(gs),
      K(gs),
      coconf_A(gs),
      contraconf_christoffel_trace(gs) {}

w2_bssn_uniform_grid::constraints_type::constraints_type(const grid_size gs)
    : momentum(gs),
      hamiltonian(gs) {}

void
w2_bssn_uniform_grid::enforce_algebraic_constraints() {
    auto [det_ptr, contraconf_metric_ptr] = det_n_inv3D(coconf_metric_);

    coconf_metric_.for_each_index([SPTR(det_ptr), this](const auto idx) {
        const auto det3 = sycl::pow((*det_ptr)[idx][], real{ -1 } / real{ 3 });

        coconf_metric_[idx][0, 0] *= det3;
        coconf_metric_[idx][0, 1] *= det3;
        coconf_metric_[idx][0, 2] *= det3;
        coconf_metric_[idx][1, 0] *= det3;
        coconf_metric_[idx][1, 1] *= det3;
        coconf_metric_[idx][1, 2] *= det3;
        coconf_metric_[idx][2, 0] *= det3;
        coconf_metric_[idx][2, 1] *= det3;
        coconf_metric_[idx][2, 2] *= det3;
    });

    auto trace_remover_ptr = allocate_buffer<2>(grid_size_);

    // It does not matter if trace remover is calculated with conformal or
    // nor conformal metric. At least with W^2, they cancel out:
    // g_{ij}g^{nm} = \tilde{g}_{ij}\tilde{g}^{nm}
    trace_remover_ptr->for_each_index(
        [SPTR(trace_remover_ptr, contraconf_metric_ptr), this](const auto idx) {
            static constexpr auto third = constant_geometric_mdspan<0, 3, real{ 1 } / real{ 3 }>();
            u8",ij,nm,nm"_einsum((*trace_remover_ptr)[idx],
                                 third,
                                 coconf_metric_[idx],
                                 (*contraconf_metric_ptr)[idx],
                                 coconf_A_[idx]);
        });

    coconf_A_.for_each_index([SPTR(trace_remover_ptr), this](const auto idx, const auto tidx) {
        coconf_A_[idx][tidx] = coconf_A_[idx][tidx] - (*trace_remover_ptr)[idx][tidx];
    });

    // We have to wait, because after return this might move before kernel is executed.
    // This is bad architecture, but too late to refactor whole thing.
    tensor_buffer_queue.wait();
}

w2_bssn_uniform_grid::pre_calculations_type
w2_bssn_uniform_grid::pre_calculations() const {
    auto dfdt_ptr =
        std::allocate_shared<time_derivative_type>(allocator{ tensor_buffer_queue }, grid_size_);

    auto constraints_ptr =
        std::allocate_shared<constraints_type>(allocator{ tensor_buffer_queue }, grid_size_);

    dfdt_ptr->lapse.for_each_index([SPTR(dfdt_ptr), this](const auto idx) {
        dfdt_ptr->lapse[idx][] = -lapse_[idx][] * lapse_[idx][] * K_[idx][];
    });

    dfdt_ptr->W.for_each_index([SPTR(dfdt_ptr), this](const auto idx) {
        dfdt_ptr->W[idx][] = W_[idx][] * lapse_[idx][] * K_[idx][] / real{ 3 };
    });

    dfdt_ptr->coconf_metric.for_each_index([SPTR(dfdt_ptr), this](const auto idx) {
        static constexpr auto minus2 = constant_geometric_mdspan<0, 3, real{ -2 }>();
        u8",,ij"_einsum(dfdt_ptr->coconf_metric[idx], minus2, lapse_[idx], coconf_A_[idx]);
    });

    [[maybe_unused]] const auto [_, contraconf_metric_ptr] = det_n_inv3D(coconf_metric_);

    const auto [coconf_christoffels_ptr, coconf_metric_derivative_ptr] =
        finite_difference::co_christoffel_symbols(coconf_metric_);

    const auto W_derivative_ptr = finite_difference::periodic_4th_order_central_1st_derivative(W_);
    const auto lapse_derivative_ptr =
        finite_difference::periodic_4th_order_central_1st_derivative(lapse_);

    const auto co_W2DiDj_lapse_ptr = std::invoke([&] {
        const auto coconf_DiDj_lapse_ptr = std::invoke([&] {
            auto lapse_2nd_derivative_ptr =
                finite_difference::periodic_4th_order_central_1st_derivative(*lapse_derivative_ptr);

            auto christoffel_term_ptr = allocate_buffer<2>(grid_size_);
            christoffel_term_ptr->for_each_index([SPTR(christoffel_term_ptr,
                                                       contraconf_metric_ptr,
                                                       coconf_christoffels_ptr,
                                                       lapse_derivative_ptr)](const auto idx) {
                u8"ij,jab,i"_einsum((*christoffel_term_ptr)[idx],
                                    (*contraconf_metric_ptr)[idx],
                                    (*coconf_christoffels_ptr)[idx],
                                    (*lapse_derivative_ptr)[idx]);
            });

            lapse_2nd_derivative_ptr->for_each_index(
                [SPTR(lapse_2nd_derivative_ptr, christoffel_term_ptr)](const auto idx,
                                                                       const auto tidx) {
                    (*lapse_2nd_derivative_ptr)[idx][tidx] -= (*christoffel_term_ptr)[idx][tidx];
                });

            tensor_buffer_queue.wait();
            return lapse_2nd_derivative_ptr;
        });

        const auto dWdlapse_ptr = std::invoke([&] {
            auto temp_ptr = allocate_buffer<2>(grid_size_);
            temp_ptr->for_each_index(
                [SPTR(temp_ptr, W_derivative_ptr, lapse_derivative_ptr)](const auto idx) {
                    u8"i,j"_einsum((*temp_ptr)[idx],
                                   (*W_derivative_ptr)[idx],
                                   (*lapse_derivative_ptr)[idx]);
                });
            tensor_buffer_queue.wait();
            return temp_ptr;
        });

        const auto last_term_ptr = std::invoke([&] {
            auto temp_ptr = allocate_buffer<2>(grid_size_);
            temp_ptr->for_each_index(
                [SPTR(temp_ptr, contraconf_metric_ptr, dWdlapse_ptr), this](const auto idx) {
                    u8"ij,nm,nm"_einsum((*temp_ptr)[idx],
                                        coconf_metric_[idx],
                                        (*contraconf_metric_ptr)[idx],
                                        (*dWdlapse_ptr)[idx]);
                });
            tensor_buffer_queue.wait();
            return temp_ptr;
        });

        auto temp_ptr = allocate_buffer<2>(grid_size_);
        temp_ptr->for_each_index(
            [SPTR(temp_ptr, coconf_DiDj_lapse_ptr, dWdlapse_ptr, last_term_ptr),
             this](const auto idx, const auto tidx) {
                (*temp_ptr)[idx][tidx] = W_[idx][] * (*coconf_DiDj_lapse_ptr)[idx][tidx]
                                         + (*dWdlapse_ptr)[idx][tidx]
                                         + (*dWdlapse_ptr)[idx][std::array{ tidx[1], tidx[0] }]
                                         - (*last_term_ptr)[idx][tidx];
                (*temp_ptr)[idx][tidx] *= W_[idx][];
            });
        tensor_buffer_queue.wait();

        return temp_ptr;
    });

    const auto contraconf_A_ptr      = std::invoke([&] {
        auto temp_ptr = allocate_buffer<2>(grid_size_);
        temp_ptr->for_each_index([SPTR(temp_ptr, contraconf_metric_ptr), this](const auto idx) {
            u8"ia,jb,ab"_einsum((*temp_ptr)[idx],
                                (*contraconf_metric_ptr)[idx],
                                (*contraconf_metric_ptr)[idx],
                                coconf_A_[idx]);
        });
        tensor_buffer_queue.wait();

        return temp_ptr;
    });
    static constexpr auto minus_half = constant_geometric_mdspan<0, 3, real{ -1 } / real{ 2 }>();

    { // calculate dfdt_ptr->K
        const auto term1_ptr = std::invoke([&] {
            auto temp_ptr = allocate_buffer<0>(grid_size_);
            temp_ptr->for_each_index(
                [SPTR(temp_ptr, contraconf_metric_ptr, co_W2DiDj_lapse_ptr)](const auto idx) {
                    u8"nm,nm"_einsum((*temp_ptr)[idx],
                                     (*contraconf_metric_ptr)[idx],
                                     (*co_W2DiDj_lapse_ptr)[idx]);
                });
            tensor_buffer_queue.wait();

            return temp_ptr;
        });

        const auto term2_ptr = std::invoke([&] {
            auto temp_ptr = allocate_buffer<0>(grid_size_);
            temp_ptr->for_each_index([SPTR(temp_ptr, contraconf_A_ptr), this](const auto idx) {
                u8",nm,nm"_einsum((*temp_ptr)[idx],
                                  lapse_[idx],
                                  (*contraconf_A_ptr)[idx],
                                  coconf_A_[idx]);
            });
            tensor_buffer_queue.wait();
            return temp_ptr;
        });

        dfdt_ptr->K.for_each_index([SPTR(dfdt_ptr, term1_ptr, term2_ptr), this](const auto idx) {
            const auto term3   = lapse_[idx][] * K_[idx][] * K_[idx][] / real{ 3 };
            dfdt_ptr->K[idx][] = -(*term1_ptr)[idx][] + (*term2_ptr)[idx][] + term3;
        });
        tensor_buffer_queue.wait();
    }

    static constexpr auto two = constant_geometric_mdspan<0, 3, real{ 2 }>();
    { // calculate dfdt_ptr->coconf_A
        auto term1_ptr = allocate_buffer<2>(grid_size_);
        term1_ptr->for_each_index([SPTR(term1_ptr), this](const auto idx) {
            u8",,ij"_einsum((*term1_ptr)[idx], lapse_[idx], K_[idx], coconf_A_[idx]);
        });

        { // term 2
            auto term2_ptr = allocate_buffer<2>(grid_size_);
            term2_ptr->for_each_index(
                [SPTR(term2_ptr, contraconf_metric_ptr), this](const auto idx) {
                    u8",,im,mn,nj"_einsum((*term2_ptr)[idx],
                                          two,
                                          lapse_[idx],
                                          coconf_A_[idx],
                                          (*contraconf_metric_ptr)[idx],
                                          coconf_A_[idx]);
                });

            term1_ptr->for_each_index(
                [SPTR(term1_ptr, term2_ptr)](const auto idx, const auto tidx) {
                    (*term1_ptr)[idx][tidx] -= (*term2_ptr)[idx][tidx];
                });
            tensor_buffer_queue.wait();
        }

        // trace free term
        const auto TFterm_ptr = std::invoke([&] {
            const auto coconf_R_ptr = std::invoke([&] {
                auto term1_ptr = std::invoke([&] {
                    const auto coconf_metric_2nd_derivative_ptr =
                        finite_difference::periodic_4th_order_central_1st_derivative(
                            *coconf_metric_derivative_ptr);

                    auto temp_ptr = allocate_buffer<2>(grid_size_);
                    temp_ptr->for_each_index(
                        [SPTR(temp_ptr, contraconf_metric_ptr, coconf_metric_2nd_derivative_ptr)](
                            const auto idx) {
                            u8",nm,ijnm"_einsum((*temp_ptr)[idx],
                                                minus_half,
                                                (*contraconf_metric_ptr)[idx],
                                                (*coconf_metric_2nd_derivative_ptr)[idx]);
                        });
                    tensor_buffer_queue.wait();

                    return temp_ptr;
                });

                { // terms 2 and 3
                    auto term2_ptr = allocate_buffer<2>(grid_size_);
                    auto term3_ptr = allocate_buffer<2>(grid_size_);

                    const auto contraconf_christoffel_trace_derivative_ptr =
                        finite_difference::periodic_4th_order_central_1st_derivative(
                            contraconf_christoffel_trace_);

                    term2_ptr->for_each_index([SPTR(term2_ptr,
                                                    contraconf_christoffel_trace_derivative_ptr),
                                               this](const auto idx) {
                        u8"mi,mj -> ij"_einsum((*term2_ptr)[idx],
                                               coconf_metric_[idx],
                                               (*contraconf_christoffel_trace_derivative_ptr)[idx]);
                    });

                    term3_ptr->for_each_index([SPTR(term3_ptr,
                                                    contraconf_christoffel_trace_derivative_ptr),
                                               this](const auto idx) {
                        u8"mi,mj -> ji"_einsum((*term3_ptr)[idx],
                                               coconf_metric_[idx],
                                               (*contraconf_christoffel_trace_derivative_ptr)[idx]);
                    });

                    term1_ptr->for_each_index(
                        [SPTR(term1_ptr, term2_ptr, term3_ptr)](const auto idx, const auto tidx) {
                            (*term1_ptr)[idx][tidx] +=
                                (real{ 1 } / real{ 2 })
                                * ((*term2_ptr)[idx][tidx] + (*term3_ptr)[idx][tidx]);
                        });
                    tensor_buffer_queue.wait();
                }

                { // terms 4 and 5
                    auto term4_ptr = allocate_buffer<2>(grid_size_);
                    auto term5_ptr = allocate_buffer<2>(grid_size_);

                    term4_ptr->for_each_index(
                        [SPTR(term4_ptr, contraconf_metric_ptr, coconf_christoffels_ptr)](
                            const auto idx) {
                            u8"ab,cm,cab,ijm -> ij"_einsum((*term4_ptr)[idx],
                                                           (*contraconf_metric_ptr)[idx],
                                                           (*contraconf_metric_ptr)[idx],
                                                           (*coconf_christoffels_ptr)[idx],
                                                           (*coconf_christoffels_ptr)[idx]);
                        });

                    term5_ptr->for_each_index(
                        [SPTR(term5_ptr, contraconf_metric_ptr, coconf_christoffels_ptr)](
                            const auto idx) {
                            u8"ab,cm,cab,ijm -> ji"_einsum((*term5_ptr)[idx],
                                                           (*contraconf_metric_ptr)[idx],
                                                           (*contraconf_metric_ptr)[idx],
                                                           (*coconf_christoffels_ptr)[idx],
                                                           (*coconf_christoffels_ptr)[idx]);
                        });

                    term1_ptr->for_each_index(
                        [SPTR(term1_ptr, term4_ptr, term5_ptr)](const auto idx, const auto tidx) {
                            (*term1_ptr)[idx][tidx] +=
                                (real{ 1 } / real{ 2 })
                                * ((*term4_ptr)[idx][tidx] + (*term5_ptr)[idx][tidx]);
                        });
                    tensor_buffer_queue.wait();
                }

                { // terms 6 and 7
                    auto term6_ptr = allocate_buffer<2>(grid_size_);
                    auto term7_ptr = allocate_buffer<2>(grid_size_);

                    term6_ptr->for_each_index(
                        [SPTR(term6_ptr, contraconf_metric_ptr, coconf_christoffels_ptr)](
                            const auto idx) {
                            u8"ab,nm,ani,jbm -> ij"_einsum((*term6_ptr)[idx],
                                                           (*contraconf_metric_ptr)[idx],
                                                           (*contraconf_metric_ptr)[idx],
                                                           (*coconf_christoffels_ptr)[idx],
                                                           (*coconf_christoffels_ptr)[idx]);
                        });

                    term7_ptr->for_each_index(
                        [SPTR(term7_ptr, contraconf_metric_ptr, coconf_christoffels_ptr)](
                            const auto idx) {
                            u8"ab,nm,ani,jbm -> ji"_einsum((*term7_ptr)[idx],
                                                           (*contraconf_metric_ptr)[idx],
                                                           (*contraconf_metric_ptr)[idx],
                                                           (*coconf_christoffels_ptr)[idx],
                                                           (*coconf_christoffels_ptr)[idx]);
                        });

                    term1_ptr->for_each_index(
                        [SPTR(term1_ptr, term6_ptr, term7_ptr)](const auto idx, const auto tidx) {
                            (*term1_ptr)[idx][tidx] +=
                                (*term6_ptr)[idx][tidx] + (*term7_ptr)[idx][tidx];
                        });
                    tensor_buffer_queue.wait();
                }

                { // term 8
                    auto term8_ptr = allocate_buffer<2>(grid_size_);

                    term8_ptr->for_each_index(
                        [SPTR(term8_ptr, contraconf_metric_ptr, coconf_christoffels_ptr)](
                            const auto idx) {
                            u8"ab,nm,ain,bjm -> ij"_einsum((*term8_ptr)[idx],
                                                           (*contraconf_metric_ptr)[idx],
                                                           (*contraconf_metric_ptr)[idx],
                                                           (*coconf_christoffels_ptr)[idx],
                                                           (*coconf_christoffels_ptr)[idx]);
                        });

                    term1_ptr->for_each_index(
                        [SPTR(term1_ptr, term8_ptr)](const auto idx, const auto tidx) {
                            (*term1_ptr)[idx][tidx] += (*term8_ptr)[idx][tidx];
                        });
                    tensor_buffer_queue.wait();
                }

                return term1_ptr;
            });

            const auto coconf_W2Rw_ptr = std::invoke([&] {
                const auto coconf_DiDj_W_ptr = std::invoke([&] {
                    auto christoffel_term_ptr = allocate_buffer<2>(grid_size_);

                    christoffel_term_ptr->for_each_index([SPTR(christoffel_term_ptr,
                                                               contraconf_metric_ptr,
                                                               coconf_christoffels_ptr,
                                                               W_derivative_ptr)](const auto idx) {
                        u8"ij,jab,i"_einsum((*christoffel_term_ptr)[idx],
                                            (*contraconf_metric_ptr)[idx],
                                            (*coconf_christoffels_ptr)[idx],
                                            (*W_derivative_ptr)[idx]);
                    });

                    auto W_2nd_derivative_ptr =
                        finite_difference::periodic_4th_order_central_1st_derivative(
                            *W_derivative_ptr);

                    W_2nd_derivative_ptr->for_each_index(
                        [SPTR(W_2nd_derivative_ptr, christoffel_term_ptr)](const auto idx,
                                                                           const auto tidx) {
                            (*W_2nd_derivative_ptr)[idx][tidx] -=
                                (*christoffel_term_ptr)[idx][tidx];
                        });
                    tensor_buffer_queue.wait();

                    return W_2nd_derivative_ptr;
                });

                auto term1_ptr = allocate_buffer<2>(*coconf_DiDj_W_ptr);

                { // term 2
                    auto term2_ptr = allocate_buffer<2>(grid_size_);

                    term2_ptr->for_each_index(
                        [SPTR(term2_ptr, contraconf_metric_ptr, coconf_DiDj_W_ptr),
                         this](const auto idx) {
                            u8"ij,nm,nm"_einsum((*term2_ptr)[idx],
                                                coconf_metric_[idx],
                                                (*contraconf_metric_ptr)[idx],
                                                (*coconf_DiDj_W_ptr)[idx]);
                        });

                    term1_ptr->for_each_index(
                        [SPTR(term1_ptr, term2_ptr), this](const auto idx, const auto tidx) {
                            (*term1_ptr)[idx][tidx] =
                                W_[idx][] * ((*term1_ptr)[idx][tidx] - (*term2_ptr)[idx][tidx]);
                        });
                    tensor_buffer_queue.wait();
                }

                { // term 3
                    auto term3_ptr = allocate_buffer<2>(grid_size_);
                    term3_ptr->for_each_index(
                        [SPTR(term3_ptr, contraconf_metric_ptr, W_derivative_ptr),
                         this](const auto idx) {
                            u8",ij,mn,m,n"_einsum((*term3_ptr)[idx],
                                                  two,
                                                  coconf_metric_[idx],
                                                  (*contraconf_metric_ptr)[idx],
                                                  (*W_derivative_ptr)[idx],
                                                  (*W_derivative_ptr)[idx]);
                        });

                    term1_ptr->for_each_index(
                        [SPTR(term1_ptr, term3_ptr)](const auto idx, const auto tidx) {
                            (*term1_ptr)[idx][tidx] -= (*term3_ptr)[idx][tidx];
                        });
                    tensor_buffer_queue.wait();
                }

                return term1_ptr;
            });

            auto lapseW2R_ptr = allocate_buffer<2>(grid_size_);

            lapseW2R_ptr->for_each_index([SPTR(lapseW2R_ptr, coconf_W2Rw_ptr, coconf_R_ptr),
                                          this](const auto idx, const auto tidx) {
                (*lapseW2R_ptr)[idx][tidx] =
                    lapse_[idx][]
                    * ((*coconf_W2Rw_ptr)[idx][tidx]
                       + W_[idx][] * W_[idx][] * (*coconf_R_ptr)[idx][tidx]);
            });

            lapseW2R_ptr->for_each_index(
                [SPTR(lapseW2R_ptr, co_W2DiDj_lapse_ptr)](const auto idx, const auto tidx) {
                    (*lapseW2R_ptr)[idx][tidx] -= (*co_W2DiDj_lapse_ptr)[idx][tidx];
                });
            tensor_buffer_queue.wait();

            return lapseW2R_ptr;
        });

        const auto trace_remover_ptr = std::invoke([&] {
            auto temp_ptr = allocate_buffer<2>(grid_size_);
            // It does not matter if trace remover is calculated with conformal or
            // nor conformal metric. At least with W^2, they cancel out:
            // g_{ij}g^{nm} = \tilde{g}_{ij}\tilde{g}^{nm}
            temp_ptr->for_each_index(
                [SPTR(temp_ptr, contraconf_metric_ptr, TFterm_ptr), this](const auto idx) {
                    static constexpr auto third =
                        constant_geometric_mdspan<0, 3, real{ 1 } / real{ 3 }>();
                    u8",ij,nm,nm"_einsum((*temp_ptr)[idx],
                                         third,
                                         coconf_metric_[idx],
                                         (*contraconf_metric_ptr)[idx],
                                         (*TFterm_ptr)[idx]);
                });

            tensor_buffer_queue.wait();
            return temp_ptr;
        });

        dfdt_ptr->coconf_A.for_each_index(
            [SPTR(dfdt_ptr, term1_ptr, TFterm_ptr, trace_remover_ptr)](const auto idx,
                                                                       const auto tidx) {
                dfdt_ptr->coconf_A[idx][tidx] = (*term1_ptr)[idx][tidx] + (*TFterm_ptr)[idx][tidx]
                                                - (*trace_remover_ptr)[idx][tidx];
            });
        tensor_buffer_queue.wait();
    }

    const auto K_derivative_ptr = finite_difference::periodic_4th_order_central_1st_derivative(K_);

    { // calculate dfdt_ptr->contraconf_christoffel_trace
        auto term1_ptr = std::invoke([&] {
            auto tempA_ptr = allocate_buffer<1>(*W_derivative_ptr);
            tempA_ptr->for_each_index(
                [SPTR(tempA_ptr, lapse_derivative_ptr), this](const auto idx, const auto tidx) {
                    (*tempA_ptr)[idx][tidx] *= real{ -6 } * lapse_[idx][] / W_[idx][];
                    (*tempA_ptr)[idx][tidx] += real{ -2 } * (*lapse_derivative_ptr)[idx][tidx];
                });

            auto tempB_ptr = allocate_buffer<1>(grid_size_);
            tempB_ptr->for_each_index(
                [SPTR(tempB_ptr, contraconf_A_ptr, tempA_ptr)](const auto idx) {
                    u8"im,m"_einsum((*tempB_ptr)[idx], (*contraconf_A_ptr)[idx], (*tempA_ptr)[idx]);
                });
            tensor_buffer_queue.wait();
            return tempB_ptr;
        });

        { // term 2
            auto term2_ptr = allocate_buffer<1>(grid_size_);
            term2_ptr->for_each_index(
                [SPTR(term2_ptr, contraconf_metric_ptr, coconf_christoffels_ptr, contraconf_A_ptr),
                 this](const auto idx) {
                    u8",,ia,abc,bc"_einsum((*term2_ptr)[idx],
                                           two,
                                           lapse_[idx],
                                           (*contraconf_metric_ptr)[idx],
                                           (*coconf_christoffels_ptr)[idx],
                                           (*contraconf_A_ptr)[idx]);
                });

            term1_ptr->for_each_index(
                [SPTR(term1_ptr, term2_ptr)](const auto idx, const auto tidx) {
                    (*term1_ptr)[idx][tidx] += (*term2_ptr)[idx][tidx];
                });
            tensor_buffer_queue.wait();
        }

        { // term 3

            auto term3_ptr = allocate_buffer<1>(grid_size_);
            term3_ptr->for_each_index(
                [SPTR(term3_ptr, contraconf_metric_ptr, K_derivative_ptr)](const auto idx) {
                    u8"im,m"_einsum((*term3_ptr)[idx],
                                    (*contraconf_metric_ptr)[idx],
                                    (*K_derivative_ptr)[idx]);
                });

            dfdt_ptr->contraconf_christoffel_trace.for_each_index(
                [SPTR(dfdt_ptr, term1_ptr, term3_ptr), this](const auto idx, const auto tidx) {
                    dfdt_ptr->contraconf_christoffel_trace[idx][tidx] =
                        (*term1_ptr)[idx][tidx]
                        - (real{ 4 } * lapse_[idx][] / real{ 3 }) * (*term3_ptr)[idx][tidx];
                });
            tensor_buffer_queue.wait();
        }
        tensor_buffer_queue.wait();
    }

    { // momentum constraint
        const auto contracoconf_A_ptr = std::invoke([&] {
            auto temp_ptr = allocate_buffer<2>(grid_size_);
            temp_ptr->for_each_index([SPTR(temp_ptr, contraconf_metric_ptr), this](const auto idx) {
                u8"ij,jk"_einsum((*temp_ptr)[idx], (*contraconf_metric_ptr)[idx], coconf_A_[idx]);
            });

            tensor_buffer_queue.wait();
            return temp_ptr;
        });

        auto term1_ptr = std::invoke([&] {
            const auto Ad_ptr =
                finite_difference::periodic_4th_order_central_1st_derivative(*contracoconf_A_ptr);

            auto temp_ptr = allocate_buffer<1>(grid_size_);

            temp_ptr->for_each_index([SPTR(temp_ptr, Ad_ptr)](const auto idx) {
                u8"jij"_einsum((*temp_ptr)[idx], (*Ad_ptr)[idx]);
            });

            tensor_buffer_queue.wait();
            return temp_ptr;
        });

        { // term 2
            const auto coconf_A_derivative_ptr =
                finite_difference::periodic_4th_order_central_1st_derivative(coconf_A_);
            auto term2_ptr = allocate_buffer<1>(grid_size_);

            term2_ptr->for_each_index(
                [SPTR(term2_ptr, contraconf_metric_ptr, coconf_A_derivative_ptr)](const auto idx) {
                    u8",jk,jki"_einsum((*term2_ptr)[idx],
                                       minus_half,
                                       (*contraconf_metric_ptr)[idx],
                                       (*coconf_A_derivative_ptr)[idx]);
                });
            term1_ptr->for_each_index(
                [SPTR(term1_ptr, term2_ptr)](const auto idx, const auto tidx) {
                    (*term1_ptr)[idx][tidx] += (*term2_ptr)[idx][tidx];
                });
            tensor_buffer_queue.wait();
        }

        { // terms 3 and 4
            auto term3_ptr = allocate_buffer<1>(grid_size_);

            term3_ptr->for_each_index(
                [SPTR(term3_ptr, W_derivative_ptr, contracoconf_A_ptr)](const auto idx) {
                    u8"j,ji"_einsum((*term3_ptr)[idx],
                                    (*W_derivative_ptr)[idx],
                                    (*contracoconf_A_ptr)[idx]);
                });

            term3_ptr->for_each_index([SPTR(term1_ptr, K_derivative_ptr, term3_ptr),
                                       this](const auto idx, const auto tidx) {
                const auto c = (real{ 2 } / real{ 3 }) * (*K_derivative_ptr)[idx][tidx];
                (*term1_ptr)[idx][tidx] += real{ -3 } * (*term3_ptr)[idx][tidx] / W_[idx][] - c;
            });
            tensor_buffer_queue.wait();
        }

        constraints_ptr->momentum = std::move(*term1_ptr);
    }

    { // momentum constraint damping
        const auto DiMj_ptr = std::invoke([&] {
            auto M_derivative_ptr = finite_difference::periodic_4th_order_central_1st_derivative(
                constraints_ptr->momentum);

            const auto chris_term_ptr = std::invoke([&] {
                auto temp_ptr = allocate_buffer<2>(grid_size_);
                temp_ptr->for_each_index([SPTR(temp_ptr,
                                               contraconf_metric_ptr,
                                               coconf_christoffels_ptr,
                                               constraints_ptr)](const auto idx) {
                    u8"ab,aij,b"_einsum((*temp_ptr)[idx],
                                        (*contraconf_metric_ptr)[idx],
                                        (*coconf_christoffels_ptr)[idx],
                                        constraints_ptr->momentum[idx]);
                });

                tensor_buffer_queue.wait();
                return temp_ptr;
            });

            M_derivative_ptr->for_each_index(
                [SPTR(M_derivative_ptr, chris_term_ptr)](const auto idx, const auto tidx) {
                    (*M_derivative_ptr)[idx][tidx] -= (*chris_term_ptr)[idx][tidx];
                });

            tensor_buffer_queue.wait();
            return M_derivative_ptr;
        });

        dfdt_ptr->coconf_A.for_each_index([SPTR(DiMj_ptr, dfdt_ptr), this](const auto idx,
                                                                           const auto tidx) {
            const auto symmDiMj = real{ 0.5 } * ((*DiMj_ptr)[idx][tidx] + (*DiMj_ptr)[idx][tidx]);

            // 0.025 is used in NR101.
            const auto damping_coeff = real{ 50 } * static_cast<real>(this->grid_size_.Nx);

            dfdt_ptr->coconf_A[idx][tidx] += damping_coeff * lapse_[idx][] * symmDiMj;
        });

        tensor_buffer_queue.wait();
    }

    return pre_calculations_type{ std::move(dfdt_ptr), std::move(constraints_ptr) };
}

[[nodiscard]]
w2_bssn_uniform_grid::w2_bssn_uniform_grid(const grid_size gs, gauge_wave_spacetime_tag)
    : grid_size_(gs),
      W_(gs),
      lapse_(gs),
      coconf_metric_(gs),
      K_(gs),
      coconf_A_(gs),
      contraconf_christoffel_trace_(gs) {
    assert(gs.Ny == 1);
    assert(gs.Ny == gs.Nz);

    const auto A = real{ 0.1 };
    // Assume that x coordinates are 0, 1 / Nx, ..., (Nx - 1) / Nx.
    const auto d = real{ 1 };

    // co_metric:
    auto co_metric_ptr = allocate_buffer<2>(grid_size_);

    co_metric_ptr->for_each_index(
        [this, d, A, SPTR(co_metric_ptr)](const auto idx, const auto tidx) {
            const auto diagonal         = tidx[0] == tidx[1];
            (*co_metric_ptr)[idx][tidx] = static_cast<real>(diagonal);

            const auto g0 = tidx[0] == 0;
            if (diagonal and g0) {
                // Assume that x coordinates are 0, 1 / Nx, ..., (Nx - 1) / Nx.
                const auto x = static_cast<real>(idx[0]) / static_cast<real>(this->grid_size_.Nx);
                const auto H = A * sycl::sin(real{ 2 } * std::numbers::pi_v<real> * x / d);

                (*co_metric_ptr)[idx][tidx] *= real{ 1 } - H;
            }
        });

    // W:
    //   - det(co_metric)

    const auto [det_metric_ptr, contra_metric_ptr] = det_n_inv3D(*co_metric_ptr);

    W_.for_each_index([this, SPTR(det_metric_ptr)](const auto idx) {
        W_[idx][] = sycl::pow((*det_metric_ptr)[idx][], real{ -1 / 6 });
    });

    // coconf_metric:
    //   - W
    //   - co_metric

    coconf_metric_.for_each_index([this, SPTR(co_metric_ptr)](const auto idx, const auto tidx) {
        coconf_metric_[idx][tidx] = W_[idx][] * W_[idx][] * (*co_metric_ptr)[idx][tidx];
    });

    // K:
    //   - contra_metric
    //   - co_K
    //      - lapse

    lapse_.for_each_index([this, d, A](const auto idx) {
        // Assume that x coordinates are 0, 1 / Nx, ..., (Nx - 1) / Nx.
        const auto x  = static_cast<real>(idx[0]) / static_cast<real>(this->grid_size_.Nx);
        const auto H  = A * sycl::sin(real{ 2 } * std::numbers::pi_v<real> * x / d);
        lapse_[idx][] = sycl::sqrt(real{ 1 } - H);
    });

    auto co_K_ptr = allocate_buffer<2>(grid_size_);

    co_K_ptr->for_each_index([this, SPTR(co_K_ptr), d, A](const auto idx, const auto tidx) {
        const auto g00 = (tidx[0] == 0uz) and (tidx[1] == 0uz);

        if (not g00) {
            (*co_K_ptr)[idx][tidx] = 0;
        } else {
            static constexpr auto two_pi = real{ 2 } * std::numbers::pi_v<real>;
            // Assume that x coordinates are 0, 1 / Nx, ..., (Nx - 1) / Nx.
            const auto x   = static_cast<real>(idx[0]) / static_cast<real>(this->grid_size_.Nx);
            const auto phi = two_pi * x / d;
            (*co_K_ptr)[idx][tidx] = -two_pi * A * sycl::cos(phi);
            (*co_K_ptr)[idx][tidx] /= real{ 2 } * lapse_[idx][] * d;
        }
    });

    K_.for_each_index([this, SPTR(contra_metric_ptr, co_K_ptr)](const auto idx) {
        u8"nm,nm"_einsum(K_[idx], (*contra_metric_ptr)[idx], (*co_K_ptr)[idx]);
    });

    // coconf_A:
    //   - W
    //   - co_K
    //   - co_metric
    //   - K

    auto K_trace_remover_ptr = allocate_buffer<2>(grid_size_);

    K_trace_remover_ptr->for_each_index(
        [SPTR(K_trace_remover_ptr, co_metric_ptr), this](const auto idx, const auto tidx) {
            static constexpr auto third = real{ 1 } / real{ 3 };

            (*K_trace_remover_ptr)[idx][tidx] = third * (*co_metric_ptr)[idx][tidx] * K_[idx][];
        });

    coconf_A_.for_each_index([this, SPTR(co_K_ptr, K_trace_remover_ptr)](const auto idx,
                                                                         const auto tidx) {
        const auto W2        = W_[idx][] * W_[idx][];
        coconf_A_[idx][tidx] = W2 * ((*co_K_ptr)[idx][tidx] - (*K_trace_remover_ptr)[idx][tidx]);
    });

    // contraconf_chriss:
    //   - contraconf_metric
    //   - contra_christoffels

    const auto [coconf_christoffels_ptr, _] =
        finite_difference::co_christoffel_symbols(coconf_metric_);
    const auto [_, contraconf_metric_ptr] = det_n_inv3D(coconf_metric_);

    contraconf_christoffel_trace_.for_each_index(
        [this, SPTR(coconf_christoffels_ptr, contraconf_metric_ptr)](const auto idx) {
            u8"mn,ij,jmn"_einsum(contraconf_christoffel_trace_[idx],
                                 (*contraconf_metric_ptr)[idx],
                                 (*contraconf_metric_ptr)[idx],
                                 (*coconf_christoffels_ptr)[idx]);
        });

    // We have to wait, because after return this might move before kernel is executed.
    // This is bad architecture, but too late to refactor whole thing.
    tensor_buffer_queue.wait();
}
