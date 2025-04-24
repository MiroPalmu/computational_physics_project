#include "w2_bssn_uniform_grid.hpp"

#include <experimental/mdspan>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "idg/einsum.hpp"
#include "mdspan_utils.hpp"

using namespace idg::literals;

namespace finite_difference {

/// Calculates derivative of arbitrary tensor T_{abc...} -> T_{abc...,i}
///
/// Assumes that tensor buffer elements are at seperated by h = 1.
template<std::size_t rank, typename T, typename Allocator>
[[nodiscard]]
auto
periodic_4th_order_central_1st_derivative(const tensor_buffer<rank, 3uz, T, Allocator>& f) {
    using buff_type = tensor_buffer<rank + 1uz, 3uz, T, Allocator>;

    auto derivatives = buff_type(f.size());

    f.for_each_index([&](const auto idx, const auto tidx) {
        const auto iuz = idx[0];
        const auto juz = idx[1];
        const auto kuz = idx[2];

        const auto i = static_cast<std::ptrdiff_t>(iuz);
        const auto j = static_cast<std::ptrdiff_t>(juz);
        const auto k = static_cast<std::ptrdiff_t>(kuz);

        const auto im2 = (i - 2) % f.size().Nx;
        const auto im1 = (i - 1) % f.size().Nx;
        const auto ip1 = (i + 1) % f.size().Nx;
        const auto ip2 = (i + 2) % f.size().Nx;

        const auto jm2 = (j - 2) % f.size().Ny;
        const auto jm1 = (j - 1) % f.size().Ny;
        const auto jp1 = (j + 1) % f.size().Ny;
        const auto jp2 = (j + 2) % f.size().Ny;

        const auto km2 = (k - 2) % f.size().Nz;
        const auto km1 = (k - 1) % f.size().Nz;
        const auto kp1 = (k + 1) % f.size().Nz;
        const auto kp2 = (k + 2) % f.size().Nz;

        using derivative_tidx_type = std::array<std::size_t, rank + 1>;

        auto xtidx = derivative_tidx_type{};
        auto ytidx = derivative_tidx_type{};
        auto ztidx = derivative_tidx_type{};
        rn::copy(tidx, xtidx.begin());
        rn::copy(tidx, ytidx.begin());
        rn::copy(tidx, ztidx.begin());
        xtidx.back() = 0uz;
        ytidx.back() = 1uz;
        ztidx.back() = 2uz;

        static constexpr auto a = T{ 1 } / T{ 12 };
        static constexpr auto b = T{ 2 } / T{ 3 };

        derivatives[idx][xtidx] = a * f[{ im2, juz, kuz }][tidx] - b * f[{ im1, juz, kuz }][tidx]
                                  + b * f[{ ip1, juz, kuz }][tidx] - a * f[{ ip2, juz, kuz }][tidx];

        derivatives[idx][ytidx] = a * f[{ iuz, jm2, kuz }][tidx] - b * f[{ iuz, jm1, kuz }][tidx]
                                  + b * f[{ iuz, jp1, kuz }][tidx] - a * f[{ iuz, jp2, kuz }][tidx];

        derivatives[idx][ztidx] = a * f[{ iuz, juz, km2 }][tidx] - b * f[{ iuz, juz, km1 }][tidx]
                                  + b * f[{ iuz, juz, kp1 }][tidx] - a * f[{ iuz, juz, kp2 }][tidx];
    });

    return derivatives;
}

/// Covariant christoffel symbols, i.e. Christoffel symbols of the 1st kind.
///
/// Also, as a side product returns derivatives of metric g_{ij}, i.e. g_{ij,k}.
template<typename T, typename Allocator>
[[nodiscard]]
std::pair<w2_bssn_uniform_grid::buffer3, w2_bssn_uniform_grid::buffer3>
co_christoffel_symbols(const tensor_buffer<2, 3, T, Allocator>& co_spatial_metric) {
    auto dg           = periodic_4th_order_central_1st_derivative(co_spatial_metric);
    auto christoffels = tensor_buffer<3, 3, T, Allocator>(co_spatial_metric.size());

    using tidx_type = std::array<std::size_t, 3>;
    christoffels.for_each_index([&](const auto idx, const tidx_type tidx) {
        const auto cab = tidx;
        const auto cba = tidx_type{ cab[0], cab[2], cab[1] };
        const auto abc = tidx_type{ cab[1], cab[2], cab[0] };

        christoffels[idx][cab] = (T{ 1 } / T{ 2 }) * (dg[idx][cab] + dg[idx][cba] - dg[idx][abc]);
    });

    return { std::move(christoffels), std::move(dg) };
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
    coconf_metric_.for_each_index([&](const auto idx, const auto tidx) {
        coconf_metric_[idx][tidx] = static_cast<real>(tidx[0] == tidx[1]);
    });

    lapse_.for_each_index([&](const auto idx) { lapse_[idx][] = 1; });

    K_.for_each_index([&](const auto idx) { K_[idx][] = 0; });
    W_.for_each_index([&](const auto idx) { W_[idx][] = 1; });

    coconf_A_.for_each_index([&](const auto idx, const auto tidx) { coconf_A_[idx][tidx] = 0; });

    contraconf_christoffel_trace_.for_each_index(
        [&](const auto idx, const auto tidx) { contraconf_christoffel_trace_[idx][tidx] = 0; });
}

[[nodiscard]]
std::pair<w2_bssn_uniform_grid::buffer0, w2_bssn_uniform_grid::buffer2>
det_n_inv3D(const w2_bssn_uniform_grid::buffer2& matrix) {
    auto det = w2_bssn_uniform_grid::buffer0(matrix.size());
    auto inv = w2_bssn_uniform_grid::buffer2(matrix.size());
    det.for_each_index([&](const auto idx) {
        const auto a = matrix[idx][0, 0];
        const auto b = matrix[idx][0, 1];
        const auto c = matrix[idx][0, 2];
        const auto d = matrix[idx][1, 0];
        const auto e = matrix[idx][1, 1];
        const auto f = matrix[idx][1, 2];
        const auto g = matrix[idx][2, 0];
        const auto h = matrix[idx][2, 1];
        const auto i = matrix[idx][2, 2];

        const auto A = (e * i - f * h);
        const auto B = -(d * i - f * g);
        const auto C = (d * h - e * g);

        const auto D = -(b * i - c * h);
        const auto E = (a * i - c * g);
        const auto F = -(a * h - b * g);

        const auto G = (b * f - c * e);
        const auto H = -(a * f - c * d);
        const auto I = (a * e - d * d);

        det[idx][] = a * A + b * b + c * C;

        inv[idx][0, 0] = A / det[idx][];
        inv[idx][1, 0] = B / det[idx][];
        inv[idx][2, 0] = C / det[idx][];
        inv[idx][0, 1] = D / det[idx][];
        inv[idx][1, 1] = E / det[idx][];
        inv[idx][2, 1] = F / det[idx][];
        inv[idx][0, 2] = G / det[idx][];
        inv[idx][1, 2] = H / det[idx][];
        inv[idx][2, 2] = I / det[idx][];
    });

    return { std::move(det), std::move(inv) };
}

void
w2_bssn_uniform_grid::beve_dump(const std::filesystem::path& dump_dir_name) {
    const auto dir_path = std::filesystem::weakly_canonical(dump_dir_name);
    std::filesystem::create_directory(dir_path);
    auto file_path = [&](const std::filesystem::path& filename) { return dir_path / filename; };

    {
        [[maybe_unused]] auto [S, _] = det_n_inv3D(coconf_metric_);
        S.for_each_index([&](const auto idx) { S[idx][] -= 1; });
        S.write_as_beve(file_path("algebraic_constraint_S.beve"));
    }

    {
        auto conformal_A_trace = buffer0(coconf_A_.size());

        [[maybe_unused]] const auto [_, contraconf_spatial_metric] = det_n_inv3D(coconf_metric_);
        conformal_A_trace.for_each_index([&](const auto idx) {
            u8"ij,ij"_einsum(conformal_A_trace[idx],
                             contraconf_spatial_metric[idx],
                             coconf_A_[idx]);
        });
        conformal_A_trace.write_as_beve(file_path("algebraic_constraint_conformal_A.beve"));
    }
}

w2_bssn_uniform_grid::time_derivative_type::time_derivative_type(const grid_size gs)
    : lapse(gs),
      W(gs),
      coconf_metric(gs),
      K(gs),
      coconf_A(gs),
      contraconf_christoffel_trace(gs) {}

w2_bssn_uniform_grid::time_derivative_type
w2_bssn_uniform_grid::time_derivative() {
    w2_bssn_uniform_grid::time_derivative_type dfdt(grid_size_);

    dfdt.lapse.for_each_index(
        [&](const auto idx) { dfdt.lapse[idx][] = -lapse_[idx][] * lapse_[idx][] * K_[idx][]; });

    dfdt.W.for_each_index(
        [&](const auto idx) { dfdt.W[idx][] = W_[idx][] * lapse_[idx][] * K_[idx][] / real{ 3 }; });

    dfdt.coconf_metric.for_each_index([&](const auto idx) {
        static constexpr auto minus2 = constant_geometric_mdspan<0, 3, real{ -2 }>();
        u8",,ij"_einsum(dfdt.coconf_metric[idx], minus2, lapse_[idx], coconf_A_[idx]);
    });

    [[maybe_unused]] const auto [_, contraconf_spatial_metric] = det_n_inv3D(coconf_metric_);

    const auto [coconf_christoffels, coconf_metric_derivative] =
        finite_difference::co_christoffel_symbols(coconf_metric_);

    const auto W_derivative = finite_difference::periodic_4th_order_central_1st_derivative(W_);
    const auto lapse_derivative =
        finite_difference::periodic_4th_order_central_1st_derivative(lapse_);

    const auto co_W2DiDj_lapse = std::invoke([&] {
        const auto coconf_DiDj_lapse = std::invoke([&] {
            auto lapse_2nd_derivative =
                finite_difference::periodic_4th_order_central_1st_derivative(lapse_derivative);

            auto christoffel_term = buffer2(grid_size_);
            christoffel_term.for_each_index([&](const auto idx) {
                u8"ij,jab,i"_einsum(christoffel_term[idx],
                                    contraconf_spatial_metric[idx],
                                    coconf_christoffels[idx],
                                    lapse_derivative[idx]);
            });

            lapse_2nd_derivative.for_each_index([&](const auto idx, const auto tidx) {
                lapse_2nd_derivative[idx][tidx] -= christoffel_term[idx][tidx];
            });

            return lapse_2nd_derivative;
        });

        const auto dWdlapse = std::invoke([&] {
            auto temp = buffer2(grid_size_);
            temp.for_each_index([&](const auto idx) {
                u8"i,j"_einsum(temp[idx], W_derivative[idx], lapse_derivative[idx]);
            });
            return temp;
        });

        const auto last_term = std::invoke([&] {
            auto temp = buffer2(grid_size_);
            temp.for_each_index([&](const auto idx) {
                u8"ij,nm,nm"_einsum(temp[idx],
                                    coconf_metric_[idx],
                                    contraconf_spatial_metric[idx],
                                    dWdlapse[idx]);
            });
            return temp;
        });

        auto temp = buffer2(grid_size_);
        temp.for_each_index([&](const auto idx, const auto tidx) {
            temp[idx][tidx] = W_[idx][] * coconf_DiDj_lapse[idx][tidx] + dWdlapse[idx][tidx]
                              + dWdlapse[idx][std::array{ tidx[1], tidx[0] }]
                              - last_term[idx][tidx];
            temp[idx][tidx] *= W_[idx][];
        });
        return temp;
    });

    const auto contraconf_A = std::invoke([&] {
        auto temp = buffer2(grid_size_);
        temp.for_each_index([&](const auto idx) {
            u8"ia,jb,ab"_einsum(temp[idx],
                                contraconf_spatial_metric[idx],
                                contraconf_spatial_metric[idx],
                                coconf_A_[idx]);
        });
        return temp;
    });

    { // calculate dfdt.K
        const auto term1 = std::invoke([&] {
            auto temp = buffer0(grid_size_);
            temp.for_each_index([&](const auto idx) {
                u8"nm,nm"_einsum(temp[idx], contraconf_spatial_metric[idx], co_W2DiDj_lapse[idx]);
            });
            return temp;
        });

        const auto term2 = std::invoke([&] {
            auto temp = buffer0(grid_size_);
            temp.for_each_index([&](const auto idx) {
                u8",nm,nm"_einsum(temp[idx], lapse_[idx], contraconf_A[idx], coconf_A_[idx]);
            });
            return temp;
        });

        dfdt.K.for_each_index([&](const auto idx) {
            const auto term3 = lapse_[idx][] * K_[idx][] * K_[idx][] / real{ 3 };
            dfdt.K[idx][]    = -term1[idx][] + term2[idx][] + term3;
        });
    }

    static constexpr auto two = constant_geometric_mdspan<0, 3, real{ 2 }>();
    { // calculate dfdt.coconf_A
        auto term1 = buffer2(grid_size_);
        term1.for_each_index([&](const auto idx) {
            u8",,ij"_einsum(term1[idx], lapse_[idx], K_[idx], coconf_A_[idx]);
        });

        { // term 2
            auto term2 = buffer2(grid_size_);
            term2.for_each_index([&](const auto idx) {
                u8",,im,mn,nj"_einsum(term2[idx],
                                      two,
                                      lapse_[idx],
                                      coconf_A_[idx],
                                      contraconf_spatial_metric[idx],
                                      coconf_A_[idx]);
            });

            term1.for_each_index(
                [&](const auto idx, const auto tidx) { term1[idx][tidx] -= term2[idx][tidx]; });
        }

        // trace free term
        auto TFterm = std::invoke([&] {
            const auto coconf_R = std::invoke([&] {
                auto term1 = std::invoke([&] {
                    static constexpr auto minus_half =
                        constant_geometric_mdspan<0, 3, real{ -1 } / real{ 2 }>();

                    const auto coconf_metric_2nd_derivative =
                        finite_difference::periodic_4th_order_central_1st_derivative(
                            coconf_metric_derivative);

                    auto temp = buffer2(grid_size_);
                    temp.for_each_index([&](const auto idx) {
                        u8",nm,ijnm"_einsum(temp[idx],
                                            minus_half,
                                            contraconf_spatial_metric[idx],
                                            coconf_metric_2nd_derivative[idx]);
                    });
                    return temp;
                });

                { // terms 2 and 3
                    auto term2 = buffer2(grid_size_);
                    auto term3 = buffer2(grid_size_);

                    const auto contraconf_christoffel_trace_derivative =
                        finite_difference::periodic_4th_order_central_1st_derivative(
                            contraconf_christoffel_trace_);

                    term2.for_each_index([&](const auto idx) {
                        u8"mi,mj -> ij"_einsum(term2[idx],
                                               coconf_metric_[idx],
                                               contraconf_christoffel_trace_derivative[idx]);
                    });

                    term3.for_each_index([&](const auto idx) {
                        u8"mi,mj -> ji"_einsum(term3[idx],
                                               coconf_metric_[idx],
                                               contraconf_christoffel_trace_derivative[idx]);
                    });

                    term1.for_each_index([&](const auto idx, const auto tidx) {
                        term1[idx][tidx] +=
                            (real{ 1 } / real{ 2 }) * (term2[idx][tidx] + term3[idx][tidx]);
                    });
                }

                { // terms 4 and 5
                    auto term4 = buffer2(grid_size_);
                    auto term5 = buffer2(grid_size_);

                    term4.for_each_index([&](const auto idx) {
                        u8"ab,cm,cab,ijm -> ij"_einsum(term4[idx],
                                                       contraconf_spatial_metric[idx],
                                                       contraconf_spatial_metric[idx],
                                                       coconf_christoffels[idx],
                                                       coconf_christoffels[idx]);
                    });

                    term5.for_each_index([&](const auto idx) {
                        u8"ab,cm,cab,ijm -> ji"_einsum(term5[idx],
                                                       contraconf_spatial_metric[idx],
                                                       contraconf_spatial_metric[idx],
                                                       coconf_christoffels[idx],
                                                       coconf_christoffels[idx]);
                    });

                    term1.for_each_index([&](const auto idx, const auto tidx) {
                        term1[idx][tidx] +=
                            (real{ 1 } / real{ 2 }) * (term4[idx][tidx] + term5[idx][tidx]);
                    });
                }

                { // terms 6 and 7
                    auto term6 = buffer2(grid_size_);
                    auto term7 = buffer2(grid_size_);

                    term6.for_each_index([&](const auto idx) {
                        u8"ab,nm,ani,jbm -> ij"_einsum(term6[idx],
                                                       contraconf_spatial_metric[idx],
                                                       contraconf_spatial_metric[idx],
                                                       coconf_christoffels[idx],
                                                       coconf_christoffels[idx]);
                    });

                    term7.for_each_index([&](const auto idx) {
                        u8"ab,nm,ani,jbm -> ji"_einsum(term6[idx],
                                                       contraconf_spatial_metric[idx],
                                                       contraconf_spatial_metric[idx],
                                                       coconf_christoffels[idx],
                                                       coconf_christoffels[idx]);
                    });

                    term1.for_each_index([&](const auto idx, const auto tidx) {
                        term1[idx][tidx] += term6[idx][tidx] + term7[idx][tidx];
                    });
                }

                { // term 8
                    auto term8 = buffer2(grid_size_);

                    term8.for_each_index([&](const auto idx) {
                        u8"ab,nm,ain,bjm -> ij"_einsum(term8[idx],
                                                       contraconf_spatial_metric[idx],
                                                       contraconf_spatial_metric[idx],
                                                       coconf_christoffels[idx],
                                                       coconf_christoffels[idx]);
                    });

                    term1.for_each_index([&](const auto idx, const auto tidx) {
                        term1[idx][tidx] += term8[idx][tidx];
                    });
                }

                return term1;
            });

            const auto coconf_W2Rw = std::invoke([&] {
                const auto coconf_DiDj_W = std::invoke([&] {
                    auto christoffel_term = buffer2(grid_size_);
                    christoffel_term.for_each_index([&](const auto idx) {
                        u8"ij,jab,i"_einsum(christoffel_term[idx],
                                            contraconf_spatial_metric[idx],
                                            coconf_christoffels[idx],
                                            W_derivative[idx]);
                    });

                    auto W_2nd_derivative =
                        finite_difference::periodic_4th_order_central_1st_derivative(W_derivative);

                    W_2nd_derivative.for_each_index([&](const auto idx, const auto tidx) {
                        W_2nd_derivative[idx][tidx] -= christoffel_term[idx][tidx];
                    });

                    return W_2nd_derivative;
                });

                auto term1 = coconf_DiDj_W;

                { // term 2
                    auto term2 = buffer2(grid_size_);
                    term2.for_each_index([&](const auto idx) {
                        u8"ij,nm,nm"_einsum(term2[idx],
                                            coconf_metric_[idx],
                                            contraconf_spatial_metric[idx],
                                            coconf_DiDj_W[idx]);
                    });

                    term1.for_each_index([&](const auto idx, const auto tidx) {
                        term1[idx][tidx] = W_[idx][] * (term1[idx][tidx] - term2[idx][tidx]);
                    });
                }

                { // term 3
                    auto term3 = buffer2(grid_size_);
                    term3.for_each_index([&](const auto idx) {
                        u8",ij,mn,m,n"_einsum(term3[idx],
                                              two,
                                              coconf_metric_[idx],
                                              contraconf_spatial_metric[idx],
                                              W_derivative[idx],
                                              W_derivative[idx]);
                    });

                    term1.for_each_index([&](const auto idx, const auto tidx) {
                        term1[idx][tidx] -= term3[idx][tidx];
                    });
                }

                return term1;
            });

            auto lapseW2R = buffer2(grid_size_);
            lapseW2R.for_each_index([&](const auto idx, const auto tidx) {
                lapseW2R[idx][tidx] =
                    lapse_[idx][]
                    * (coconf_W2Rw[idx][tidx] + W_[idx][] * W_[idx][] * coconf_R[idx][tidx]);
            });

            lapseW2R.for_each_index([&](const auto idx, const auto tidx) {
                lapseW2R[idx][tidx] -= co_W2DiDj_lapse[idx][tidx];
            });

            return lapseW2R;
        });

        const auto trace_remover = std::invoke([&] {
            auto temp = buffer2(grid_size_);
            // It does not matter if trace remover is calculated with conformal or
            // nor conformal metric. At least with W^2, they cancel out:
            // g_{ij}g^{nm} = \tilde{g}_{ij}\tilde{g}^{nm}
            temp.for_each_index([&](const auto idx) {
                static constexpr auto third =
                    constant_geometric_mdspan<0, 3, real{ 1 } / real{ 3 }>();
                u8",ij,nm,nm"_einsum(temp[idx],
                                     third,
                                     coconf_metric_[idx],
                                     contraconf_spatial_metric[idx],
                                     TFterm[idx]);
            });
            return temp;
        });

        dfdt.coconf_A.for_each_index([&](const auto idx, const auto tidx) {
            dfdt.coconf_A[idx][tidx] =
                term1[idx][tidx] + TFterm[idx][tidx] - trace_remover[idx][tidx];
        });
    }

    { // calculate dfdt.contraconf_christoffel_trace
        auto term1 = std::invoke([&] {
            auto tempA = W_derivative;
            tempA.for_each_index([&](const auto idx, const auto tidx) {
                tempA[idx][tidx] *= real{ -6 } * lapse_[idx][] / W_[idx][];
                tempA[idx][tidx] += real{ -2 } * lapse_derivative[idx][tidx];
            });

            auto tempB = buffer1(grid_size_);
            tempB.for_each_index([&](const auto idx) {
                u8"im,m"_einsum(tempB[idx], contraconf_A[idx], tempA[idx]);
            });
            return tempB;
        });

        { // term 2
            auto term2 = buffer1(grid_size_);
            term2.for_each_index([&](const auto idx) {
                u8",,ia,abc,bc"_einsum(term2[idx],
                                       two,
                                       lapse_[idx],
                                       contraconf_spatial_metric[idx],
                                       coconf_christoffels[idx],
                                       contraconf_A[idx]);
            });

            term1.for_each_index(
                [&](const auto idx, const auto tidx) { term1[idx][tidx] += term2[idx][tidx]; });
        }

        { // term 3
            const auto K_derivative =
                finite_difference::periodic_4th_order_central_1st_derivative(K_);

            auto term3 = buffer1(grid_size_);
            term3.for_each_index([&](const auto idx) {
                u8"im,m"_einsum(term3[idx], contraconf_spatial_metric[idx], K_derivative[idx]);
            });

            dfdt.contraconf_christoffel_trace.for_each_index([&](const auto idx, const auto tidx) {
                dfdt.contraconf_christoffel_trace[idx][tidx] =
                    term1[idx][tidx] - (real{ 4 } * lapse_[idx][] / real{ 3 }) * term3[idx][tidx];
            });
        }
    }

    return dfdt;
};
