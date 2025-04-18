#pragma once
/// @file Implements numpy einsum inspired functionality for mdspans.

#include <algorithm>
#include <iterator>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <experimental/mdspan>

#include "idg/generic_algorithm.hpp"
#include "idg/sstd.hpp"
#include "idg/string_manipulation.hpp"
#include "idg/tensor_network.hpp"

namespace idg {

namespace rn = std::ranges;
namespace rv = std::views;
using namespace std::literals;

class einsum_parser {
  public:
    using index_label             = std::u8string;
    using factor_index_labels_vec = std::vector<index_label>;

    /// Cursor concept borrowed from flux c++ library.
    struct index_cursor {
        std::size_t factor, index;
        [[nodiscard]] constexpr bool operator==(const index_cursor&) const = default;
    };

    // constexpr std::[flat_]set is not implemented
    using contraction = sstd::constexpr_set<index_cursor>;

  private:
    std::vector<factor_index_labels_vec> factor_index_labels_{};
    factor_index_labels_vec output_index_labels_{};

    factor_index_labels_vec free_index_labels_{};
    std::vector<contraction> contractions_{};

    [[nodiscard]] constexpr auto iterate_index_labels_with_cursor(this auto&& self) {
        return self.factor_index_labels_ | rv::enumerate | rv::transform([](const auto& t) {
                   const auto factor_ordinal       = static_cast<std::size_t>(std::get<0>(t));
                   const auto& factor_index_labels = std::get<1>(t);
                   auto cursors                    = rv::iota(0uz, rn::size(factor_index_labels))
                                  | rv::transform([=](const std::size_t i) {
                                        return index_cursor{ factor_ordinal, i };
                                    });
                   return rv::zip(cursors, factor_index_labels);
               })
               | rv::join;
    };

    [[nodiscard]] static constexpr rn::view auto split_to_index_labels(rn::view auto const str) {
        return rv::all(str) | rv::transform([](const auto x) { return std::u8string{ x }; });
    }

  public:
    [[nodiscard]] constexpr einsum_parser(const std::u8string_view str) {
        auto factor_and_maybe_output_index_labels =
            str | rv::filter([](const char8_t c) { return not str::is_whitespace(c); })
            | rv::split(u8"->"sv);

        for (const auto factor_str :
             *factor_and_maybe_output_index_labels.begin() | rv::split(u8',')) {
            factor_index_labels_.push_back({});

            for (const auto x : split_to_index_labels(factor_str)) {
                factor_index_labels_.back().push_back(std::move(x));
            }
        }

        // Fill contractions_:

        // std::size_t represent the ordinal of the contraction corresponding to the index label.
        // index_cursor stores the first time the label is encountered.
        auto seen_labels = std::vector<std::tuple<index_label, std::size_t, index_cursor>>{};
        // When label is encountered first time, this is used as contraction ordinal.
        static constexpr std::size_t not_contraction = static_cast<std::size_t>(-1);

        rn::for_each(iterate_index_labels_with_cursor(), [&](const auto t) {
            const auto cursor = std::get<0>(t);
            const auto label  = std::get<1>(t);

            const auto get_label = [](const auto t) { return std::get<0>(t); };
            const auto prev      = rn::find(seen_labels, label, get_label);

            if (prev == seen_labels.end()) {
                // First time.
                seen_labels.push_back({ label, not_contraction, cursor });
            } else {
                const auto second_time = std::get<1>(*prev) == not_contraction;

                if (second_time) {
                    std::get<1>(*prev) = std::size(contractions_);
                    contractions_.push_back({});
                    std::ignore = contractions_.back().successfully_insert(std::get<2>(*prev));
                }

                if (not contractions_[std::get<1>(*prev)].successfully_insert(cursor)) {
                    throw std::logic_error{ "Each cursor should only appear once." };
                }
            }
        });

        // Figure out free index labels.
        rn::copy(seen_labels | rv::filter([](const auto t) {
                     return not_contraction == std::get<1>(t);
                 }) | rv::transform([](const auto t) { return std::get<0>(t); }),
                 std::back_inserter(free_index_labels_));

        if (rn::distance(factor_and_maybe_output_index_labels) == 1) {
            // Implicit return indices.
            output_index_labels_ = free_index_labels_;

        } else if (rn::distance(factor_and_maybe_output_index_labels) == 2) {
            // Explicit return indices.
            const auto return_str = *rn::next(rn::begin(factor_and_maybe_output_index_labels));
            rn::copy(split_to_index_labels(return_str), std::back_inserter(output_index_labels_));
        } else {
            // Error.
            throw std::logic_error{ "-> can only appear once." };
        }
    }

    [[nodiscard]] constexpr std::size_t number_of_factors(this auto&& self) {
        return self.factor_index_labels_.size();
    };

    [[nodiscard]] constexpr std::span<factor_index_labels_vec const>
        factor_index_labels(this auto&& self) {
        return { self.factor_index_labels_ };
    }

    [[nodiscard]] constexpr std::span<const contraction> contractions(this auto&& self) {
        return { self.contractions_ };
    }

    [[nodiscard]] constexpr std::span<const index_label> output_index_labels(this auto&& self) {
        return { self.output_index_labels_ };
    }

    [[nodiscard]] constexpr std::span<const index_label> free_index_labels(this auto&& self) {
        return { self.free_index_labels_ };
    }
};

template<typename OutMDS>
[[nodiscard]] constexpr bool einsum_valid_ouput_type(const std::u8string_view estr) {
    const auto parser = einsum_parser(estr);
    return rn::size(parser.output_index_labels()) == OutMDS::rank();
}

template<typename... MDS>
[[nodiscard]] constexpr bool einsum_valid_factor_types(const std::u8string_view estr) {
    const auto parser = einsum_parser(estr);

    if (sizeof...(MDS) != parser.number_of_factors()) { return false; }

    return std::invoke(
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            return ((MDS::rank() == rn::size(parser.factor_index_labels()[I])) and ...);
        },
        std::make_index_sequence<sizeof...(MDS)>());
}

template<typename... MDS>
[[nodiscard]] constexpr bool einsum_consistent_geometric_dimentions() {
    struct mds_dim_result {
        std::optional<std::size_t> dim;
        bool consistent;
    };

    const auto mds_dim = [&]<typename M>(std::type_identity<M>) -> mds_dim_result {
        if (M::rank() == 0) { return { .dim = std::nullopt, .consistent{ true } }; }

        const auto found_dim = M::static_extent(0);
        for (const auto i : rv::iota(1uz, M::rank())) {
            if (M::static_extent(i) != found_dim) {
                return { .dim = std::nullopt, .consistent{ false } };
            }
        }

        if (found_dim == 0) { return { .dim = 0, .consistent = false }; }

        return { .dim = found_dim, .consistent{ true } };
    };

    if (not(mds_dim(std::type_identity<MDS>{}).consistent and ...)) { return false; }

    const auto found_dims = std::array{ mds_dim(std::type_identity<MDS>{}).dim... };
    auto found_dims_wout_rank0s =
        found_dims | rv::filter([](const auto x) { return x.has_value(); });

    if (rn::distance(found_dims_wout_rank0s) == 0) { return true; }
    return rn::max(found_dims_wout_rank0s) == rn::min(found_dims_wout_rank0s);
};

template<typename MDS>
concept static_extent_mdspan =
    (sstd::is_mdspan_v<MDS>)
    and (MDS::rank() == 0uz
         or rn::all_of(rv::iota(0uz, MDS::rank())
                           | rv::transform([](const auto i) { return MDS::static_extent(i); }),
                       [](const auto e) { return e != std::dynamic_extent; }));

template<str::fixed_string estr, typename OutMDS, typename... MDS>
concept einsum_compatible = (static_extent_mdspan<std::remove_cvref_t<OutMDS>> and ...
                             and static_extent_mdspan<std::remove_cvref_t<MDS>>)
                            and (einsum_consistent_geometric_dimentions<OutMDS, MDS...>())
                            and (einsum_valid_ouput_type<OutMDS>(estr.sv()))
                            and (einsum_valid_factor_types<MDS...>(estr.sv()));

template<str::fixed_string estr>
class einsum {
    static constexpr einsum_parser parser() { return einsum_parser(estr.sv()); }

    [[nodiscard]] static constexpr std::pair<std::vector<tensor_network::node_id>, tensor_network>
        network() {
        auto net = tensor_network();
        auto p   = parser();

        const auto id_vec = p.factor_index_labels() | rv::transform(rn::size)
                            | rv::transform([&](const auto r) { return net.add_node(r); })
                            | rn::to<std::vector>();

        for (const auto& contraction : p.contractions()) {
            const auto indices = contraction.get_data();
            if (rn::size(indices) != 2) {
                throw std::logic_error{ "Reuction has to have to connect two indices." };
            }
            const auto [lhs_factor, lhs_index] = indices[0];
            const auto [rhs_factor, rhs_index] = indices[1];

            net.add_edge({ id_vec[lhs_factor], lhs_index }, { id_vec[rhs_factor], rhs_index });
        }
        return { id_vec, net };
    }

    /// Deduce dimension from T... mdspans which are assumed to satisfy einsum_compatible.
    ///
    /// Finds first non rank-0 mdspan and returns it's static_extent(0).
    /// If all mdspans are rank-0, choose the dimension to be 1.
    template<typename... T>
    static constexpr std::size_t deduce_dimension() {
        return []<typename Head, typename... Tail>(this auto&& self,
                                                   std::type_identity<Head>,
                                                   std::type_identity<Tail>...) -> std::size_t {
            if constexpr (Head::rank() != 0) {
                return Head::static_extent(0);
            } else if constexpr (0 == sizeof...(Tail)) {
                // OutMDS and every MDS... (i.e. T...) is rank 0,
                // so we set dimension to one.
                return 1uz;
            } else {
                return self.template operator()(std::type_identity<Tail>{}...);
            }
        }(std::type_identity<T>{}...);
    }

  public:
    static constexpr std::size_t rank() {
        return rn::distance(einsum_parser(estr.sv()).free_index_labels());
    };

    // Given a element from output index space and a element from contraction index space,
    // which are concatted together. This tuple holds indices to the concatted elements
    // for each factor.
    static constexpr auto index_map = std::invoke([] {
        static constexpr auto number_of_contractions = rn::size(parser().contractions());
        const auto find_contraction =
            [&](const std::size_t factor_ordinal,
                const std::size_t index_rank) -> std::optional<std::size_t> {
            for (const auto i : rv::iota(0uz, number_of_contractions)) {
                if (parser().contractions()[i].contains({ factor_ordinal, index_rank })) {
                    return i;
                }
            }
            return {};
        };

        const auto handle_factor = [&]<std::size_t J>(std::integral_constant<std::size_t, J>) {
            constexpr auto this_factor_rank = rn::size(parser().factor_index_labels()[J]);
            auto this_factor_index_map      = std::array<std::size_t, this_factor_rank>{};

            for (const auto i : rv::iota(0uz, this_factor_rank)) {
                if (const auto contraction_ordinal = find_contraction(J, i)) {
                    this_factor_index_map[i] =
                        rn::size(parser().output_index_labels()) + contraction_ordinal.value();
                } else {
                    // .value() should newer throw.
                    this_factor_index_map[i] = alg::argfind(parser().output_index_labels(),
                                                            parser().factor_index_labels()[J][i])
                                                   .value();
                }
            }
            return this_factor_index_map;
        };

        return std::invoke(
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                return std::tuple{ handle_factor(std::integral_constant<std::size_t, I>{})... };
            },
            std::make_index_sequence<parser().number_of_factors()>());
    });

    static constexpr auto apply_index_map(
        const std::array<std::size_t, rn::size(parser().output_index_labels())>& out_idx,
        const std::array<std::size_t, rn::size(parser().contractions())>& reduced_idx) {
        // If rv::concat is implemented:
        // rn::random_access_range auto const concatted_indices =
        //     rv::concat(out_indices, reduced_indices);

        // Until then:
        std::array<std::size_t,
                   rn::size(parser().output_index_labels()) + rn::size(parser().contractions())>
            concatted_indices{};
        rn::copy(out_idx, rn::begin(concatted_indices));

        static constexpr auto num_of_output_index_labels = rn::size(parser().output_index_labels());
        rn::copy(reduced_idx, rn::next(rn::begin(concatted_indices), num_of_output_index_labels));

        return std::invoke(
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                const auto handle_factor =
                    [&]<std::size_t n>(const std::array<std::size_t, n>& fac) {
                        auto b = std::array<std::size_t, n>{};
                        rn::copy(fac | rv::transform([&](const std::size_t i) {
                                     return concatted_indices[i];
                                 }),
                                 b.begin());
                        return b;
                    };

                return std::tuple{ handle_factor(std::get<I>(index_map))... };
            },
            std::make_index_sequence<rn::size(parser().factor_index_labels())>());
    }

    template<typename OutMDS, typename... MDS>
        requires einsum_compatible<estr, OutMDS, MDS...>
    static constexpr void operator()(OutMDS out, MDS... factors) {
        static constexpr auto number_of_free_indices = OutMDS::rank();
        static constexpr auto dimension              = einsum::deduce_dimension<OutMDS, MDS...>();

        [[maybe_unused]] static constexpr auto number_of_connected_components = std::invoke([] {
            const auto [_, net] = network();
            return rn::size(net.connected_components());
        });

        static constexpr auto needs_optimization =
            sizeof...(factors) <= 2uz or rn::empty(parser().contractions());
        if constexpr (needs_optimization) {
            // static constexpr auto contraction_index_space =
            //     sstd::geometric_index_space<rn::size(parser().contractions()), dimension>();

            // static constexpr auto out_index_space =
            //     sstd::geometric_index_space<number_of_free_indices, dimension>();

            // Technically we could use std::execution::unseq policy with std::for_each,
            // but it requires tbb dependency on gcc and
            // the calculation gets optimized anyway.
            // rn::for_each(out_index_space, handle_output_index_space_element);

            static constexpr auto out_index_space_length =
                sstd::integer_pow(dimension, number_of_free_indices);
            static constexpr auto out_index_space_dividers = std::invoke(
                []<std::size_t... I>(std::index_sequence<I...>) {
                    return std::array<std::size_t, number_of_free_indices>{ sstd::integer_pow(
                        dimension,
                        static_cast<std::size_t>(number_of_free_indices - 1uz - I))... };
                },
                std::make_index_sequence<number_of_free_indices>());

            static constexpr auto number_of_contractions = rn::size(parser().contractions());
            static constexpr auto contraction_index_space_length =
                sstd::integer_pow(dimension, number_of_contractions);
            static constexpr auto contraction_index_space_dividers = std::invoke(
                []<std::size_t... I>(std::index_sequence<I...>) {
                    return std::array<std::size_t, number_of_contractions>{ sstd::integer_pow(
                        dimension,
                        static_cast<std::size_t>(number_of_contractions - 1uz - I))... };
                },
                std::make_index_sequence<number_of_contractions>());

            for (const auto i : rv::iota(0uz, out_index_space_length)) {
                auto out_idx = std::array<std::size_t, number_of_free_indices>{};
                for (const auto ii : rv::iota(0uz, out_idx.size())) {
                    out_idx[ii] = (i / out_index_space_dividers[ii]) % dimension;
                }

                out[out_idx] = typename OutMDS::value_type{};

                for (const auto j : rv::iota(0uz, contraction_index_space_length)) {
                    auto contraction_idx = std::array<std::size_t, number_of_contractions>{};
                    for (const auto jj : rv::iota(0uz, contraction_idx.size())) {
                        contraction_idx[jj] =
                            (j / contraction_index_space_dividers[jj]) % dimension;
                    }

                    const auto sorted_indices = apply_index_map(out_idx, contraction_idx);

                    out[out_idx] += std::invoke(
                        [&]<std::size_t... I>(std::index_sequence<I...>) {
                            return (factors[std::get<I>(sorted_indices)] * ...);
                        },
                        std::make_index_sequence<parser().number_of_factors()>());
                }
            }
        } else {
            // There are three different connected component types:
            //
            //     A) one node, no contractions
            //     B) one node, contractions
            //     C) multiple nodes, (implies) contractions
            //
            // For each connected component there is "out_mdspan",
            // but it means different things for different cases:
            //
            //     A) just the factor it represents
            //     B) output of the single node contractions
            //     C) output of the last pairwise contraction
            //
            // In any case, if there is just one connected component,
            // then its output mdspan will be the overall out mdspan
            // and the outer product of the components is not needed.

            struct connected_component_info {
                std::size_t rank;
                std::size_t number_of_contractions;
                // Ordinal of the factor which corresponds to the one node connected component.
                // Empty optional corresponds to case C).
                std::optional<std::size_t> one_node_factor_ordinal;

                [[nodiscard]] constexpr std::size_t out_buff_size(const std::size_t D) const {
                    if (number_of_contractions == 0uz) {
                        return 0uz;
                    } else {
                        return sstd::integer_pow(D, rank);
                    }
                }
                [[nodiscard]] constexpr bool case_A() const {
                    return one_node_factor_ordinal.has_value() and number_of_contractions == 0uz;
                }

                [[nodiscard]] constexpr bool case_B() const {
                    return one_node_factor_ordinal.has_value() and number_of_contractions != 0uz;
                }

                [[nodiscard]] constexpr bool case_C() const {
                    return not one_node_factor_ordinal.has_value();
                }
            };

            static constexpr auto connected_component_infos = std::invoke([&] {
                const auto [id_vec, net] = network();
                const auto cc_vec        = net.connected_components();

                auto arr = std::array<connected_component_info, number_of_connected_components>{};

                for (const auto& [i, cc] : cc_vec | rv::enumerate) {
                    const auto pcs = cc.pairwise_contraction_sequence(dimension);

                    const auto one_node = cc.size() == 1uz;
                    const auto n        = one_node ? rn::size(cc.view_edges()) : rn::size(pcs);
                    auto fac            = std::optional<std::size_t>{};
                    if (one_node) { fac = alg::argfind(id_vec, cc.view_nodes()[0].id); }

                    if (one_node and not rn::empty(pcs)) {
                        throw std::logic_error{ "One node connected component should have"
                                                "empty pairwise contraction sequence" };
                    }

                    arr[i] = connected_component_info{ .rank                    = cc.rank(),
                                                       .number_of_contractions  = n,
                                                       .one_node_factor_ordinal = fac };
                }

                return arr;
            });

            auto connected_component_out_buffs = std::invoke(
                [&]<std::size_t... I>(std::index_sequence<I...>) {
                    return std::tuple{
                        std::array<typename OutMDS::element_type,
                                   connected_component_infos[I].out_buff_size(dimension)>{}...
                    };
                },
                std::make_index_sequence<number_of_connected_components>());

            const auto connected_component_out_mdspans = std::invoke(
                [&]<std::size_t... I>(std::index_sequence<I...>) {
                    if constexpr (number_of_connected_components == 1uz) {
                        return std::tuple{ out };
                    } else {
                        auto deduce_out_mds = [&]<std::size_t N>() {
                            if constexpr (connected_component_infos[N].case_A()) {
                                return std::get<
                                    connected_component_infos[N].one_node_factor_ordinal.value()>(
                                    std::forward_as_tuple(factors...));
                            } else {
                                return sstd::geometric_mdspan<typename OutMDS::element_type,
                                                              connected_component_infos[N].rank,
                                                              dimension>(
                                    std::get<N>(connected_component_out_buffs).data());
                            }
                        };

                        return std::tuple { deduce_out_mds.template operator()<I>()... };
                    }
                },
                std::make_index_sequence<number_of_connected_components>());

            auto handle_connected_component = [&]<std::size_t N>() {
                static constexpr auto info = connected_component_infos[N];

                if constexpr (info.case_A()) {
                    return;
                } else if constexpr (info.case_B()) {
                    static constexpr auto einsum_str = std::invoke([] {
                        const auto p = parser();
                        auto s       = std::u8string{};
                        for (const auto& label : *rn::next(rn::begin(p.factor_index_labels()), N)) {
                            s += label;
                        }
                        return str::fixed_string{ s };
                    });

                    einsum<einsum_str>{}(std::get<N>(connected_component_out_mdspans),
                                         std::get<info.one_node_factor_ordinal.value()>(
                                             std::forward_as_tuple(factors...)));

                    return;
                } else {
                    struct pairwise_contraction_info {
                        std::size_t out_register, out_register_rank, lhs_register, rhs_register;
                        str::fixed_string einsum_str{ u8"to be replaced" };
                    };

                    static constexpr auto pairwise_contractions = std::invoke([] {
                        auto arr =
                            std::array<pairwise_contraction_info, info.number_of_contractions>{};
                        if constexpr (info.number_of_contractions != 0uz) {
                            auto [id_vec, net] = network();
                            const auto cc      = net.connected_components()[N];

                            if (rn::size(id_vec) < 3uz) {
                                throw std::logic_error{ "There should be at least 3 nodes." };
                            }

                            for (const auto [n, c] :
                                 cc.pairwise_contraction_sequence(dimension) | rv::enumerate) {
                                // These should always be found.
                                const auto lhs_reg = alg::argfind(id_vec, c.lhs_id()).value();
                                const auto rhs_reg = alg::argfind(id_vec, c.rhs_id()).value();

                                id_vec.push_back(c.out_id());
                                const auto out_reg = rn::size(id_vec) - 1uz;

                                const auto [lhs_str, rhs_str] = c.index_labels();

                                const auto s = lhs_str + u8"," + rhs_str;
                                arr[n]       = { .out_register      = out_reg,
                                                 .out_register_rank = c.out_rank(),
                                                 .lhs_register      = lhs_reg,
                                                 .rhs_register      = rhs_reg,
                                                 .einsum_str        = str::fixed_string(s) };
                            }
                        }
                        return arr;
                    });

                    auto tensor_register_buffs = std::invoke(
                        [&]<std::size_t... I>(std::index_sequence<I...>) {
                            return std::tuple{
                                std::array<typename OutMDS::element_type,
                                           sstd::integer_pow(
                                               dimension,
                                               pairwise_contractions[I].out_register_rank)>{}...
                            };
                        },
                        std::make_index_sequence<info.number_of_contractions - 1uz>());

                    const auto tensor_register_mdspans = std::invoke([&] {
                        return std::tuple_cat(
                            std::forward_as_tuple(factors...),
                            std::invoke(
                                [&]<std::size_t... I>(std::index_sequence<I...>) {
                                    return std::tuple{ sstd::geometric_mdspan<
                                        typename OutMDS::element_type,
                                        pairwise_contractions[I].out_register_rank,
                                        dimension>(std::get<I>(tensor_register_buffs).data())... };
                                },
                                std::make_index_sequence<info.number_of_contractions - 1uz>()),
                            std::tuple{ std::get<N>(connected_component_out_mdspans) });
                    });

                    std::invoke(
                        [&]<std::size_t... I>(std::index_sequence<I...>) {
                            (einsum<pairwise_contractions[I].einsum_str>{}(
                                 std::get<pairwise_contractions[I].out_register>(
                                     tensor_register_mdspans),
                                 std::get<pairwise_contractions[I].lhs_register>(
                                     tensor_register_mdspans),
                                 std::get<pairwise_contractions[I].rhs_register>(
                                     tensor_register_mdspans)),
                             ...);
                        },
                        std::make_index_sequence<info.number_of_contractions>());
                }
            };

            std::invoke(
                [&]<size_t... I>(std::index_sequence<I...>) {
                    (handle_connected_component.template operator()<I>(), ...);
                },
                std::make_index_sequence<number_of_connected_components>());

            // What is left is to outer product connected components together.
            if constexpr (number_of_connected_components > 1uz) {
                static constexpr auto connected_components_estr = std::invoke([] {
                    const auto p                 = parser();
                    const auto free_index_labels = p.free_index_labels();

                    auto str        = std::u8string{};
                    auto next_label = 0uz;

                    // gcc 14 gives goto is not a constant expression error??
                    // for (const auto i : rv::iota(0uz, number_of_connected_components)) {
                    for (auto i = 0uz; i < number_of_connected_components; ++i) {
                        for (const auto _ : rv::iota(0uz, connected_component_infos[i].rank)) {
                            str += free_index_labels[next_label++];
                        }
                        if (i != number_of_connected_components - 1uz) { str += u8','; }
                    }

                    str += u8"->";
                    for (const auto& c : p.output_index_labels()) { str += c; }

                    return str::fixed_string{ str };
                });

                std::apply(einsum<connected_components_estr>{},
                           std::tuple_cat(std::tuple{ out }, connected_component_out_mdspans));
            }
        }
    }
};

namespace literals {
template<str::fixed_string estr>
[[nodiscard]] constexpr auto operator""_einsum() -> einsum<estr> {
    return {};
};
} // namespace literals

} // namespace idg
