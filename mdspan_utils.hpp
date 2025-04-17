#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include <experimental/mdspan>

#include "idg/sstd.hpp"

template<std::size_t rank, std::size_t D, std::invocable<std::array<std::size_t, rank>> F>
static constexpr auto
constant_geometric_mdspan(F&& f) {
    constexpr static auto buff = [&] {
        using T  = std::invoke_result_t<F, std::array<std::size_t, rank>>;
        auto arr = std::array<T, idg::sstd::integer_pow(D, rank)>{};
        auto mds = idg::sstd::geometric_mdspan<T, rank, D>(arr.data());
        for (const auto idx : idg::sstd::geometric_index_space<rank, D>()) {
            mds[idx] = std::invoke(std::forward<F>(f), idx);
        }
        return arr;
    }();
    return idg::sstd::geometric_mdspan<const typename decltype(buff)::value_type, rank, D>(
        buff.data());
}

template<std::size_t rank, std::size_t D, auto c>
static constexpr auto
constant_geometric_mdspan() {
    return constant_geometric_mdspan<rank, D>([](auto) { return c; });
}
